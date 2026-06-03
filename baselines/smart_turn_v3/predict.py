#!/usr/bin/env python3
"""Smart Turn v3 — Whisper-Tiny + linear classifier for end-of-turn detection.

Uses the VAD+accumulate+predict pipeline from smart_turn/record_and_predict.py:
  - SileroVAD (ONNX, stateful) on 512-sample chunks at 16kHz
  - Pre-speech buffer (200ms), segment accumulation (up to 8s)
  - On 1s silence or 8s max, Smart Turn runs on the accumulated segment
  - Probability is forward-filled at 12.5Hz for the benchmark format

Model: pipecat-ai/smart-turn-v3 (~8M params, ONNX)
Frame rate: 12.5Hz

Setup:
    git submodule update --init baselines/smart_turn_v3/smart_turn
    pip install onnxruntime-gpu transformers torch torchaudio soundfile huggingface_hub
"""
from __future__ import annotations

import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import hf_hub_download

_HERE = Path(__file__).resolve().parent
_SMART_TURN = _HERE / "smart_turn"

# Download ONNX weights into smart_turn/ before importing inference.py,
# which loads the model at module level from a relative path.
HF_REPO   = "pipecat-ai/smart-turn-v3"
ONNX_FILE = "smart-turn-v3.1.onnx"
_onnx_dst = _SMART_TURN / ONNX_FILE
if not _onnx_dst.exists():
    print("Downloading Smart Turn v3 ONNX weights...", flush=True)
    _src = hf_hub_download(HF_REPO, ONNX_FILE)
    os.symlink(_src, _onnx_dst)

# Now inference.py can find the model at its expected relative path.
os.chdir(_SMART_TURN)
sys.path.insert(0, str(_SMART_TURN))
from inference import predict_endpoint          # noqa: E402
from record_and_predict import SileroVAD, ensure_model  # noqa: E402

sys.path.insert(0, str(_HERE.parent))
from runner import run  # noqa: E402

SR         = 16000
CHUNK      = 512
OUT_HZ     = 12.5
VAD_THRESH = 0.5
PRE_SPEECH_MS = 200
STOP_MS    = 1000
MAX_SEG_S  = 8.0

_vad: SileroVAD | None = None


def _load_models():
    global _vad
    if _vad is not None:
        return
    print("Loading Silero VAD...", flush=True)
    _vad = SileroVAD(ensure_model())
    print("Models loaded.", flush=True)


def _vad_and_predict(wav: np.ndarray) -> np.ndarray:
    n_out = int(len(wav) / SR * OUT_HZ) + 1
    scores = np.zeros(n_out, dtype=np.float32)

    chunk_ms    = CHUNK / SR * 1000.0
    pre_chunks  = math.ceil(PRE_SPEECH_MS / chunk_ms)
    stop_chunks = math.ceil(STOP_MS / chunk_ms)
    max_chunks  = math.ceil(MAX_SEG_S / (CHUNK / SR))

    pre_buffer      = deque(maxlen=pre_chunks)
    segment         = []
    speech_active   = False
    trailing_silence = 0
    since_trigger   = 0

    _vad._init_states()

    for i in range(0, len(wav) - CHUNK + 1, CHUNK):
        chunk = wav[i:i + CHUNK].astype(np.float32)
        is_speech = _vad.prob(chunk) > VAD_THRESH

        if not speech_active:
            pre_buffer.append(chunk)
            if is_speech:
                segment = list(pre_buffer)
                segment.append(chunk)
                speech_active = True
                trailing_silence = 0
                since_trigger = 1
        else:
            segment.append(chunk)
            since_trigger += 1
            trailing_silence = 0 if is_speech else trailing_silence + 1

            if trailing_silence >= stop_chunks or since_trigger >= max_chunks:
                prob = predict_endpoint(np.concatenate(segment, dtype=np.float32))["probability"]
                out_idx = min(int(i / SR * OUT_HZ), n_out - 1)
                scores[out_idx] = prob
                segment.clear()
                speech_active = False
                trailing_silence = 0
                since_trigger = 0
                pre_buffer.clear()

    # forward-fill between predictions
    last = 0.0
    for i in range(len(scores)):
        if scores[i] > 0:
            last = scores[i]
        scores[i] = last

    return scores


def predict_scores(sample_dir: Path) -> dict:
    t0 = time.time()
    _load_models()

    def load_wav(path):
        wav, sr = sf.read(path, dtype="float32")
        if sr != SR:
            wav = torchaudio.functional.resample(
                torch.from_numpy(wav), sr, SR
            ).numpy()
        return wav

    wav1 = load_wav(sample_dir / "speaker_1_audio.wav")
    wav2 = load_wav(sample_dir / "speaker_2_audio.wav")

    scores1 = _vad_and_predict(wav1)
    scores2 = _vad_and_predict(wav2)

    T = min(len(scores1), len(scores2))

    duration = len(wav1) / SR
    elapsed  = time.time() - t0
    print(f"{sample_dir.name}: {duration:.1f}s audio → {elapsed:.1f}s ({duration/elapsed:.1f}x RT)", flush=True)

    return {
        "frame_rate_hz":                OUT_HZ,
        "eot_score_speaker_1":          scores1[:T],
        "eot_score_speaker_2":          scores2[:T],
        "interruption_score_speaker_1": scores1[:T],
        "interruption_score_speaker_2": scores2[:T],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",    default=None)
    parser.add_argument("--run-name", default="smart_turn_v3")
    args = parser.parse_args()

    split_file = Path(args.split) if args.split else None
    sys.exit(run("smart_turn_v3", predict_scores, run_name=args.run_name, split_file=split_file))
