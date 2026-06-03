#!/usr/bin/env python3
"""Voice Activity Projection (VAP).

Two-stream GPT-like transformer over CPC features. Takes stereo audio and
produces continuous floor-state probabilities at 50Hz.

Inference uses step_extraction() from VoiceActivityProjection/run.py:
  overlapping windows (context=20s, step=5s) for long audio.

Model: VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt
Frame rate: 50Hz

Setup:
    git submodule update --init baselines/vap/VoiceActivityProjection
    cd baselines/vap/VoiceActivityProjection && pip install -e . && cd ../../..
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

_HERE = Path(__file__).resolve().parent
_VAP_REPO = _HERE / "VoiceActivityProjection"
sys.path.insert(0, str(_VAP_REPO))

from vap.model import VapGPT, VapConfig  # noqa: E402
from run import step_extraction           # noqa: E402

sys.path.insert(0, str(_HERE.parent))
from runner import run  # noqa: E402

CHECKPOINT = _VAP_REPO / "example" / "VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt"
CONTEXT_S  = 20
STEP_S     = 5

_model = None


def _load_model(device: str):
    global _model
    if _model is not None:
        return
    print("Loading VAP model...", flush=True)
    conf  = VapConfig()
    _model = VapGPT(conf)
    sd    = torch.load(str(CHECKPOINT), map_location=device)
    _model.load_state_dict(sd)
    _model = _model.to(device).eval()
    print("Model loaded.", flush=True)


def predict_scores(sample_dir: Path) -> dict:
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _load_model(device)

    target_sr = _model.sample_rate

    def load_wav(path):
        wav, sr = sf.read(path, dtype="float32")
        t = torch.from_numpy(wav)
        if sr != target_sr:
            t = torchaudio.functional.resample(t, sr, target_sr)
        return t  # (T,)

    wav1 = load_wav(sample_dir / "speaker_1_audio.wav")
    wav2 = load_wav(sample_dir / "speaker_2_audio.wav")

    T = min(len(wav1), len(wav2))
    # stack as stereo (1, 2, T)
    waveform = torch.stack([wav1[:T], wav2[:T]], dim=0).unsqueeze(0)

    with torch.no_grad():
        out = step_extraction(
            waveform, _model, device=device,
            context_time=CONTEXT_S, step_time=STEP_S, pbar=True,
        )

    # p_now: (1, frames, 2) — P(speaker holds floor right now)
    # p_future: (1, frames, 2) — P(speaker holds floor in near future)
    p_now    = out["p_now"][0].cpu().numpy()     # (frames, 2)
    p_future = out["p_future"][0].cpu().numpy()  # (frames, 2)

    # EOT: high score = speaker's turn is ending → use 1 - p_future
    # Interruption: high score = speaker barging in → use p_now rising
    eot1 = (1.0 - p_future[:, 0]).astype(np.float32)
    eot2 = (1.0 - p_future[:, 1]).astype(np.float32)
    int1 = p_now[:, 0].astype(np.float32)
    int2 = p_now[:, 1].astype(np.float32)

    duration = T / target_sr
    elapsed  = time.time() - t0
    print(f"{sample_dir.name}: {duration:.1f}s audio → {elapsed:.1f}s ({duration/elapsed:.1f}x RT)", flush=True)

    return {
        "frame_rate_hz":                _model.frame_hz,
        "eot_score_speaker_1":          eot1,
        "eot_score_speaker_2":          eot2,
        "interruption_score_speaker_1": int1,
        "interruption_score_speaker_2": int2,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",    default=None)
    parser.add_argument("--run-name", default="vap")
    args = parser.parse_args()

    split_file = Path(args.split) if args.split else None
    sys.exit(run("vap", predict_scores, run_name=args.run_name, split_file=split_file))
