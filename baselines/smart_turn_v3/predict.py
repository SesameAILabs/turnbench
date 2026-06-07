#!/usr/bin/env python3
"""Smart Turn v3 — Whisper-Tiny + linear classifier for end-of-turn detection.

Adapts the VAD+accumulate+predict pipeline from smart_turn/record_and_predict.py
for offline multi-turn evaluation of dual-channel audio.

Pipeline (per channel, both processed simultaneously):
  1. Silero VAD (ONNX, stateful): 512-sample chunks at 16kHz (32ms)
  2. Speech accumulated in a rolling 8s buffer (oldest chunks dropped when full)
  3. On 1s trailing silence: classify, then enter a 2s settling window where the
     classifier re-runs each chunk — P(complete) can rise as silence grows
  4. Speech during settling resets score and starts a new segment
  5. Scores forward-filled at 12.5Hz

Model: pipecat-ai/smart-turn-v3 (~8M params, ONNX fp32 GPU)

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
ONNX_FILE = "smart-turn-v3.1-gpu.onnx"
# inference.py hardcodes "smart-turn-v3.1.onnx" at module level
_onnx_dst = _SMART_TURN / "smart-turn-v3.1.onnx"
if not _onnx_dst.exists():
    print("Downloading Smart Turn v3 ONNX weights...", flush=True)
    _src = hf_hub_download(HF_REPO, ONNX_FILE)
    os.symlink(_src, _onnx_dst)

# Now inference.py can find the model at its expected relative path.
os.chdir(_SMART_TURN)
sys.path.insert(0, str(_SMART_TURN))
from inference import predict_endpoint          # noqa: E402
# record_and_predict imports pyaudio at module level (for live mic recording).
# Stub it out — SileroVAD and ensure_model don't use it.
import types as _types
_pyaudio_stub = _types.ModuleType("pyaudio")
_pyaudio_stub.paInt16 = 0
sys.modules.setdefault("pyaudio", _pyaudio_stub)
from record_and_predict import SileroVAD, ensure_model  # noqa: E402

sys.path.insert(0, str(_HERE.parent))
from runner import run  # noqa: E402

SR         = 16000
CHUNK      = 512
OUT_HZ     = 12.5
VAD_THRESH = 0.5
PRE_SPEECH_MS = 200
STOP_MS    = 1000

_silero_model_path: str | None = None

_CHUNK_MS      = CHUNK / SR * 1000.0
_PRE_CHUNKS    = math.ceil(PRE_SPEECH_MS / _CHUNK_MS)
_STOP_CHUNKS   = math.ceil(STOP_MS / _CHUNK_MS)
_SETTLE_CHUNKS = math.ceil(2000.0 / _CHUNK_MS)   # 2s settling window after 1s silence
_MAX_CHUNKS    = math.ceil(8000.0 / _CHUNK_MS)    # rolling 8s cap on seg during accumulation


def _load_models():
    global _silero_model_path
    if _silero_model_path is not None:
        return
    print("Loading Silero VAD...", flush=True)
    _silero_model_path = ensure_model()
    print("Models loaded.", flush=True)


def _vad_and_predict_joint(
    wav1: np.ndarray, wav2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate real-time streaming: process both speaker channels chunk by chunk
    simultaneously, maintaining independent VAD+accumulate state for each.
    Returns (scores1, scores2, vad_probs1, vad_probs2)."""
    n_out1 = int(len(wav1) / SR * OUT_HZ) + 1
    n_out2 = int(len(wav2) / SR * OUT_HZ) + 1
    scores1 = np.zeros(n_out1, dtype=np.float32)
    scores2 = np.zeros(n_out2, dtype=np.float32)
    vad_probs1 = np.zeros(n_out1, dtype=np.float32)
    vad_probs2 = np.zeros(n_out2, dtype=np.float32)

    vad1 = SileroVAD(_silero_model_path)
    vad2 = SileroVAD(_silero_model_path)

    pre1, pre2 = deque(maxlen=_PRE_CHUNKS), deque(maxlen=_PRE_CHUNKS)
    seg1, seg2 = [], []
    active1 = active2 = False
    sil1 = sil2 = 0
    extra1 = extra2 = 0  # settling window counter (0 = not settling, 1-10 = settling)
    cur1 = cur2 = 0.0

    T = max(len(wav1), len(wav2))
    wav1 = np.pad(wav1, (0, max(0, T - len(wav1))))
    wav2 = np.pad(wav2, (0, max(0, T - len(wav2))))

    for i in range(0, T - CHUNK + 1, CHUNK):
        c1 = wav1[i:i + CHUNK].astype(np.float32)
        c2 = wav2[i:i + CHUNK].astype(np.float32)
        p1 = vad1.prob(c1)
        p2 = vad2.prob(c2)
        sp1 = p1 > VAD_THRESH
        sp2 = p2 > VAD_THRESH
        out_idx = int(i / SR * OUT_HZ)

        vad_probs1[min(out_idx, n_out1 - 1)] = p1
        vad_probs2[min(out_idx, n_out2 - 1)] = p2

        # --- channel 1 ---
        if not active1:
            pre1.append(c1)
            if sp1:
                cur1 = 0.0
                seg1 = list(pre1); seg1.append(c1)
                active1 = True; sil1 = 0; extra1 = 0
        elif extra1 == 0:                         # active, accumulating
            seg1.append(c1)
            if len(seg1) > _MAX_CHUNKS:           # rolling 8s window — drop oldest
                seg1.pop(0)
            sil1 = 0 if sp1 else sil1 + 1
            if sil1 >= _STOP_CHUNKS:
                extra1 = 1
                audio = np.concatenate(seg1, dtype=np.float32)
                cur1 = predict_endpoint(audio)["probability"]
                print(f"  spk1 settle@1  t={i/SR:.1f}s  seg={len(audio)/SR:.2f}s  P={cur1:.3f}", flush=True)
        else:                                     # settling window
            if sp1:
                cur1 = 0.0
                seg1 = [c1]; active1 = True; sil1 = 0; extra1 = 0
            else:
                seg1.append(c1); extra1 += 1
                audio = np.concatenate(seg1, dtype=np.float32)
                cur1 = predict_endpoint(audio)["probability"]
                print(f"  spk1 settle@{extra1}  t={i/SR:.1f}s  seg={len(audio)/SR:.2f}s  P={cur1:.3f}", flush=True)
                if extra1 >= _SETTLE_CHUNKS:
                    active1 = False; sil1 = 0; extra1 = 0; pre1.clear()
        scores1[min(out_idx, n_out1 - 1)] = cur1

        # --- channel 2 ---
        if not active2:
            pre2.append(c2)
            if sp2:
                cur2 = 0.0
                seg2 = list(pre2); seg2.append(c2)
                active2 = True; sil2 = 0; extra2 = 0
        elif extra2 == 0:                         # active, accumulating
            seg2.append(c2)
            if len(seg2) > _MAX_CHUNKS:           # rolling 8s window — drop oldest
                seg2.pop(0)
            sil2 = 0 if sp2 else sil2 + 1
            if sil2 >= _STOP_CHUNKS:
                extra2 = 1
                audio = np.concatenate(seg2, dtype=np.float32)
                cur2 = predict_endpoint(audio)["probability"]
                print(f"  spk2 settle@1  t={i/SR:.1f}s  seg={len(audio)/SR:.2f}s  P={cur2:.3f}", flush=True)
        else:                                     # settling window
            if sp2:
                cur2 = 0.0
                seg2 = [c2]; active2 = True; sil2 = 0; extra2 = 0
            else:
                seg2.append(c2); extra2 += 1
                audio = np.concatenate(seg2, dtype=np.float32)
                cur2 = predict_endpoint(audio)["probability"]
                print(f"  spk2 settle@{extra2}  t={i/SR:.1f}s  seg={len(audio)/SR:.2f}s  P={cur2:.3f}", flush=True)
                if extra2 >= _SETTLE_CHUNKS:
                    active2 = False; sil2 = 0; extra2 = 0; pre2.clear()
        scores2[min(out_idx, n_out2 - 1)] = cur2

    return scores1, scores2, vad_probs1, vad_probs2


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

    scores1, scores2, _, _ = _vad_and_predict_joint(wav1, wav2)

    T = min(len(scores1), len(scores2))

    duration = len(wav1) / SR
    elapsed  = time.time() - t0
    print(f"{sample_dir.name}: {duration:.1f}s audio → {elapsed:.1f}s ({duration/elapsed:.1f}x RT)", flush=True)

    return {
        "frame_rate_hz":                OUT_HZ,
        "eot_score_speaker_1":          scores1[:T],
        "eot_score_speaker_2":          scores2[:T],
        "interruption_score_speaker_1": 1 - scores1[:T],
        "interruption_score_speaker_2": 1 - scores2[:T],
    }


def _inspect(sample_dir: Path) -> None:
    """TEMPORARY: run predict_scores on one sample and plot waveforms + VAD probs + scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as _sf

    _load_models()
    wav1 = np.array([], dtype=np.float32)
    wav2 = np.array([], dtype=np.float32)

    def load_wav(path):
        import torchaudio as _ta, torch as _torch
        w, sr = _sf.read(path, dtype="float32")
        if sr != SR:
            w = _ta.functional.resample(_torch.from_numpy(w), sr, SR).numpy()
        return w

    wav1 = load_wav(sample_dir / "speaker_1_audio.wav")
    wav2 = load_wav(sample_dir / "speaker_2_audio.wav")
    scores1, scores2, vad1, vad2 = _vad_and_predict_joint(wav1, wav2)

    T = min(len(scores1), len(scores2))
    scores1, scores2 = scores1[:T], scores2[:T]
    vad1, vad2 = vad1[:T], vad2[:T]

    hz = OUT_HZ
    t = np.arange(T) / hz
    duration = t[-1]
    vlines = np.arange(1, int(duration) + 1)
    tick_locs = np.arange(0, int(duration) + 1, 5)

    score_keys = ["eot_score_speaker_1", "eot_score_speaker_2",
                  "interruption_score_speaker_1", "interruption_score_speaker_2"]
    score_arrays = [scores1, scores2, 1 - scores1, 1 - scores2]
    score_colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
    score_labels = ["EOT spk1", "EOT spk2", "INT spk1", "INT spk2"]

    fig, axes = plt.subplots(8, 1, figsize=(max(duration / 10, 20), 12),
                             gridspec_kw={"hspace": 0.05})

    # rows 0-1: waveforms
    for spk_idx, ax in enumerate(axes[:2]):
        p = sample_dir / f"speaker_{spk_idx+1}_audio.wav"
        if p.exists():
            wav, sr = _sf.read(p, dtype="float32")
            ax.plot(np.arange(len(wav)) / sr, wav, lw=0.3, color="#555", alpha=0.7)
            peak = float(np.abs(wav).max()) or 1.0
            ax.set_ylim(-peak * 1.1, peak * 1.1)
        for vl in vlines:
            ax.axvline(vl, color="#ccc", lw=0.4, zorder=0)
        ax.set_ylabel(f"Spk {spk_idx+1}\nwav", fontsize=7)
        ax.set_xlim(0, duration); ax.set_xticks(tick_locs); ax.tick_params(labelbottom=False)

    # rows 2-3: Silero VAD probs
    for spk_idx, (vad_arr, ax) in enumerate(zip([vad1, vad2], axes[2:4])):
        ax.plot(t, vad_arr, lw=0.6, color="#9C27B0")
        ax.fill_between(t, vad_arr, alpha=0.15, color="#9C27B0")
        ax.axhline(VAD_THRESH, color="#9C27B0", lw=0.5, ls="--", alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        for vl in vlines:
            ax.axvline(vl, color="#ccc", lw=0.4, zorder=0)
        ax.set_ylabel(f"VAD spk{spk_idx+1}", fontsize=7)
        ax.set_xlim(0, duration); ax.set_xticks(tick_locs); ax.tick_params(labelbottom=False)

    # rows 4-7: EOT/interruption scores
    for i, (arr, label, color, ax) in enumerate(zip(score_arrays, score_labels, score_colors, axes[4:])):
        ax.plot(t, arr, lw=0.6, color=color)
        ax.fill_between(t, arr, alpha=0.15, color=color)
        ax.set_ylim(-0.05, 1.05)
        for vl in vlines:
            ax.axvline(vl, color="#ccc", lw=0.4, zorder=0)
        ax.set_ylabel(label, fontsize=7)
        ax.set_xlim(0, duration); ax.set_xticks(tick_locs)
        if i < 3:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.suptitle(f"smart_turn_v3  task={sample_dir.name}  {hz}Hz", fontsize=10)
    out = Path(__file__).parent / f"inspect_{sample_dir.name}.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(f"Saved → {out}")
    for label, arr in zip(["VAD spk1", "VAD spk2"] + score_labels, [vad1, vad2] + score_arrays):
        print(f"  {label:<40s}  min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",    default=None)
    parser.add_argument("--run-name", default="smart_turn_v3")
    parser.add_argument("--inspect",  default=None, help="Inspect single sample dir and plot")
    args = parser.parse_args()

    if args.inspect:
        _load_models()
        _inspect(Path(args.inspect))
    else:
        split_file = Path(args.split) if args.split else None
        sys.exit(run("smart_turn_v3", predict_scores, run_name=args.run_name, split_file=split_file))
