#!/usr/bin/env python3
"""DualTurn — Qwen2.5-0.5B + Mimi codec for multi-signal turn-taking prediction.

Dual-channel audio is fed to a Qwen2.5-0.5B backbone with LoRA adapters and
12 per-channel classification heads. The model handles Mimi encoding internally.

Signals used (per channel):
  eot_probs  — high during speech, low at turn end → invert for EOT score
  hold_probs — speaker actively holding floor → proxy for barge-in

Score mapping:
  eot_score_speaker_{1,2}          ← 1 - eot_probs[:, spk]
  interruption_score_speaker_{1,2} ← hold_probs[:, spk]

Model: anyreach-ai/dualturn-qwen2.5-mimi-0.5B (HuggingFace)
Frame rate: 12.5Hz

Setup:
    git submodule update --init baselines/dualturn/dualturn
    pip install transformers torch torchaudio soundfile
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoModel

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from runner import run  # noqa: E402

FRAME_RATE = 12.5
HF_REPO    = "anyreach-ai/dualturn-qwen2.5-mimi-0.5B"

_model = None


def _load_model(device: str):
    global _model
    if _model is not None:
        return
    print("Loading DualTurn model...", flush=True)
    _model = AutoModel.from_pretrained(HF_REPO, trust_remote_code=True)
    _model.to(device).eval()
    print("Model loaded.", flush=True)


def predict_scores(sample_dir: Path) -> dict:
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _load_model(device)

    def load_wav(path):
        wav, sr = sf.read(path, dtype="float32")
        return torch.from_numpy(wav), sr

    wav1, sr1 = load_wav(sample_dir / "speaker_1_audio.wav")
    wav2, sr2 = load_wav(sample_dir / "speaker_2_audio.wav")

    # resample to same sr if needed, then stack as stereo [2, T]
    if sr1 != sr2:
        wav2 = torchaudio.functional.resample(wav2, sr2, sr1)
        sr2 = sr1
    T = min(len(wav1), len(wav2))
    stereo = torch.stack([wav1[:T], wav2[:T]], dim=0)  # [2, T]

    with torch.no_grad():
        out = _model(stereo, sr=sr1)

    eot  = out.eot_probs[0].float().cpu().numpy()   # [T, 2]
    hold = out.hold_probs[0].float().cpu().numpy()  # [T, 2]

    duration = T / sr1
    elapsed  = time.time() - t0
    print(f"{sample_dir.name}: {duration:.1f}s audio → {elapsed:.1f}s ({duration/elapsed:.1f}x RT)", flush=True)

    return {
        "frame_rate_hz":                FRAME_RATE,
        "eot_score_speaker_1":          (1 - eot[:, 0]).astype(np.float32),
        "eot_score_speaker_2":          (1 - eot[:, 1]).astype(np.float32),
        "interruption_score_speaker_1": hold[:, 0].astype(np.float32),
        "interruption_score_speaker_2": hold[:, 1].astype(np.float32),
    }


def _inspect(sample_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores    = predict_scores(sample_dir)
    hz        = scores["frame_rate_hz"]
    eot1      = scores["eot_score_speaker_1"]
    eot2      = scores["eot_score_speaker_2"]
    int1      = scores["interruption_score_speaker_1"]
    int2      = scores["interruption_score_speaker_2"]

    T         = min(len(eot1), len(eot2))
    t         = np.arange(T) / hz
    duration  = t[-1]
    vlines    = np.arange(1, int(duration) + 1)
    tick_locs = np.arange(0, int(duration) + 1, 5)

    arrays = [eot1[:T], eot2[:T], int1[:T], int2[:T]]
    labels = ["EOT spk1", "EOT spk2", "INT spk1", "INT spk2"]
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

    fig, axes = plt.subplots(6, 1, figsize=(max(duration / 10, 20), 10),
                             gridspec_kw={"hspace": 0.05})

    for spk_idx, ax in enumerate(axes[:2]):
        p = sample_dir / f"speaker_{spk_idx + 1}_audio.wav"
        wav, sr = sf.read(p, dtype="float32")
        ax.plot(np.arange(len(wav)) / sr, wav, lw=0.3, color="#555", alpha=0.7)
        peak = float(np.abs(wav).max()) or 1.0
        ax.set_ylim(-peak * 1.1, peak * 1.1)
        for vl in vlines: ax.axvline(vl, color="#ccc", lw=0.4, zorder=0)
        ax.set_ylabel(f"Spk {spk_idx + 1}\nwav", fontsize=7)
        ax.set_xlim(0, duration); ax.set_xticks(tick_locs); ax.tick_params(labelbottom=False)

    for i, (arr, label, color, ax) in enumerate(zip(arrays, labels, colors, axes[2:])):
        ax.plot(t, arr, lw=0.6, color=color)
        ax.fill_between(t, arr, alpha=0.15, color=color)
        ax.set_ylim(-0.05, 1.05)
        for vl in vlines: ax.axvline(vl, color="#ccc", lw=0.4, zorder=0)
        ax.set_ylabel(label, fontsize=7)
        ax.set_xlim(0, duration); ax.set_xticks(tick_locs)
        if i < 3: ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.suptitle(f"dualturn [eot=1-eot_probs, int=hold_probs]  task={sample_dir.name}  {hz}Hz", fontsize=10)
    out = _HERE / f"inspect_{sample_dir.name}.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(f"Saved → {out}")
    for label, arr in zip(labels, arrays):
        print(f"  {label:<20s}  min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",    default=None)
    parser.add_argument("--run-name", default="dualturn")
    parser.add_argument("--inspect",  default=None, help="Inspect single sample dir and plot")
    args = parser.parse_args()

    if args.inspect:
        _inspect(Path(args.inspect))
    else:
        split_file = Path(args.split) if args.split else None
        sys.exit(run("dualturn", predict_scores, run_name=args.run_name, split_file=split_file))
