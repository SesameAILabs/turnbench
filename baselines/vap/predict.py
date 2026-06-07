#!/usr/bin/env python3
"""Voice Activity Projection (VAP).

Two-stream GPT-like transformer over CPC features. Takes stereo audio and
produces continuous floor-state probabilities at 50Hz.

Inference: overlapping windows (context=20s, step=5s) simulating the bounded
context a real-time system would have. Not truly stateful — the transformer
runs on each 25s window independently; only the new step_frames are kept.

Score mapping:
  p_now[:, spk] = P(speaker spk holds floor in next 0-0.4s)
  eot  = 1 - p_now[:, spk]  (speaker relinquishing floor)
  int  = p_now[:, spk]      (speaker taking/holding floor)

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


def step_extraction(waveform, model, device="cpu", context_time=20, step_time=5, pbar=True):
    """Overlapping-window inference for long audio. Copied from VoiceActivityProjection/run.py
    to avoid its plot_utils → parselmouth dependency. Only p_now is kept."""
    step_samples    = int(step_time * model.sample_rate)
    chunk_samples   = int((context_time + step_time) * model.sample_rate)
    step_frames     = int(step_time * model.frame_hz)
    n_samples       = waveform.shape[-1]
    expected_frames = round(n_samples / model.sample_rate * model.frame_hz)

    folds = waveform.unfold(-1, chunk_samples, step_samples).permute(2, 0, 1, 3)
    p_now = model.probs(folds[0].to(device))["p_now"]

    iterator = folds[1:]
    if pbar:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"VAP context={context_time}s step={step_time}s")
    for w in iterator:
        p_now = torch.cat([p_now, model.probs(w.to(device))["p_now"][:, -step_frames:]], dim=1)

    if expected_frames != p_now.shape[1]:
        omitted = expected_frames - p_now.shape[1]
        p_now = torch.cat([p_now, model.probs(waveform[..., -chunk_samples:].to(device))["p_now"][:, -omitted:]], dim=1)

    return p_now.cpu()  # (1, frames, 2)

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
        p_now = step_extraction(
            waveform, _model, device=device,
            context_time=CONTEXT_S, step_time=STEP_S, pbar=True,
        )

    # p_now[:, spk] = P(speaker spk holds floor in next 0-0.4s)
    # EOT: speaker relinquishing floor → 1 - p_now[:, spk]
    # Interruption: speaker taking/holding floor → p_now[:, spk]
    p_now = p_now[0].numpy()   # (frames, 2)

    eot1 = (1.0 - p_now[:, 0]).astype(np.float32)
    eot2 = (1.0 - p_now[:, 1]).astype(np.float32)
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


def _inspect(sample_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = predict_scores(sample_dir)
    hz     = scores["frame_rate_hz"]
    eot1   = scores["eot_score_speaker_1"]
    eot2   = scores["eot_score_speaker_2"]
    int1   = scores["interruption_score_speaker_1"]
    int2   = scores["interruption_score_speaker_2"]

    T        = min(len(eot1), len(eot2))
    t        = np.arange(T) / hz
    duration = t[-1]
    vlines   = np.arange(1, int(duration) + 1)
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
    fig.suptitle(f"vap  task={sample_dir.name}  {hz}Hz", fontsize=10)
    out = Path(__file__).parent / f"inspect_{sample_dir.name}.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(f"Saved → {out}")
    for label, arr in zip(labels, arrays):
        print(f"  {label:<20s}  min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",    default=None)
    parser.add_argument("--run-name", default="vap")
    parser.add_argument("--inspect",  default=None, help="Inspect single sample dir and plot")
    args = parser.parse_args()

    if args.inspect:
        _inspect(Path(args.inspect))
    else:
        split_file = Path(args.split) if args.split else None
        sys.exit(run("vap", predict_scores, run_name=args.run_name, split_file=split_file))
