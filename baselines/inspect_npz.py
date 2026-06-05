#!/usr/bin/env python3
"""Inspect and plot a single NPZ trace from a baseline prediction run.

Usage:
    python3 baselines/inspect_npz.py <run_name> <task_id> [--wav-dir DATA_DIR]

Example:
    python3 baselines/inspect_npz.py mimi_endpointer_dev 20

Loads predictions/<run_name>/traces/<task_id>.npz, prints array stats,
and plots waveforms + all 4 score arrays stacked vertically.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf

_REPO = Path(__file__).resolve().parent.parent
REQUIRED = (
    "eot_score_speaker_1",
    "eot_score_speaker_2",
    "interruption_score_speaker_1",
    "interruption_score_speaker_2",
)
COLORS = {
    "eot_score_speaker_1":          "#2196F3",   # blue
    "eot_score_speaker_2":          "#F44336",   # red
    "interruption_score_speaker_1": "#4CAF50",   # green
    "interruption_score_speaker_2": "#FF9800",   # orange
}


def load_manifest(run_dir: Path) -> dict:
    import json
    return json.loads((run_dir / "manifest.json").read_text())


def load_npz(run_dir: Path, task_id: str) -> dict[str, np.ndarray]:
    path = run_dir / "traces" / f"{task_id}.npz"
    if not path.exists():
        sys.exit(f"NPZ not found: {path}")
    with np.load(path) as f:
        return {k: f[k] for k in f}


def print_stats(arrays: dict[str, np.ndarray], frame_rate_hz: float) -> None:
    n_frames = len(next(iter(arrays.values())))
    duration_s = n_frames / frame_rate_hz
    print(f"\nFrames: {n_frames}  |  Frame rate: {frame_rate_hz} Hz  |  Duration: {duration_s:.1f}s\n")
    for name in REQUIRED:
        arr = arrays[name]
        print(f"  {name:<40s}  min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}  dtype={arr.dtype}")
    print()


def plot(
    arrays: dict[str, np.ndarray],
    frame_rate_hz: float,
    task_id: str,
    wav_dir: Path | None,
    out_path: Path,
) -> None:
    n_frames = len(next(iter(arrays.values())))
    t_scores = np.arange(n_frames) / frame_rate_hz  # time axis for scores

    n_rows = 6  # wav1, wav2, eot1, eot2, int1, int2
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(min(max(t_scores[-1] / 10, 20), 80), 14),
        gridspec_kw={"hspace": 0.08}
    )

    # --- Waveforms ---
    for spk_idx, ax in enumerate(axes[:2]):
        spk = spk_idx + 1
        wav_path = wav_dir / f"speaker_{spk}_audio.wav" if wav_dir else None
        if wav_path and wav_path.exists():
            wav, sr = sf.read(wav_path, dtype="float32")
            t_wav = np.arange(len(wav)) / sr
            ax.plot(t_wav, wav, lw=0.3, color="#555", alpha=0.7)
            ax.set_ylim(-1.05, 1.05)
        else:
            ax.text(0.5, 0.5, f"speaker_{spk}_audio.wav not found",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_ylabel(f"Spk {spk}\nwav", fontsize=8)
        ax.set_xlim(0, t_scores[-1])
        ax.tick_params(labelbottom=False)

    # --- Score arrays ---
    score_keys = [
        "eot_score_speaker_1",
        "eot_score_speaker_2",
        "interruption_score_speaker_1",
        "interruption_score_speaker_2",
    ]
    labels = ["EOT spk1", "EOT spk2", "INT spk1", "INT spk2"]
    for i, (key, label, ax) in enumerate(zip(score_keys, labels, axes[2:])):
        arr = arrays[key]
        ax.plot(t_scores, arr, lw=0.6, color=COLORS[key])
        ax.fill_between(t_scores, arr, alpha=0.15, color=COLORS[key])
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(label, fontsize=8)
        ax.set_xlim(0, t_scores[-1])
        if i < len(score_keys) - 1:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.suptitle(f"task_id={task_id}  |  frame_rate={frame_rate_hz}Hz", fontsize=10)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    print(f"Saved plot → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="e.g. mimi_endpointer_dev")
    parser.add_argument("task_id", help="e.g. 20")
    parser.add_argument("--wav-dir", default=None,
                        help="Path to the task_id folder containing speaker wavs "
                             "(default: $TT_BENCHMARK_DATA/<task_id>)")
    parser.add_argument("--frame-rate", type=float, default=None,
                        help="Frame rate in Hz (required if manifest.json is missing)")
    args = parser.parse_args()

    run_dir = _REPO / "predictions" / args.run_name
    if not run_dir.exists():
        sys.exit(f"Run dir not found: {run_dir}")

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = load_manifest(run_dir)
        frame_rate_hz = float(manifest["frame_rate_hz"])
        print(f"Run: {manifest['run_name']}  baseline: {manifest['baseline']}  "
              f"split: {manifest['split']}  tasks: {len(manifest['task_ids'])}")
    elif args.frame_rate:
        frame_rate_hz = args.frame_rate
        print(f"Run: {args.run_name}  (no manifest — frame_rate={frame_rate_hz})")
    else:
        sys.exit("No manifest.json found — pass --frame-rate <hz>")

    arrays = load_npz(run_dir, args.task_id)
    print_stats(arrays, frame_rate_hz)

    # resolve wav dir
    wav_dir = Path(args.wav_dir) if args.wav_dir else None
    if wav_dir is None:
        env_path = _REPO / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("TT_BENCHMARK_DATA="):
                    wav_dir = Path(line.split("=", 1)[1].strip()) / args.task_id
                    break

    out_path = run_dir / f"inspect_{args.task_id}.png"
    plot(arrays, frame_rate_hz, args.task_id, wav_dir, out_path)


if __name__ == "__main__":
    main()
