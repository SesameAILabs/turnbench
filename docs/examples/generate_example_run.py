#!/usr/bin/env python3
"""Generate a minimal example run conforming to the unified submission format.

The output is synthetic (no dataset content) — useful for collaborators
implementing a baseline to see exactly what a valid `predictions/<run>/`
tree looks like end-to-end.

Run from the repo root:

    python3 docs/examples/generate_example_run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
from eval.submission_format import Manifest, write_manifest, write_traces  # noqa: E402


def make_synthetic_traces(n_frames: int, seed: int) -> dict[str, np.ndarray]:
    """Synthetic per-frame scores. EOT is a slow noisy ramp peaking near the
    end; Interruption is a sparse bump in the middle. No real audio used."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)

    eot_1 = np.clip(0.4 * x + 0.1 * rng.standard_normal(n_frames).astype(np.float32) + 0.05, 0, 1)
    eot_2 = np.clip(0.4 * (1 - x) + 0.1 * rng.standard_normal(n_frames).astype(np.float32) + 0.05, 0, 1)

    onset = np.zeros(n_frames, dtype=np.float32)
    peak = n_frames // 2
    span = max(1, n_frames // 20)
    bump = np.exp(-((np.arange(n_frames) - peak) ** 2) / (2 * span ** 2)).astype(np.float32)
    int_1 = np.clip(bump + 0.05 * rng.standard_normal(n_frames).astype(np.float32), 0, 1)
    int_2 = np.clip(0.05 * rng.standard_normal(n_frames).astype(np.float32).clip(min=0), 0, 1)

    return {
        "eot_score_speaker_1": eot_1,
        "eot_score_speaker_2": eot_2,
        "interruption_score_speaker_1": int_1,
        "interruption_score_speaker_2": int_2,
    }


def main() -> int:
    out_dir = _REPO / "docs" / "examples" / "example_run"
    if out_dir.exists():
        for f in (out_dir / "traces").glob("*.npz"):
            f.unlink()
        for f in out_dir.glob("manifest.json"):
            f.unlink()
    frame_rate_hz = 12.5
    task_ids = ["EXAMPLE_001", "EXAMPLE_002"]
    for i, tid in enumerate(task_ids):
        n_frames = 150 + i * 50  # ~12-16 s of audio
        arrays = make_synthetic_traces(n_frames, seed=42 + i)
        write_traces(out_dir, tid, **arrays)
    write_manifest(out_dir, Manifest(
        run_name="example_run",
        baseline="example",
        frame_rate_hz=frame_rate_hz,
        split="example",
        task_ids=task_ids,
        checkpoint="synthetic",
        lookahead_ms=0.0,
        extra={
            "purpose": "synthetic example to illustrate the submission layout",
            "head_mapping": {"eot_score": "<n/a>", "interruption_score": "<n/a>"},
        },
    ))
    print(f"Wrote example run with {len(task_ids)} synthetic tasks to {out_dir}")
    print(f"  manifest: {out_dir / 'manifest.json'}")
    for tid in task_ids:
        print(f"  trace:    {out_dir / 'traces' / (tid + '.npz')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
