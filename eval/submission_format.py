"""Unified prediction-trace storage and loading for benchmark baselines.

Every baseline writes its predictions through this module so that the
evaluation code is baseline-agnostic.

Layout on disk:

    predictions/<run-name>/
    ├── manifest.json
    └── traces/
        └── <task_id>.npz

Each per-task NPZ holds four float32 arrays of length T (T frames at the
declared frame rate):

    eot_score_speaker_1
    eot_score_speaker_2
    interruption_score_speaker_1
    interruption_score_speaker_2

Scores are continuous (probabilities or logits). The eval module derives
binary detections by thresholding, which lets us sweep the threshold for
the latency-vs-interruption-rate curve.

The manifest carries metadata that is constant across the run:

    {
        "schema_version": "1.0",
        "run_name": ...,
        "baseline": ...,            # e.g., "sesame"
        "checkpoint": ...,          # optional, free-form
        "frame_rate_hz": 12.5,      # rate at which the per-frame arrays are sampled
        "lookahead_ms": 0,          # declared lookahead L per the protocol
        "split": "dev",
        "task_ids": [...],          # task_ids covered by this run
        "extra": {...},             # baseline-specific notes
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "1.0"
REQUIRED_ARRAYS = (
    "eot_score_speaker_1",
    "eot_score_speaker_2",
    "interruption_score_speaker_1",
    "interruption_score_speaker_2",
)


@dataclass
class Manifest:
    run_name: str
    baseline: str
    frame_rate_hz: float
    split: str
    task_ids: list[str]
    checkpoint: str = ""
    lookahead_ms: float = 0.0
    schema_version: str = SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)


def _traces_dir(run_dir: Path) -> Path:
    return run_dir / "traces"


def write_traces(
    run_dir: Path,
    task_id: str,
    *,
    eot_score_speaker_1: np.ndarray,
    eot_score_speaker_2: np.ndarray,
    interruption_score_speaker_1: np.ndarray,
    interruption_score_speaker_2: np.ndarray,
) -> Path:
    """Persist the four per-speaker score arrays for a single task_id.

    All four arrays must be 1D float32 of the same length. The file is written
    compressed to keep on-disk size small.
    """
    arrays = {
        "eot_score_speaker_1": np.asarray(eot_score_speaker_1, dtype=np.float32),
        "eot_score_speaker_2": np.asarray(eot_score_speaker_2, dtype=np.float32),
        "interruption_score_speaker_1": np.asarray(interruption_score_speaker_1, dtype=np.float32),
        "interruption_score_speaker_2": np.asarray(interruption_score_speaker_2, dtype=np.float32),
    }
    lengths = {k: len(v) for k, v in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"All trace arrays must have equal length; got {lengths}")
    out_dir = _traces_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task_id}.npz"
    np.savez_compressed(path, **arrays)
    return path


def load_traces(run_dir: Path, task_id: str) -> dict[str, np.ndarray]:
    """Load the four trace arrays for a task_id. Returns a dict keyed by the
    canonical array names. Raises FileNotFoundError if the file is missing,
    ValueError if any required array is absent or lengths disagree."""
    path = _traces_dir(run_dir) / f"{task_id}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        out = {name: np.asarray(data[name], dtype=np.float32) for name in REQUIRED_ARRAYS}
    lengths = {k: len(v) for k, v in out.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Trace arrays in {path} have mismatched lengths: {lengths}")
    return out


def write_manifest(run_dir: Path, manifest: Manifest) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(asdict(manifest), indent=2))
    return path


def load_manifest(run_dir: Path) -> Manifest:
    data = json.loads((run_dir / "manifest.json").read_text())
    data.setdefault("extra", {})
    return Manifest(**data)
