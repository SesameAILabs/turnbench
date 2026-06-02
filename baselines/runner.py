"""Shared runner for baseline scripts.

Each baseline's `predict.py` implements one function,
`predict_scores(sample_dir) -> dict`, that returns the four per-frame score
arrays for one conversation in the **unified submission format**:

    {
        "frame_rate_hz": <float>,
        "eot_score_speaker_1": np.ndarray,
        "eot_score_speaker_2": np.ndarray,
        "interruption_score_speaker_1": np.ndarray,
        "interruption_score_speaker_2": np.ndarray,
    }

The four arrays are continuous scores (probabilities or logits) at the
declared `frame_rate_hz`. Storing scores rather than thresholded events lets
`eval/metrics.py` sweep the detection threshold for the latency-IR curve.

This module:

  - loads `TT_BENCHMARK_DATA` from the repo's `.env`;
  - iterates over each sample directory;
  - calls `predict_scores` once per sample (the function is responsible for
    handling both speaker directions internally);
  - writes the four trace arrays to
    `predictions/<run-name>/traces/<task_id>.npz` via
    `eval.submission_format.write_traces`;
  - writes a `predictions/<run-name>/manifest.json` describing the run.

A baseline's entry point is typically:

    from baselines.runner import run

    def predict_scores(sample_dir):
        ...  # produce the four trace arrays
        return {
            "frame_rate_hz": 12.5,
            "eot_score_speaker_1": ...,
            "eot_score_speaker_2": ...,
            "interruption_score_speaker_1": ...,
            "interruption_score_speaker_2": ...,
        }

    if __name__ == "__main__":
        run("gemini", predict_scores)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np

# eval/ is a sibling of baselines/. Make it importable.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from eval.submission_format import Manifest, write_manifest, write_traces  # noqa: E402


def load_env(p: Path = _REPO / ".env") -> dict[str, str]:
    env: dict[str, str] = {}
    if not p.exists():
        return env
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def data_root() -> Path:
    env = load_env()
    if "TT_BENCHMARK_DATA" not in env or not env["TT_BENCHMARK_DATA"]:
        sys.exit("Set TT_BENCHMARK_DATA in .env — see .env.example.")
    return Path(env["TT_BENCHMARK_DATA"])


def _task_ids_in_split(split_file: Path | None) -> list[str] | None:
    if split_file is None or not split_file.exists():
        return None
    return [
        line.strip() for line in split_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def run(
    baseline_name: str,
    predict_scores: Callable[[Path], dict],
    *,
    run_name: str | None = None,
    split_file: Path | None = None,
    checkpoint: str = "",
    lookahead_ms: float = 0.0,
) -> int:
    """Drive a baseline end-to-end and persist its unified-format predictions.

    `predict_scores(sample_dir)` returns the dict described in the module
    docstring. `run_name` defaults to `baseline_name`; override when a single
    baseline runs multiple checkpoints. `split_file`, if given, restricts the
    run to the listed task_ids (one per line, like eval/splits/dev.txt)."""
    root = data_root()
    name = run_name or baseline_name
    out_dir = _REPO / "predictions" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    only = set(_task_ids_in_split(split_file) or [])
    sample_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if only:
        sample_dirs = [d for d in sample_dirs if d.name in only]

    task_ids_done: list[str] = []
    frame_rate_hz: float | None = None

    for d in sample_dirs:
        result = predict_scores(d)
        if frame_rate_hz is None:
            frame_rate_hz = float(result["frame_rate_hz"])
        elif float(result["frame_rate_hz"]) != frame_rate_hz:
            sys.exit(
                f"Inconsistent frame_rate_hz across tasks for {name}: "
                f"{frame_rate_hz} vs {result['frame_rate_hz']}"
            )
        write_traces(
            out_dir, d.name,
            eot_score_speaker_1=np.asarray(result["eot_score_speaker_1"]),
            eot_score_speaker_2=np.asarray(result["eot_score_speaker_2"]),
            interruption_score_speaker_1=np.asarray(result["interruption_score_speaker_1"]),
            interruption_score_speaker_2=np.asarray(result["interruption_score_speaker_2"]),
        )
        task_ids_done.append(d.name)

    manifest = Manifest(
        run_name=name,
        baseline=baseline_name,
        frame_rate_hz=frame_rate_hz or 0.0,
        split=(split_file.name if split_file else "all"),
        task_ids=task_ids_done,
        checkpoint=checkpoint,
        lookahead_ms=lookahead_ms,
    )
    write_manifest(out_dir, manifest)
    print(
        f"Wrote {len(task_ids_done)} traces + manifest to {out_dir}",
        file=sys.stderr,
    )
    return 0
