#!/usr/bin/env python3
"""ESPnet Turn-Taking Prediction (Switchboard).

Trained using the Switchboard recipe in ESPnet. Built on a frozen
Whisper-medium encoder (~306M params) producing 1024-dim features at
50 Hz, with a small trainable classification head (~1M params):
    Tanh -> Linear(1024->1024) -> Linear(1024->5)

Trained and evaluated as part of "Talking Turns: Benchmarking Audio
Foundation Models on Turn-Taking Dynamics" (Arora et al., ICLR 2025) —
the closest prior work to this benchmark.

Input:  single-channel 16 kHz audio, up to 30 s of left context.
Output: 5-class probability distribution per 40 ms chunk (25 Hz):
        {Continuation, Silence, Interruption, Backchannel, Turn-change}.

Evaluation: per-class one-vs-rest ROC-AUC; predictions are NOT thresholded
to binary.

Reads the dataset root from `TT_BENCHMARK_DATA` (see top-level
`.env.example`). Emits predictions to
`predictions/espnet_turntaking/traces/<task_id>.npz` in the unified submission format
(see `eval/submission_format.py`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runner import run  # noqa: E402


def predict_scores(sample_dir: Path) -> dict:
    """Return per-frame score arrays for one conversation in the unified
    submission format. Must include `frame_rate_hz` and four
    `eot_score_speaker_{1,2}` / `interruption_score_speaker_{1,2}`
    float arrays of equal length."""
    raise NotImplementedError("TODO: implement espnet_turntaking")


if __name__ == "__main__":
    sys.exit(run("espnet_turntaking", predict_scores))
