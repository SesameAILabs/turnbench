#!/usr/bin/env python3
"""Moshi (Kyutai).

Full-duplex spoken-language model from Kyutai (Defossez et al., 2024).
A multi-stream Transformer that natively models two audio streams (user
and system) plus a text "inner monologue", operating at 12.5 Hz on top of
Mimi codec tokens.

Because Moshi is generative and continuously predicts whether it should
speak on the system stream, turn-taking events can be read out directly
from the system-stream voice-activity prediction rather than from a
separate classifier head.

Output label space: per-frame voice-activity probability on the system
stream at 12.5 Hz (binary after thresholding).

Evaluation:
  EOT:          fire when system-stream activity rises while user has
                been active (model decides to take the floor).
  Interruption: fire when system-stream activity rises while user is
                still active.

Reads the dataset root from `TT_BENCHMARK_DATA` (see top-level
`.env.example`). Emits predictions to
`predictions/moshi/traces/<task_id>.npz` in the unified submission format
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
    raise NotImplementedError("TODO: implement moshi")


if __name__ == "__main__":
    sys.exit(run("moshi", predict_scores))
