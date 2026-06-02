#!/usr/bin/env python3
"""Voice Activity Projection (VAP).

Two-stream model that takes both speaker channels as input and produces a
continuous projection of conversational floor state at 50 Hz. Rather than
detecting silence, VAP is trained to predict the probability of future
voice activity for each speaker, allowing it to anticipate turn
transitions before they occur.

Params: > 100M.
Output: continuous probability per speaker per frame — aggregated binary
probabilities for speaker 1 and speaker 2 — thresholded to produce
binary turn-state predictions.

Evaluation:
  EOT:          threshold on speaker-1 (user) probability dropping
                below a value.
  Interruption: threshold on speaker-2 (system) probability rising while
                speaker 1 is still active.

Reference: Ekstedt & Skantze (2022),
https://github.com/ErikEkstedt/VoiceActivityProjection

Reads the dataset root from `TT_BENCHMARK_DATA` (see top-level
`.env.example`). Emits predictions to
`predictions/vap/traces/<task_id>.npz` in the unified submission format
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
    raise NotImplementedError("TODO: implement vap")


if __name__ == "__main__":
    sys.exit(run("vap", predict_scores))
