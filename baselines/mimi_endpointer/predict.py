#!/usr/bin/env python3
"""Mimi-based Endpointer (Kyutai).

Uses Kyutai's Mimi neural audio codec as the acoustic backbone for
end-of-turn detection. Mimi is a streaming neural audio codec that
combines semantic and acoustic information into audio tokens at 12.5 Hz
and 1.1 kbps.

The endpointer operates on the Mimi token stream at 12.5 Hz and processes
two audio streams simultaneously — one per speaker channel.

Params: < 50M.
Output label space: 4-class per frame {user, user-end, system, system-end}.

Evaluation:
  EOT:           fire when `user-end` is predicted.
  Interruption:  fire when `system` or `system-end` appears while user
                 was active.

Reads the dataset root from `TT_BENCHMARK_DATA` (see top-level
`.env.example`). Emits predictions to
`predictions/mimi_endpointer/traces/<task_id>.npz` in the unified submission format
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
    raise NotImplementedError("TODO: implement mimi_endpointer")


if __name__ == "__main__":
    sys.exit(run("mimi_endpointer", predict_scores))
