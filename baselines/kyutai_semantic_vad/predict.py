#!/usr/bin/env python3
"""Semantic VAD — Kyutai / Unmute STT.

Combines Kyutai's streaming ASR system from the Delayed Streams Modeling
framework with a semantic end-of-turn classifier. The ASR model runs at
~12.5 Hz on a single channel, producing a word-level token stream used to
predict turn completion based on linguistic content rather than acoustic
silence alone.

Params: > 1B.
Output: binary per frame {end-of-turn, not-end-of-turn}, derived from the
VAD head exposed via `lm_gen.step_with_extra_heads`, reflecting user
activity only.

Evaluation:
  EOT:          threshold on speaker-1 (user) probability dropping
                below a value.
  Interruption: threshold on speaker-2 (system) probability rising while
                speaker 1 is still active.

Reads the dataset root from `TT_BENCHMARK_DATA` (see top-level
`.env.example`). Emits predictions to
`predictions/kyutai_semantic_vad/traces/<task_id>.npz` in the unified submission format
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
    raise NotImplementedError("TODO: implement kyutai_semantic_vad")


if __name__ == "__main__":
    sys.exit(run("kyutai_semantic_vad", predict_scores))
