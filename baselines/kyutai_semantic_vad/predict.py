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
"""
from __future__ import annotations

from pathlib import Path


def predict(sample_dir: Path) -> dict[int, list[tuple[float, float, str]]]:
    raise NotImplementedError("TODO: load Kyutai DSM ASR + VAD head")


if __name__ == "__main__":
    raise SystemExit("Not implemented yet.")
