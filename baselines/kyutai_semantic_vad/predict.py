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
`.env.example`). Runs bidirectionally --- once with each speaker as the
agent --- and writes predictions to `predictions/kyutai_semantic_vad/<task_id>.jsonl`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runner import run  # noqa: E402


def predict_for_agent(sample_dir: Path, agent_speaker: int) -> list[dict]:
    """Returns a list of {"time": float, "speaker": int, "label": str}
    events predicted by the model when running as the agent on
    speaker `agent_speaker`'s channel (listening to the other speaker)."""
    raise NotImplementedError("TODO: implement kyutai_semantic_vad")


if __name__ == "__main__":
    sys.exit(run("kyutai_semantic_vad", predict_for_agent))
