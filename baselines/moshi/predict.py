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
`.env.example`). Runs bidirectionally --- once with each speaker as the
agent --- and writes predictions to `predictions/moshi/<task_id>.jsonl`.
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
    raise NotImplementedError("TODO: implement moshi")


if __name__ == "__main__":
    sys.exit(run("moshi", predict_for_agent))
