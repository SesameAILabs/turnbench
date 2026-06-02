#!/usr/bin/env python3
"""Gemini baseline.

TODO: fill in description and eval methodology.

Reads the dataset root from `TT_BENCHMARK_DATA` (see top-level
`.env.example`). Runs bidirectionally — once with each speaker as the
agent — and writes predictions to `predictions/gemini/<task_id>.jsonl`.
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
    raise NotImplementedError("TODO: implement Gemini prompting + parsing")


if __name__ == "__main__":
    sys.exit(run("gemini", predict_for_agent))
