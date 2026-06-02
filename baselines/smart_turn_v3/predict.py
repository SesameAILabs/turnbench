#!/usr/bin/env python3
"""Smart Turn v3 (Pipecat).

Uses Whisper Tiny as a base with a linear classifier head; transformer-
based, ~40M params. Available in int8-quantized and full fp32 versions.

Small and fast enough to run on CPU (~12 ms inference on modern CPUs).
Non-causal and chunk-based — processes a complete audio segment rather
than streaming frame-by-frame, making it a strong accuracy reference
point at the cost of added latency.

Input:  16 kHz mono PCM audio, up to 8 s.
Output: binary per chunk {turn-complete, turn-incomplete}, reflecting
user activity only.

Reads the dataset root from `TT_BENCHMARK_DATA` (see top-level
`.env.example`). Runs bidirectionally --- once with each speaker as the
agent --- and writes predictions to `predictions/smart_turn_v3/<task_id>.jsonl`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _runner import run  # noqa: E402


def predict_for_agent(sample_dir: Path, agent_speaker: int) -> list[dict]:
    """Returns a list of {"time": float, "speaker": int, "label": str}
    events predicted by the model when running as the agent on
    speaker `agent_speaker`'s channel (listening to the other speaker)."""
    raise NotImplementedError("TODO: implement smart_turn_v3")


if __name__ == "__main__":
    sys.exit(run("smart_turn_v3", predict_for_agent))
