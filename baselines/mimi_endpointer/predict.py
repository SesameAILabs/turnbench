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
`.env.example`). Runs bidirectionally --- once with each speaker as the
agent --- and writes predictions to `predictions/mimi_endpointer/<task_id>.jsonl`.
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
    raise NotImplementedError("TODO: implement mimi_endpointer")


if __name__ == "__main__":
    sys.exit(run("mimi_endpointer", predict_for_agent))
