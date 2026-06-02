#!/usr/bin/env python3
"""ESPnet Turn-Taking Prediction (Switchboard).

Trained using the Switchboard recipe in ESPnet. Built on a frozen
Whisper-medium encoder (~306M params) producing 1024-dim features at
50 Hz, with a small trainable classification head (~1M params):
    Tanh -> Linear(1024->1024) -> Linear(1024->5)

Trained and evaluated as part of "Talking Turns: Benchmarking Audio
Foundation Models on Turn-Taking Dynamics" (Arora et al., ICLR 2025) —
the closest prior work to this benchmark.

Input:  single-channel 16 kHz audio, up to 30 s of left context.
Output: 5-class probability distribution per 40 ms chunk (25 Hz):
        {Continuation, Silence, Interruption, Backchannel, Turn-change}.

Evaluation: per-class one-vs-rest ROC-AUC; predictions are NOT thresholded
to binary.

Reads the dataset root from `TT_BENCHMARK_DATA` (see top-level
`.env.example`). Runs bidirectionally --- once with each speaker as the
agent --- and writes predictions to `predictions/espnet_turntaking/<task_id>.jsonl`.
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
    raise NotImplementedError("TODO: implement espnet_turntaking")


if __name__ == "__main__":
    sys.exit(run("espnet_turntaking", predict_for_agent))
