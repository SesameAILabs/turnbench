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
"""
from __future__ import annotations

from pathlib import Path


def predict(sample_dir: Path) -> dict[int, list[tuple[float, float, str]]]:
    raise NotImplementedError("TODO: load ESPnet Switchboard checkpoint")


if __name__ == "__main__":
    raise SystemExit("Not implemented yet.")
