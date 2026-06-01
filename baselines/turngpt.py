#!/usr/bin/env python3
"""TurnGPT baseline.

Text-only turn-shift prediction model from Skantze (2017+). Operates on
the ASR transcript and predicts turn boundaries from a language model's
distribution over a special <ts> token.

Reference: https://github.com/ErikEkstedt/TurnGPT
"""
from __future__ import annotations

from pathlib import Path


def predict(sample_dir: Path) -> dict[int, list[tuple[float, float, str]]]:
    raise NotImplementedError("TODO: ASR -> TurnGPT -> align <ts> probabilities to time")


if __name__ == "__main__":
    raise SystemExit("Not implemented yet.")
