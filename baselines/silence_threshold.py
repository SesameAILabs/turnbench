#!/usr/bin/env python3
"""Silence-threshold baseline.

Predicts turn boundaries by thresholding short-time energy on each speaker
channel and marking gaps longer than `silence_min_s` as turn ends. The
simplest possible non-trivial baseline.
"""
from __future__ import annotations

from pathlib import Path


def predict(sample_dir: Path, silence_min_s: float = 0.5) -> dict[int, list[tuple[float, float, str]]]:
    """Return {speaker_id: [(start_s, end_s, label), ...]} for each speaker."""
    raise NotImplementedError("TODO: implement energy-based VAD + gap detection")


if __name__ == "__main__":
    raise SystemExit("Not implemented yet.")
