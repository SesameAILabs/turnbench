#!/usr/bin/env python3
"""Gemini baseline.

TODO: fill in description and eval methodology.
"""
from __future__ import annotations

from pathlib import Path


def predict(sample_dir: Path) -> dict[int, list[tuple[float, float, str]]]:
    """Return {speaker_id: [(start_s, end_s, label), ...]} per speaker."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("Not implemented yet.")
