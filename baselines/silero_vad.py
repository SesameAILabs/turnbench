#!/usr/bin/env python3
"""Silero VAD baseline.

Runs Silero VAD on each per-speaker channel and emits speech segments as
turn predictions. Labels all detected speech as `Normal Turn`.

Pretrained model: https://github.com/snakers4/silero-vad
"""
from __future__ import annotations

from pathlib import Path


def predict(sample_dir: Path) -> dict[int, list[tuple[float, float, str]]]:
    raise NotImplementedError("TODO: load silero-vad, run on speaker_{1,2}_audio.wav")


if __name__ == "__main__":
    raise SystemExit("Not implemented yet.")
