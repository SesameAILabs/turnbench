#!/usr/bin/env python3
"""pyannote speaker-segmentation baseline.

Uses pyannote.audio's segmentation pipeline on the combined audio,
mapping detected speakers back to the two ground-truth channels.

Pretrained model: pyannote/segmentation-3.0
"""
from __future__ import annotations

from pathlib import Path


def predict(sample_dir: Path) -> dict[int, list[tuple[float, float, str]]]:
    raise NotImplementedError("TODO: load pyannote segmentation, align to gold speakers")


if __name__ == "__main__":
    raise SystemExit("Not implemented yet.")
