#!/usr/bin/env python3
"""Voice Activity Projection (VAP) baseline.

Continuous turn-taking model from Ekstedt & Skantze (2022). Predicts joint
voice-activity for both speakers a few hundred ms into the future; turn
boundaries are extracted from the projection head's shift/hold decisions.

Reference: https://github.com/ErikEkstedt/VoiceActivityProjection
"""
from __future__ import annotations

from pathlib import Path


def predict(sample_dir: Path) -> dict[int, list[tuple[float, float, str]]]:
    raise NotImplementedError("TODO: load VAP checkpoint, run on combined audio")


if __name__ == "__main__":
    raise SystemExit("Not implemented yet.")
