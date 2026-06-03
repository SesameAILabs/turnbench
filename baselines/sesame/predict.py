#!/usr/bin/env python3
"""Sesame internal turn-taking system (placeholder).

Description and evaluation methodology forthcoming. Predictions
generated outside this repository (under a separate sesame branch)
and written into predictions/sesame_<checkpoint>_<split>/ in the
unified submission format (eval/submission_format.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runner import run  # noqa: E402


def predict_scores(sample_dir: Path) -> dict:
    raise NotImplementedError("Sesame predictions are generated externally; see README.")


if __name__ == "__main__":
    sys.exit(run("sesame", predict_scores))
