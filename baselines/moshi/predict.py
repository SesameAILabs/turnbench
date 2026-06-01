#!/usr/bin/env python3
"""Moshi (Kyutai).

Full-duplex spoken-language model from Kyutai (Defossez et al., 2024).
A multi-stream Transformer that natively models two audio streams (user
and system) plus a text "inner monologue", operating at 12.5 Hz on top of
Mimi codec tokens.

Because Moshi is generative and continuously predicts whether it should
speak on the system stream, turn-taking events can be read out directly
from the system-stream voice-activity prediction rather than from a
separate classifier head.

Output label space: per-frame voice-activity probability on the system
stream at 12.5 Hz (binary after thresholding).

Evaluation:
  EOT:          fire when system-stream activity rises while user has
                been active (model decides to take the floor).
  Interruption: fire when system-stream activity rises while user is
                still active.
"""
from __future__ import annotations

from pathlib import Path


def predict(sample_dir: Path) -> dict[int, list[tuple[float, float, str]]]:
    raise NotImplementedError("TODO: load Moshi checkpoint, read system-stream VA head")


if __name__ == "__main__":
    raise SystemExit("Not implemented yet.")
