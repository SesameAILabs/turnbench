#!/usr/bin/env python3
"""Smart Turn v3 (Pipecat).

Uses Whisper Tiny as a base with a linear classifier head; transformer-
based, ~40M params. Available in int8-quantized and full fp32 versions.

Small and fast enough to run on CPU (~12 ms inference on modern CPUs).
Non-causal and chunk-based — processes a complete audio segment rather
than streaming frame-by-frame, making it a strong accuracy reference
point at the cost of added latency.

Input:  16 kHz mono PCM audio, up to 8 s.
Output: binary per chunk {turn-complete, turn-incomplete}, reflecting
user activity only.
"""
from __future__ import annotations

from pathlib import Path


def predict(sample_dir: Path) -> dict[int, list[tuple[float, float, str]]]:
    raise NotImplementedError("TODO: load Smart Turn v3 (int8 or fp32)")


if __name__ == "__main__":
    raise SystemExit("Not implemented yet.")
