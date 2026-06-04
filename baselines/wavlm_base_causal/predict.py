#!/usr/bin/env python3
"""WavLM-Base Causal Predictor (CMU).

Description: A lightweight, fully causal turn-taking predictor trained
on Switchboard. The model uses a frozen WavLM-Base-Plus encoder
(~95M parameters) with causal attention masking applied to all
transformer layers, followed by a stride-2 Conv1d subsampler
(50 Hz to 25 Hz), a 4-layer causal Transformer encoder
(256d, 4 heads, FFN 1024), and a linear 5-class head. Total parameters:
~98M (3.8M trainable). The model runs in a single causal forward pass
over the full utterance with no sliding window required. All components
are strictly causal: the CNN feature extractor uses zero padding, the
WavLM transformer layers use a lower-triangular attention mask, and the
downstream encoder uses causal self-attention. Input: single-channel
audio at 16 kHz. Output label space: a 5-class turn-taking probability
distribution per frame, {Continuation, Silence, Interruption,
Backchannel, Turn-change}, emitted every 40 ms (25 Hz). Declared
lookahead: 0 ms.

How to eval:
  EOT: process the current speaker's channel; fire when
       P(Silence) + P(Turn-change) crosses threshold (speaker finishing,
       listener's turn).
  Interruption: process the interrupted speaker's channel; fire when
       P(Continuation) drops below threshold (speaker's turn disrupted).

Reads the dataset root from `TT_BENCHMARK_DATA` (see .env.example).
Emits predictions to `predictions/wavlm_base_causal/traces/<task_id>.npz`
in the unified submission format (see eval/submission_format.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runner import run  # noqa: E402


def predict_scores(sample_dir: Path) -> dict:
    raise NotImplementedError("TODO: implement wavlm_base_causal")


if __name__ == "__main__":
    sys.exit(run("wavlm_base_causal", predict_scores))
