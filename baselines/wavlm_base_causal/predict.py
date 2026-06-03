#!/usr/bin/env python3
"""WavLM-Base Causal Predictor.

A lightweight, fully causal turn-taking predictor trained on
Switchboard. Frozen WavLM-Base-Plus encoder (~95M parameters) with
causal attention masking applied to all transformer layers, followed
by a stride-2 Conv1d subsampler (50 Hz -> 25 Hz), a 4-layer causal
Transformer encoder (256d, 4 heads, FFN 1024), and a linear 5-class
head. Total ~98M params (3.8M trainable). Strictly causal: CNN
feature extractor uses zero padding, WavLM transformer layers use a
lower-triangular attention mask, downstream encoder uses causal
self-attention. Single forward pass over the full utterance, no
sliding window. Declared lookahead: 0 ms.

Input: single-channel 16 kHz audio.
Output: 5-class probability per frame {Continuation, Silence,
Interruption, Backchannel, Turn-change} at 25 Hz (40 ms / frame).

Eval mapping into unified submission format:
  EOT (eot_score_speaker_K)
      Process speaker K's channel; fire when
      P(Silence) + P(Turn-change) crosses threshold.
  Interruption (interruption_score_speaker_K)
      Process the interrupted speaker's channel (the OTHER speaker);
      fire when P(Continuation) drops below threshold.

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
