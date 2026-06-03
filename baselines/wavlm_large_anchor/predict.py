#!/usr/bin/env python3
"""WavLM-Large ANCHOR Judge.

A general-purpose speech quality and turn-taking model based on the
ANCHOR framework, adapted for turn-taking prediction on Switchboard.
Frozen WavLM-Large frontend (~315M parameters), a 4-layer Transformer
audio encoder, and a 12-layer autoregressive Transformer decoder
that predicts the turn-taking class via token generation. Total
~628M params (313M trainable). Inference uses 4 s sliding windows at
40 ms stride; within each window attention is bidirectional, but
because windows are independent and each ends at the current time,
no future audio is observed. Declared lookahead: 0 ms.

Input: single-channel 16 kHz audio, 4 s context per window.
Output: same 5-class distribution as the causal predictor, emitted
every 40 ms (25 Hz).

Eval mapping into unified submission format:
  EOT (eot_score_speaker_K)
      Process speaker K's channel; fire when
      P(Silence) + P(Turn-change) crosses threshold.
  Interruption (interruption_score_speaker_K)
      Process the interrupted speaker's channel; fire when
      P(Continuation) drops below threshold.

Reads the dataset root from `TT_BENCHMARK_DATA` (see .env.example).
Emits predictions to `predictions/wavlm_large_anchor/traces/<task_id>.npz`
in the unified submission format (see eval/submission_format.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runner import run  # noqa: E402


def predict_scores(sample_dir: Path) -> dict:
    raise NotImplementedError("TODO: implement wavlm_large_anchor")


if __name__ == "__main__":
    sys.exit(run("wavlm_large_anchor", predict_scores))
