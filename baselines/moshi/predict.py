#!/usr/bin/env python3
"""Moshi (Kyutai) baseline.

Full-duplex spoken-language model from Kyutai (Defossez et al., 2024).
A multi-stream Transformer that natively models two audio streams (user
and system) plus a text "inner monologue", operating at 12.5 Hz on top of
Mimi codec tokens.

Evaluation follows the generative-model protocol: each speaker's channel
is streamed in real time into a Moshi server as the "user" (two runs per
dialogue), the produced output audio is saved, and both streams are
transcribed offline with word-level timestamps. The agent's speech onsets
are the floor-taking decisions:

  EOT:          agent starts speaking while the user is silent
                -> eot_score for the user speaker.
  Interruption: agent starts speaking while the user is still speaking
                -> interruption_score for the agent's seat (the other
                   speaker).

Pipeline (see README.md in this directory):
  1. inference:   pipeline/inference_moshi_dev_release.py — streams each
                  speaker_{1,2}_audio into a Moshi server, saves output.wav
  2. ASR:         pipeline/asr_generic.py — parakeet-tdt-0.6b-v2 word
                  timestamps for output.wav and the input audio
  3. this script: converts the cached ASR JSONs to unified-format traces
                  (word merge gap 0.5 s, single-frame pulses @ 12.5 Hz,
                  trimmed to the conversation duration)

Reads the dataset root from `TT_BENCHMARK_DATA` and the cached ASR tree
from `MOSHI_ASR_DIR` (see top-level `.env.example`). Emits predictions to
`predictions/moshi/traces/<task_id>.npz` in the unified submission format
(see `eval/submission_format.py`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runner import load_env, run  # noqa: E402
from asr_floor import FRAME_RATE_HZ, MERGE_GAP_S, predict_scores_from_asr  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent.parent

EXTRA = {
    "method": "ASR-derived floor state over saved output audio; "
              "single-frame pulses at agent speech onsets "
              "(user silent -> EOT, user speaking -> interruption)",
    "asr_model": "nvidia/parakeet-tdt-0.6b-v2",
    "merge_gap_s": MERGE_GAP_S,
    "inference": "real-time streaming client against the official server "
                 "(kyutai-labs/moshi, `python -m moshi.server`, default "
                 "checkpoint); continuous input (no halt-on-response)",
}

CHECKPOINT = "kyutai/moshiko-pytorch-bf16"


def _asr_root() -> Path:
    env = load_env()
    if not env.get("MOSHI_ASR_DIR"):
        sys.exit("Set MOSHI_ASR_DIR in .env — see .env.example.")
    return Path(env["MOSHI_ASR_DIR"])


def predict_scores(sample_dir: Path) -> dict:
    """Return per-frame score arrays for one conversation in the unified
    submission format, from the cached ASR word timestamps."""
    return predict_scores_from_asr(sample_dir, _asr_root(),
                                   frame_rate_hz=FRAME_RATE_HZ,
                                   merge_gap_s=MERGE_GAP_S)


def _record_extra(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["extra"] = EXTRA
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    rc = run("moshi", predict_scores,
             checkpoint=CHECKPOINT,
             split_file=_REPO / "eval" / "splits" / "dev.txt")
    _record_extra(_REPO / "predictions" / "moshi")
    sys.exit(rc)
