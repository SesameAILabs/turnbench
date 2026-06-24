#!/usr/bin/env python3
"""Gemini baseline.

Google Gemini Live (full-duplex streaming dialogue) as a representative
commercial voice agent. Gemini participates in the conversation directly,
so its assessment of the conversational floor is read out from its output
audio rather than from a classifier head.

Evaluation follows the generative-model protocol: each speaker's channel
is streamed into a Gemini Live session as the "user" (two runs per
dialogue), the produced output audio is saved time-aligned with the input,
and both streams are transcribed offline with word-level timestamps. Both
events describe the user (the streamed-in speaker):

  EOT:          the agent starts speaking, signalling the user's turn
                ended -> the user speaker's `eot`.
  Interruption: the user starts speaking while the agent is still
                speaking (barge-in) -> the user speaker's `interruption`.

Pipeline (see README.md in this directory):
  1. inference:   pipeline/ — LiveKit client streaming each
                  speaker_{1,2}_audio into Gemini Live, saving output.wav
                  sample-aligned with the input
  2. ASR:         pipeline/asr_generic.py — parakeet-tdt-0.6b-v2 word
                  timestamps for output.wav and the input audio
  3. this script: reads the cached ASR JSONs and writes the JSON
                  submission directly (per-speaker EOT/interruption event
                  times; word merge gap 0.5 s, times quantized to 12.5 Hz)

Inference/ASR provenance (the JSON submission format carries no metadata):
  checkpoint   gemini-3.1-flash-live-preview
  asr_model    nvidia/parakeet-tdt-0.6b-v2
  merge_gap_s  see asr_floor.MERGE_GAP_S

Reads the dataset root from `TT_BENCHMARK_DATA` and the cached ASR tree
from `GEMINI_ASR_DIR` (see top-level `.env.example`). Writes the JSON
submission ({"schema_version", "predictions": [...]}) to `gemini.json` next
to this script (override with `--out`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runner import data_root, load_env  # noqa: E402
from asr_floor import (  # noqa: E402
    DEFAULT_PRECISION, DEFAULT_THRESHOLD, build_submission)

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_SPLIT_FILE = _REPO / "eval" / "splits" / "dev.txt"
_DEFAULT_OUT = _HERE / "gemini_pred.json"


def _asr_root() -> Path:
    env = load_env()
    if not env.get("GEMINI_ASR_DIR"):
        sys.exit("Set GEMINI_ASR_DIR in .env — see .env.example.")
    return Path(env["GEMINI_ASR_DIR"])


def _split_task_ids() -> list[str]:
    return [ln.strip() for ln in _SPLIT_FILE.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]


def main() -> int:
    ap = argparse.ArgumentParser(description="Gemini baseline -> JSON submission")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                    help=f"Output JSON path (default: {_DEFAULT_OUT})")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"Score threshold for an event (default: {DEFAULT_THRESHOLD})")
    ap.add_argument("--precision", type=int, default=DEFAULT_PRECISION,
                    help=f"Decimal places for event times (default: {DEFAULT_PRECISION})")
    args = ap.parse_args()

    root = data_root()
    task_ids = sorted(t for t in _split_task_ids() if (root / t).is_dir())
    sub = build_submission(root, _asr_root(), task_ids,
                           threshold=args.threshold, precision=args.precision)
    args.out.write_text(json.dumps(sub, indent=2))

    n_eot = sum(len(p[s]["eot"]) for p in sub["predictions"] for s in ("speaker_1", "speaker_2"))
    n_int = sum(len(p[s]["interruption"]) for p in sub["predictions"] for s in ("speaker_1", "speaker_2"))
    print(f"Wrote {len(sub['predictions'])} conversations to {args.out} "
          f"({n_eot} EOT, {n_int} interruption events)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
