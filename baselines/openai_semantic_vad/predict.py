#!/usr/bin/env python3
"""OpenAI Realtime `semantic_vad` end-of-turn baseline — see README.md.

Thin shim over baselines/openai_realtime.py with the semantic turn-end mode (paced
ingest, so the model's variable decision delay is folded into the commit time).

    python -m baselines.openai_semantic_vad.predict                     # score on dev
    python -m baselines.openai_semantic_vad.predict --out preds.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from baselines.openai_realtime import run_openai_baseline  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(
        run_openai_baseline("semantic_vad", Path(__file__).resolve().parent)
    )
