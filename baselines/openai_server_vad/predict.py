#!/usr/bin/env python3
"""OpenAI Realtime `server_vad` end-of-turn baseline — see README.md.

Thin shim over baselines/openai_realtime.py with the acoustic-silence VAD mode.

    python -m baselines.openai_server_vad.predict                       # score on dev
    python -m baselines.openai_server_vad.predict --out preds.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from baselines.openai_realtime import run_openai_baseline  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_openai_baseline("server_vad", Path(__file__).resolve().parent))
