#!/usr/bin/env python3
"""GPT-Live-1 EOT baseline — pyannote-VAD readout of recorded Live sessions.

Same readout as `baselines/gemini_vad` (imported, not copied), pointed at the
recordings `record.py` writes: EOT_K = agent VAD onset in direction K while
user_K is VAD-inactive, committed at the onset. EOT only; interruption lists
are empty (the benchmark has no INT methodology for full-duplex models).

    uv run python baselines/gpt_live_1/predict.py --out baselines/gpt_live_1/predictions-dev.json
    uv run python baselines/gpt_live_1/predict.py --dataset mundo-ai/turn-benchmark-test \
        --out baselines/gpt_live_1/predictions-test.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from baselines.gemini_vad.predict import (  # noqa: E402
    DEV_DATASET,
    VadCache,
    dataset_index,
    predict_conversation,
    warm_cache,
)
from turnbench.submission import SCHEMA_VERSION, Submission  # noqa: E402

DEFAULT_RECORDINGS = _HERE / "recordings"
DEFAULT_CACHE = _HERE / ".vad_cache"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEV_DATASET)
    ap.add_argument("--recordings", type=Path, default=DEFAULT_RECORDINGS)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", required=True, help="write predictions JSON here")
    ap.add_argument("--vad-workers", type=int, default=1)
    ap.add_argument("--only-recorded", action="store_true",
                    help="skip conversations without recordings (pilot runs)")
    args = ap.parse_args()

    index = dataset_index(args.dataset)
    if args.only_recorded:
        index = {t: v for t, v in index.items()
                 if all((args.recordings / t / f"speaker_{k}" / "output.flac").exists()
                        for k in (1, 2))}
    cache = VadCache(args.cache_dir)
    if args.vad_workers > 1:
        warm_cache(index, args.recordings, args.cache_dir, args.vad_workers)
    predictions = []
    for i, task_id in enumerate(sorted(index, key=int), 1):
        predictions.append(
            predict_conversation(task_id, *index[task_id], args.recordings, cache))
        print(f"[{i}/{len(index)}] {task_id}", file=sys.stderr)
    submission = Submission(schema_version=SCHEMA_VERSION, predictions=predictions)
    Path(args.out).write_text(submission.model_dump_json(indent=2), encoding="utf-8")
    print(f"Wrote {len(predictions)} predictions to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
