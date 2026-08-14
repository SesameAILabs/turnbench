#!/usr/bin/env python3
"""Full gap / pause / floor-transfer-offset distributions for the corpus.

Complements per_conversation.py, which reduces FTO to mean/median: this exports
the complete event-level timing distributions, pooled across the three
annotator tracks, overall and per conversation type. The Switchboard
counterpart is data_analysis/swbd/per_conversation_swbd.py; compare with
plot_timing_distributions.py.

Reads the HF releases (dev + private golden test = the full 154-dialogue
corpus, raw three-annotator tracks; types from the `metadata` column) at
pinned revisions; needs HF credentials (HF_TOKEN in the environment or
repo-root `.env`).

Definitions (per_conversation.turn_timing over eval.gold's turn view — the
floor-claiming spans: turn labels plus floor-taking interruptions — both
speakers, sorted by onset, consecutive pairs):
  fto    speaker change: next floor span start - previous span end
         (+gap / -overlap)
  pause  same speaker resumes after silence, no intervening floor claim by the
         other speaker: next span start - previous span end, where positive
Gap durations (silent floor transfers) are the positive FTOs, derived at read
time.

Output: stats_out/timing_distributions.json
  {"corpus": "turnbench", "groups": {"all": {...}, "<type>": {...}}}
  where each group holds {"fto": [...], "pause": [...]} in seconds (3 decimals).

Usage:
  uv run --extra eval python data_analysis/timing_distributions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_analysis.per_conversation import turn_timing  # noqa: E402
from data_analysis.results_by_conversation_type import load_metadata  # noqa: E402
from eval.data import (  # noqa: E402
    ANNOTATORS,
    FULL_CORPUS_SOURCES,
    conversation,
    conversation_ids,
    resolve_dataset,
)
from eval.gold import TURN_LABELS as FLOOR_LABELS  # noqa: E402


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    out_dir = repo / "stats_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, dict[str, list[float]]] = {}
    n_conversations = 0
    for source in FULL_CORPUS_SOURCES:
        dataset = resolve_dataset(source=source, skip_audio=True)
        types = {cid: m["type"] for cid, m in load_metadata(source).items()}
        ids = conversation_ids(dataset)
        n_conversations += len(ids)
        print(f"{source}: {len(ids)} conversations", file=sys.stderr)
        for cid in ids:
            conv = conversation(dataset, cid)
            for ann in ANNOTATORS:
                ftos, pauses = turn_timing({1: conv.annotations[(1, ann)],
                                            2: conv.annotations[(2, ann)]},
                                           labels=FLOOR_LABELS)
                ftos = [round(f, 3) for f in ftos]
                pauses = [round(p, 3) for p in pauses]
                for name in ("all", types[cid]):
                    b = groups.setdefault(name, {"fto": [], "pause": []})
                    b["fto"].extend(ftos)
                    b["pause"].extend(pauses)

    out = out_dir / "timing_distributions.json"
    out.write_text(json.dumps({"corpus": "turnbench", "n_conversations": n_conversations,
                               "sources": list(FULL_CORPUS_SOURCES),
                               "pooled_annotators": list(ANNOTATORS),
                               "groups": groups}))
    counts = ", ".join(f"{k}: {len(v['fto'])} fto / {len(v['pause'])} pause"
                       for k, v in sorted(groups.items()))
    print(f"Wrote {out}\n{counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
