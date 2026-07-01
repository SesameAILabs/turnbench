#!/usr/bin/env python3
"""Model scores broken down by conversation metadata.

Complements `per_conversation.py` (which characterizes the *dataset* — event/
backchannel/interruption rates, IAA — per conversation type). This one scores the
committed *baselines* (`baselines/*/predictions-<split>.json`) per conversation
against the gold, then pools TP/FN/FP/TN + latencies within metadata groups —
conversation_type and speaker-gender pairing — and prints group-level recall /
fp_rate / median-latency. It also prints the baseline-independent gold event density
per type. Pooling counts within a group (not averaging per-conversation rates) is
the statistically correct aggregation.

Metadata (conversation_type, per-speaker gender) rides in the dataset's `metadata`
column; it is present even in the public test parquet (labels stripped), so the
metadata axis needs no gold token — only scoring does.

    # dev (public, no token)
    uv run --extra eval python data_analysis/scores_by_metadata.py

    # test (gold is private — set HF_TOKEN to the gold-repo token first)
    uv run --extra eval python data_analysis/scores_by_metadata.py \
        --dataset mundo-ai/turn-benchmark-test-golden

The three-axis regime (acoustic high-recall/high-fp, semantic low/low, learned
balanced) holds within every type; the interesting movement is in *where* each
model's false positives concentrate — see the accompanying findings note.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import pyarrow.parquet as pq  # noqa: E402

from eval.data import (  # noqa: E402
    DEV_DATASET,
    GOLD_DATASET,
    conversation,
    conversation_ids,
    resolve_dataset,
)
from eval.gold import events_for_conversation  # noqa: E402
from eval.score import TaskScore, merge, score_conversation  # noqa: E402
from eval.submission import load_submission  # noqa: E402

PAIRING = {("female", "female"): "FF", ("male", "male"): "MM"}
BASELINES_DIR = Path(__file__).resolve().parent.parent / "baselines"


def metadata_repo(source: str) -> str:
    """A source whose parquet carries the `metadata` column without the gold token:
    the private gold repo publishes none, so read metadata from the public test
    parquet (same conversation ids, metadata intact)."""
    return "mundo-ai/turn-benchmark-test" if source == GOLD_DATASET else source


def load_metadata(source: str) -> dict[str, dict[str, str]]:
    """{conversation_id: {"type": ..., "pairing": FF|MM|mixed}} from the parquet."""
    from baselines.openai_realtime import _shard_files

    out: dict[str, dict[str, str]] = {}
    for shard in _shard_files(metadata_repo(source), None):
        table = pq.ParquetFile(shard).read(columns=["conversation_id", "metadata"])
        for cid, m in zip(table["conversation_id"].to_pylist(), table["metadata"].to_pylist()):
            genders = (m["speaker_1_actor_gender"], m["speaker_2_actor_gender"])
            out[cid] = {
                "type": m["conversation_type"],
                "pairing": PAIRING.get(tuple(sorted(genders)), "mixed"),
            }
    return out


def discover(split: str) -> dict[str, Path]:
    """{label: predictions_path} for every baseline with a committed file for `split`,
    including `-base`/`-large`-style variants (the label carries the suffix)."""
    found: dict[str, Path] = {}
    for path in sorted(BASELINES_DIR.glob(f"*/predictions-{split}*.json")):
        variant = path.stem[len(f"predictions-{split}"):].lstrip("-")
        label = path.parent.name + (f"/{variant}" if variant else "")
        found[label] = path
    return found


def cell(ts: TaskScore) -> str:
    if ts.tp + ts.fn == 0:
        return f"{'--':>16}"
    recall = ts.tp / (ts.tp + ts.fn)
    fp_rate = ts.fp / (ts.fp + ts.tn) if (ts.fp + ts.tn) else float("nan")
    lat = statistics.median(ts.latencies_ms) if ts.latencies_ms else float("nan")
    return f"{f'{recall:.2f}/{fp_rate:.2f}/{lat:.0f}':>16}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DEV_DATASET, help="gold source (HF repo or local dir)")
    ap.add_argument("--split", default=None, choices=["dev", "test"],
                    help="which predictions-<split>.json to score (default: inferred from --dataset)")
    args = ap.parse_args()
    split = args.split or ("test" if ("test" in args.dataset or "golden" in args.dataset) else "dev")

    meta = load_metadata(args.dataset)
    dataset = resolve_dataset(source=args.dataset, skip_audio=True)
    ids = [c for c in conversation_ids(dataset) if c in meta]
    types = sorted({meta[c]["type"] for c in ids})
    baselines = discover(split)
    if not baselines:
        raise SystemExit(f"no baselines/*/predictions-{split}*.json found")

    # gold event density per type (baseline-independent)
    density: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for cid in ids:
        ev = events_for_conversation(conversation(dataset, cid))
        d = density[meta[cid]["type"]]
        d[0] += len(ev.eot_positive_events); d[1] += len(ev.eot_negative_spans)
        d[2] += len(ev.int_positive_events); d[3] += len(ev.int_negative_spans)

    # per-baseline, per-conversation scoring pooled into metadata groups
    pooled: dict[tuple, TaskScore] = defaultdict(TaskScore)
    for label, path in baselines.items():
        by_conv = load_submission(path).by_conversation()
        for cid in ids:
            if cid not in by_conv:
                continue
            sc = score_conversation(by_conv[cid], conversation(dataset, cid))
            for axis, key in (("type", meta[cid]["type"]), ("pairing", meta[cid]["pairing"])):
                merge(pooled[(label, axis, key, "EOT")], sc.task_eot)
                merge(pooled[(label, axis, key, "INT")], sc.task_int)

    n_type = {t: sum(meta[c]["type"] == t for c in ids) for t in types}
    print(f"\n{args.dataset}  ({split} split, {len(ids)} conversations, {len(baselines)} baselines)")
    print("\ngold event density per conversation_type")
    print(f"{'type':<32}{'n':>4}{'EOT+':>7}{'EOT-':>7}{'INT+':>7}{'INT-':>7}{'INT+/conv':>11}")
    for t in types:
        d = density[t]
        print(f"{t:<32}{n_type[t]:>4}{d[0]:>7}{d[1]:>7}{d[2]:>7}{d[3]:>7}{d[2] / n_type[t]:>11.1f}")

    def block(axis: str, keys: list[str]) -> None:
        for task in ("EOT", "INT"):
            print(f"\n{task} by {axis}  (recall / fp_rate / lat_p50_ms)")
            print(f"{'baseline':<20}" + "".join(f"{k[:15]:>16}" for k in keys))
            for label in baselines:
                print(f"{label:<20}" + "".join(cell(pooled[(label, axis, k, task)]) for k in keys))

    block("type", types)
    block("pairing", ["FF", "MM", "mixed"])


if __name__ == "__main__":
    main()
