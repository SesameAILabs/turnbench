#!/usr/bin/env python3
"""Merge sharded per-frame probs partials into single probs files.

predict.py --shard K N --probs-out-dir D writes `D/probs-{task}.shardK-of-N.json`
partials (round-robin conversation split, ids[K::N]). This reassembles each task's
partials into `D/probs-{task}.json` in canonical dataset order and validates
coverage against the split's durations.

    uv run --extra eval python turnbench/analysis/merge_prob_shards.py \
        predictions/wavlm_probs/wavlm_large_anchor_dev --split dev
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from turnbench.durations import load_durations
from turnbench.sweep import ProbsFile, load_probs, validate_probs


def merge(shard_dir: Path, split: str) -> None:
    durations = load_durations(split)
    order = {cid: i for i, cid in enumerate(sorted(durations, key=int))}
    for task in ("eot", "int"):
        partials = sorted(shard_dir.glob(f"probs-{task}.shard*-of-*.json"))
        if not partials:
            raise FileNotFoundError(f"no probs-{task} shard partials in {shard_dir}")
        n_expected = {int(re.search(r"-of-(\d+)", p.name).group(1)) for p in partials}
        assert len(n_expected) == 1 and len(partials) == n_expected.pop(), (
            f"incomplete shard set for {task}: {[p.name for p in partials]}"
        )
        entries, frame_rate = [], None
        for p in partials:
            pf = load_probs(p)
            frame_rate = frame_rate or pf.frame_rate_hz
            assert pf.frame_rate_hz == frame_rate, "frame_rate mismatch across shards"
            entries.extend(pf.probs)
        entries.sort(key=lambda e: order[e.conversation_id])
        merged = ProbsFile(schema_version=1, task=task, frame_rate_hz=frame_rate, probs=entries)
        validate_probs(merged, durations)
        out = shard_dir / f"probs-{task}.json"
        out.write_text(merged.model_dump_json())
        print(f"merged {len(partials)} shards -> {out} ({len(entries)} convs, validated)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir", type=Path)
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    args = ap.parse_args()
    merge(args.shard_dir, args.split)
