#!/usr/bin/env python3
"""Build the consensus event set per conversation.

An event is consensus iff all three annotators (a, b, c) emit the same
canonical label and their (start, end) intervals overlap within
TIME_TOLERANCE on both endpoints. The gold (start, end) for a consensus
event is the median across the three annotators.

Inputs:
    DATA_ROOT/BATCH/<task_id>/speaker_{1,2}_annotation_{a,b,c}.srt
Outputs:
    stats_out/consensus/<task_id>.jsonl   one event per line
        {"speaker": int, "start": float, "end": float, "label": str,
         "raw": [{"a": [s, e]}, {"b": [s, e]}, {"c": [s, e]}]}
    stats_out/consensus/_summary.json     per-sample counts + drop reasons
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import yaml


SRT_TIME = re.compile(r"(\d+):(\d{2}):(\d{2}),(\d{3})")
LABEL = re.compile(r"\[([^\]]+)\]")
ANNOTATORS = ("a", "b", "c")
SPEAKERS = (1, 2)
TIME_TOLERANCE_S = 0.2  # max disagreement on start OR end across annotators
OVERLAP_WINDOW_S = 0.5  # max start-time gap for two events from different annotators to be matched


def load_env(p: Path) -> dict[str, str]:
    env = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def srt_seconds(ts: str) -> float:
    h, m, s, ms = SRT_TIME.match(ts).groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt(path: Path) -> list[tuple[float, float, str]]:
    out = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        idx = 1 if lines[0].strip().isdigit() else 0
        if idx >= len(lines) or "-->" not in lines[idx]:
            continue
        a, b = [t.strip() for t in lines[idx].split("-->")]
        body = " ".join(lines[idx + 1:]).strip()
        m = LABEL.match(body)
        label = m.group(1) if m else ""
        out.append((srt_seconds(a), srt_seconds(b), label))
    return out


def build_label_index(label_map: dict[str, list[str]]) -> dict[str, str]:
    """Returns {fine_label: canonical_label}."""
    out = {}
    for canon, fine_list in label_map.items():
        for fine in fine_list:
            out[fine] = canon
    return out


def map_events(events: list[tuple[float, float, str]],
               canonical: dict[str, str]) -> list[tuple[float, float, str]]:
    """Drop unmapped labels; rewrite the rest to canonical."""
    return [(s, e, canonical[lbl]) for s, e, lbl in events if lbl in canonical]


def find_consensus(per_annotator: dict[str, list[tuple[float, float, str]]]
                   ) -> tuple[list[dict], dict[str, int]]:
    """Given canonical events for a/b/c on one speaker, return the consensus
    event list + a breakdown of why other candidates were dropped.

    Algorithm: use annotator 'a' as the anchor. For each event in 'a', find
    the temporally-closest event of the same label in 'b' and 'c'. If both
    matches are within OVERLAP_WINDOW_S on start AND their endpoints all lie
    within TIME_TOLERANCE_S, accept; median start/end becomes the gold.
    """
    a_evs = per_annotator["a"]
    b_evs = per_annotator["b"]
    c_evs = per_annotator["c"]
    used_b: set[int] = set()
    used_c: set[int] = set()
    consensus = []
    drops = {"no_b_match": 0, "no_c_match": 0, "label_mismatch": 0,
             "endpoint_spread": 0}

    def nearest(target_s: float, target_lbl: str, evs: list, used: set) -> int | None:
        """Index of nearest unused event in evs with the same canonical label."""
        best_i, best_d = None, float("inf")
        for i, (s, _, lbl) in enumerate(evs):
            if i in used:
                continue
            if lbl != target_lbl:
                continue
            d = abs(s - target_s)
            if d < best_d and d <= OVERLAP_WINDOW_S:
                best_d, best_i = d, i
        return best_i

    for s_a, e_a, lbl in a_evs:
        bi = nearest(s_a, lbl, b_evs, used_b)
        if bi is None:
            # any unused 'b' near s_a but different label?
            if any(abs(s - s_a) <= OVERLAP_WINDOW_S and i not in used_b
                   for i, (s, _, lbl_b) in enumerate(b_evs) if lbl_b != lbl):
                drops["label_mismatch"] += 1
            else:
                drops["no_b_match"] += 1
            continue
        ci = nearest(s_a, lbl, c_evs, used_c)
        if ci is None:
            if any(abs(s - s_a) <= OVERLAP_WINDOW_S and i not in used_c
                   for i, (s, _, lbl_c) in enumerate(c_evs) if lbl_c != lbl):
                drops["label_mismatch"] += 1
            else:
                drops["no_c_match"] += 1
            continue

        s_b, e_b, _ = b_evs[bi]
        s_c, e_c, _ = c_evs[ci]
        starts = [s_a, s_b, s_c]
        ends = [e_a, e_b, e_c]
        if (max(starts) - min(starts) > TIME_TOLERANCE_S
                or max(ends) - min(ends) > TIME_TOLERANCE_S):
            drops["endpoint_spread"] += 1
            continue

        used_b.add(bi)
        used_c.add(ci)
        consensus.append({
            "label": lbl,
            "start": round(median(starts), 4),
            "end": round(median(ends), 4),
            "raw": {"a": [s_a, e_a], "b": [s_b, e_b], "c": [s_c, e_c]},
        })

    return consensus, drops


def process_sample(d: Path, canonical: dict[str, str]) -> tuple[list[dict], dict]:
    all_events = []
    sample_drops: dict[str, int] = defaultdict(int)
    sample_counts: dict[str, int] = defaultdict(int)

    for sp in SPEAKERS:
        per_ann = {}
        for ann in ANNOTATORS:
            raw = parse_srt(d / f"speaker_{sp}_annotation_{ann}.srt")
            per_ann[ann] = map_events(raw, canonical)
            sample_counts[f"sp{sp}_ann_{ann}_canonical"] = len(per_ann[ann])
        consensus, drops = find_consensus(per_ann)
        for ev in consensus:
            ev["speaker"] = sp
            all_events.append(ev)
        for k, v in drops.items():
            sample_drops[f"sp{sp}_{k}"] += v
        sample_counts[f"sp{sp}_consensus"] = len(consensus)

    return all_events, {"counts": dict(sample_counts), "drops": dict(sample_drops)}


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    env = load_env(repo / ".env")
    root = Path(env["DATA_ROOT"]) / env["BATCH"]
    out_dir = Path(env.get("STATS_DIR", repo / "stats_out")) / "consensus"
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = yaml.safe_load((repo / "eval" / "label_map.yaml").read_text())
    canonical = build_label_index(label_map)

    sample_dirs = sorted([p for p in root.iterdir() if p.is_dir()],
                         key=lambda p: int(p.name) if p.name.isdigit() else p.name)
    print(f"Building consensus over {len(sample_dirs)} conversations...", file=sys.stderr)

    summary = {}
    total = 0
    for i, d in enumerate(sample_dirs, 1):
        events, info = process_sample(d, canonical)
        total += len(events)
        with (out_dir / f"{d.name}.jsonl").open("w") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")
        summary[d.name] = {"n_events": len(events), **info}
        if i % 25 == 0:
            print(f"  {i}/{len(sample_dirs)}  total consensus events so far: {total}",
                  file=sys.stderr)

    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {total} consensus events across {len(sample_dirs)} samples to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
