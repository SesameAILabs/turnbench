#!/usr/bin/env python3
"""Build the per-conversation EOT (end-of-turn) consensus gold.

An EOT event for speaker A fires at A's turn END time IFF the next
chronological turn (per the same annotator) belongs to the OTHER speaker.
A turn followed by another turn by the same speaker (continuation across a
pause) does NOT count.

What counts as a "turn" is the canonical `Turn` bucket in
`eval/label_map.yaml` — every speaker-content event that contributes to
floor-holding (Normal Turn, Regular Turn, Strong Floor Hold, Bounded
Response, Filler, Overlap, Awkward Silence).

EOT reconciliation (3-of-3 strict):
  Anchor on annotator 'a'. For each EOT in 'a', match the temporally
  nearest EOT in 'b' and 'c' for the SAME speaker. All three times must
  lie within TIME_TOLERANCE_S of each other; gold time is the median.

Outputs:
  stats_out/consensus_eot/<task_id>.jsonl
      {"speaker": int, "time": float, "raw": {"a": float, "b": float, "c": float}}
  stats_out/consensus_eot/_summary.json     per-sample counts + drop reasons
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import yaml

from eval.consensus import (
    ANNOTATORS,
    SPEAKERS,
    build_label_index,
    load_env,
    map_events,
    parse_srt,
)


TIME_TOLERANCE_S = 0.2  # matches consensus.py
OVERLAP_WINDOW_S = 0.5  # matches consensus.py
TURN_LABEL = "Turn"  # the canonical bucket we consider as a "turn"


def annotator_eots(turns_by_speaker: dict[int, list[tuple[float, float, str]]]
                   ) -> list[tuple[float, int]]:
    """For one annotator: from per-speaker canonical Turn events, derive a
    list of (eot_time, speaker) where speaker just yielded the floor to the
    OTHER speaker.

    Logic: merge both speakers' canonical Turn events; sort by start time.
    For each turn i, if turn[i+1] is by the other speaker, emit EOT at
    turn[i].end with speaker = turn[i].speaker.
    """
    merged: list[tuple[float, float, int]] = []
    for sp, evs in turns_by_speaker.items():
        for s, e, lbl in evs:
            if lbl != TURN_LABEL:
                continue
            merged.append((s, e, sp))
    merged.sort(key=lambda x: x[0])
    out: list[tuple[float, int]] = []
    for i, (_, e_i, sp_i) in enumerate(merged):
        if i + 1 >= len(merged):
            continue  # last turn — no following turn to check
        if merged[i + 1][2] != sp_i:
            out.append((e_i, sp_i))
    return out


def reconcile_eots(per_annotator: dict[str, list[tuple[float, int]]]
                   ) -> tuple[list[dict], dict[str, int]]:
    """3-of-3 reconciliation on (time, speaker) EOT events. Anchor on 'a';
    nearest match per speaker in 'b' and 'c' within OVERLAP_WINDOW_S;
    accept if the max time-spread is ≤ TIME_TOLERANCE_S. Gold = median."""
    a_eots = per_annotator["a"]
    b_eots = per_annotator["b"]
    c_eots = per_annotator["c"]
    used_b: set[int] = set()
    used_c: set[int] = set()
    drops = {"no_b_match": 0, "no_c_match": 0, "speaker_mismatch": 0,
             "time_spread": 0}
    out: list[dict] = []

    def nearest(target_t: float, target_sp: int,
                evs: list[tuple[float, int]], used: set[int]) -> int | None:
        best_i, best_d = None, float("inf")
        for i, (t, sp) in enumerate(evs):
            if i in used or sp != target_sp:
                continue
            d = abs(t - target_t)
            if d < best_d and d <= OVERLAP_WINDOW_S:
                best_d, best_i = d, i
        return best_i

    for t_a, sp in a_eots:
        bi = nearest(t_a, sp, b_eots, used_b)
        if bi is None:
            if any(abs(t - t_a) <= OVERLAP_WINDOW_S and i not in used_b
                   and sp_b != sp for i, (t, sp_b) in enumerate(b_eots)):
                drops["speaker_mismatch"] += 1
            else:
                drops["no_b_match"] += 1
            continue
        ci = nearest(t_a, sp, c_eots, used_c)
        if ci is None:
            if any(abs(t - t_a) <= OVERLAP_WINDOW_S and i not in used_c
                   and sp_c != sp for i, (t, sp_c) in enumerate(c_eots)):
                drops["speaker_mismatch"] += 1
            else:
                drops["no_c_match"] += 1
            continue
        t_b = b_eots[bi][0]
        t_c = c_eots[ci][0]
        times = [t_a, t_b, t_c]
        if max(times) - min(times) > TIME_TOLERANCE_S:
            drops["time_spread"] += 1
            continue
        used_b.add(bi)
        used_c.add(ci)
        out.append({
            "speaker": sp,
            "time": round(median(times), 4),
            "raw": {"a": round(t_a, 4), "b": round(t_b, 4), "c": round(t_c, 4)},
        })
    return out, drops


def process_sample(d: Path, canonical: dict[str, str]
                   ) -> tuple[list[dict], dict]:
    per_annotator_eots: dict[str, list[tuple[float, int]]] = {}
    n_raw_per_ann: dict[str, int] = {}
    for ann in ANNOTATORS:
        per_speaker: dict[int, list[tuple[float, float, str]]] = {}
        for sp in SPEAKERS:
            raw = parse_srt(d / f"speaker_{sp}_annotation_{ann}.srt")
            per_speaker[sp] = map_events(raw, canonical)
        eots = annotator_eots(per_speaker)
        per_annotator_eots[ann] = eots
        n_raw_per_ann[ann] = len(eots)

    consensus, drops = reconcile_eots(per_annotator_eots)
    info = {
        "n_raw_per_annotator": n_raw_per_ann,
        "n_consensus": len(consensus),
        "drops": drops,
    }
    return consensus, info


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    env = load_env(repo / ".env")
    root = (Path(env["TT_BENCHMARK_DATA"]) if env.get("TT_BENCHMARK_DATA")
            else Path(env["DATA_ROOT"]) / env["BATCH"])
    out_dir = Path(env.get("STATS_DIR", repo / "stats_out")) / "consensus_eot"
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = yaml.safe_load((repo / "eval" / "label_map.yaml").read_text())
    canonical = build_label_index(label_map)

    sample_dirs = sorted([p for p in root.iterdir() if p.is_dir()],
                         key=lambda p: int(p.name) if p.name.isdigit() else p.name)
    print(f"Building EOT consensus over {len(sample_dirs)} conversations...",
          file=sys.stderr)

    summary: dict = {}
    total = 0
    total_per_speaker = defaultdict(int)
    for i, d in enumerate(sample_dirs, 1):
        events, info = process_sample(d, canonical)
        total += len(events)
        for ev in events:
            total_per_speaker[ev["speaker"]] += 1
        with (out_dir / f"{d.name}.jsonl").open("w") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")
        summary[d.name] = info
        if i % 25 == 0:
            print(f"  {i}/{len(sample_dirs)}  eots={total}", file=sys.stderr)

    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {total} EOT consensus events across {len(sample_dirs)} "
          f"samples to {out_dir}  (S1={total_per_speaker[1]}, "
          f"S2={total_per_speaker[2]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
