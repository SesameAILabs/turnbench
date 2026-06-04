#!/usr/bin/env python3
"""Build per-conversation turn regions from reconciled labels.

A turn region for speaker A is a maximal contiguous run of A's
floor-holding events in the merged-across-speakers timeline, terminated
by either an event by speaker B (change of turn) or an Awkward Silence
(no-man's-land, closes the region).

Three event kinds (drawn from reconciled `stats_out/consensus/`):
  TURN-CONTENT (T): canonical label `Turn`
  LAUGHTER       (L): canonical label `Laughter`
  BREAK          (B): canonical label `AwkwardSilence`

Region-building rules:
  1. T(S):  if no current region, or speaker mismatch → close current
            and open new for S; otherwise extend.
  2. L(S):  include ONLY if there's a current region with the same
            speaker AND it already contains at least one event
            (trailing or in-middle laughter). Leading laughter is
            skipped — a laughter shouldn't interrupt the other
            speaker's turn.
  3. B:     close the current region; AwkwardSilence is no-man's-land,
            not part of any region.
  4. End of stream → close any current region.

Region interval is [first_event.start, last_event.end] of its event
list. Region end = EOT time (consistent with eval/consensus_eot.py).

Inputs:
    stats_out/consensus/<task_id>.jsonl     reconciled 3-of-3 events
Outputs:
    stats_out/turn_regions/<task_id>.jsonl  one row per region:
        {"speaker": int, "region_start": float, "region_end": float,
         "n_events": int, "kinds": {"T": int, "L": int},
         "events": [{"label": str, "start": float, "end": float}, ...]}
    stats_out/turn_regions/_summary.json    per-task + global counts
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from eval.consensus import load_env


TURN_LABEL = "Turn"
LAUGHTER_LABEL = "Laughter"
BREAK_LABEL = "AwkwardSilence"


def build_regions(events: list[dict]) -> list[dict]:
    """`events` is the reconciled per-task consensus jsonl as a list of
    dicts with keys {speaker, start, end, label}. Returns a list of
    region dicts under the rules described in the module docstring."""
    # Sort by start time across both speakers; stable on (start, end).
    timeline = sorted(events, key=lambda e: (e["start"], e["end"]))

    regions: list[dict] = []
    current: dict | None = None

    def close() -> None:
        nonlocal current
        if current is not None and current["events"]:
            current["region_start"] = current["events"][0]["start"]
            current["region_end"] = current["events"][-1]["end"]
            current["n_events"] = len(current["events"])
            current["kinds"] = {
                "T": sum(1 for e in current["events"] if e["label"] == TURN_LABEL),
                "L": sum(1 for e in current["events"] if e["label"] == LAUGHTER_LABEL),
            }
            regions.append(current)
        current = None

    for ev in timeline:
        lbl = ev["label"]
        sp = ev["speaker"]
        if lbl == BREAK_LABEL:
            close()
            continue
        if lbl == TURN_LABEL:
            if current is None or current["speaker"] != sp:
                close()
                current = {"speaker": sp, "events": []}
            current["events"].append({"label": lbl, "start": ev["start"],
                                       "end": ev["end"]})
        elif lbl == LAUGHTER_LABEL:
            # Include only as trailing/in-middle: must have a current
            # region for the same speaker that already contains a T.
            if (current is not None
                    and current["speaker"] == sp
                    and len(current["events"]) > 0):
                current["events"].append({"label": lbl,
                                           "start": ev["start"],
                                           "end": ev["end"]})
            # else: leading laughter — skip entirely.
        # all other canonical labels (Backchannel, Interruption, NonContent)
        # are invisible to region-building.
    close()
    return regions


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    env = load_env(repo / ".env")
    stats_dir = Path(env.get("STATS_DIR", repo / "stats_out"))
    in_dir = stats_dir / "consensus"
    out_dir = stats_dir / "turn_regions"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"Missing {in_dir} — run `python -m eval.consensus` first.",
              file=sys.stderr)
        return 1

    paths = sorted([p for p in in_dir.glob("*.jsonl")
                    if not p.name.startswith("_")
                    and "_excluded" not in p.name],
                   key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    print(f"Building turn regions over {len(paths)} conversations...",
          file=sys.stderr)

    summary: dict = {}
    total_regions = 0
    total_by_speaker: dict[int, int] = defaultdict(int)
    total_with_laughter_tail = 0
    for i, p in enumerate(paths, 1):
        task_id = p.stem
        events = [json.loads(ln) for ln in p.read_text().splitlines()
                  if ln.strip()]
        regions = build_regions(events)
        with (out_dir / f"{task_id}.jsonl").open("w") as f:
            for r in regions:
                f.write(json.dumps(r) + "\n")
        n_with_tail = sum(1 for r in regions
                          if r["events"][-1]["label"] == LAUGHTER_LABEL)
        summary[task_id] = {
            "n_regions": len(regions),
            "n_by_speaker": {1: sum(1 for r in regions if r["speaker"] == 1),
                              2: sum(1 for r in regions if r["speaker"] == 2)},
            "n_with_laughter_tail": n_with_tail,
        }
        total_regions += len(regions)
        for r in regions:
            total_by_speaker[r["speaker"]] += 1
        total_with_laughter_tail += n_with_tail
        if i % 25 == 0:
            print(f"  {i}/{len(paths)}  regions={total_regions}", file=sys.stderr)

    (out_dir / "_summary.json").write_text(json.dumps({
        "n_tasks": len(paths),
        "n_regions": total_regions,
        "n_by_speaker": dict(total_by_speaker),
        "n_with_laughter_tail": total_with_laughter_tail,
        "per_task": summary,
    }, indent=2))
    print(f"\nWrote {total_regions} turn regions across {len(paths)} tasks "
          f"to {out_dir}  (S1={total_by_speaker[1]}, "
          f"S2={total_by_speaker[2]}, with laughter tail="
          f"{total_with_laughter_tail})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
