#!/usr/bin/env python3
"""Human reaction-time baselines, measured from the consensus gold.

Two quantities, one per benchmark track:

EOT — floor-transfer time. At every gold EOT anchor (the anchor rule is
replayed verbatim from turnbench.gold.build_conversation_events over the
consensus turn view), the interlocutor's floor-claiming response onset
relative to the turn end:

    eot_latency = response span start - anchor time

Negative = the response began in overlap, before the turn ended. Reported
both over all transfers (signed) and over positive-gap transfers only; the
latter is the setting of Stivers et al. (2009, PNAS), who report a ~208 ms
mean response offset on question-response pairs across 10 languages.

INT — yield time. At every gold floor-taking Interruption anchor (label
view; the anchor is the interrupter's onset), how long the interrupted
speaker keeps speaking:

    int_yield = interrupted speaker's active turn-view span end - anchor time

Anchors where the interrupted speaker has no active turn-view span at onset
(consensus views can disagree at the boundary) are counted and skipped.

Human recall and false-positive rate are 1 and 0 by construction, not
measured: every EOT anchor is defined by the floor actually passing, and a
floor entry away from a TRP is labeled an Interruption by the taxonomy.

Reads the annotator tracks only (no audio) from the pinned dev and golden
test splits; needs ambient HF credentials on first run, cached after.

Usage:
    uv run python turnbench/analysis/human_baseline.py

Output: a table per split plus pooled, and stats_out/human_baseline.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from turnbench.data import (  # noqa: E402
    DEV_DATASET,
    GOLD_DATASET,
    conversation,
    conversation_ids,
    resolve_dataset,
)
from turnbench.gold import (  # noqa: E402
    CANONICAL,
    TURN_CANONICAL,
    collect_turns,
    consensus_for_conversation,
    first_start_after,
)

OUT_PATH = Path(__file__).resolve().parents[2] / "stats_out" / "human_baseline.json"


def conversation_latencies(conv) -> dict:
    """One conversation's human reaction times.

    Returns {"eot": [signed floor-transfer offsets], "eot_unanswered": int,
    "int_yield": [yield times], "int_no_active_span": int}.
    """
    turn_events, _ = consensus_for_conversation(conv, canonical=TURN_CANONICAL)
    label_events, _ = consensus_for_conversation(conv, canonical=CANONICAL)
    speaker_turns = {s: collect_turns(turn_events, s) for s in (1, 2)}

    int_onsets = {
        (e.speaker, round(e.start, 3))
        for e in label_events
        if e.label == "Interruption"
    }
    eot: list[float] = []
    eot_smooth: list[float] = []
    eot_unanswered = 0
    seen: set[tuple[int, float]] = set()
    for speaker in (1, 2):
        own = speaker_turns[speaker]
        other = speaker_turns[2 if speaker == 1 else 1]
        for segment in own.segments:
            inside_own_segment = any(
                s is not segment and s.start < segment.end < s.end
                for s in own.segments
            )
            if inside_own_segment:
                continue
            next_self = first_start_after(own.start_times, segment.end)
            next_other = first_start_after(other.start_times, segment.end)
            overlapping = [
                s for s in other.segments if s.start < segment.end < s.end
            ]
            is_anchor = (
                next_self == float("inf")
                or next_other < next_self
                or bool(overlapping)
            )
            if not is_anchor or (speaker, segment.end) in seen:
                continue
            seen.add((speaker, segment.end))
            if overlapping:
                # Overlapping take-over: the latest-starting open span is it.
                response = max(overlapping, key=lambda s: s.start)
            elif next_other < float("inf"):
                response = min(
                    s for s in other.segments if s.start == next_other
                )
            else:
                eot_unanswered += 1  # conversation-final turn
                continue
            latency = response.start - segment.end
            eot.append(latency)
            # Smooth transfer: the response is not a labeled floor-taking
            # interruption (the label view's own distinction).
            if (response.speaker, round(response.start, 3)) not in int_onsets:
                eot_smooth.append(latency)

    int_yield: list[float] = []
    int_no_active_span = 0
    for event in label_events:
        if event.label != "Interruption":
            continue
        interrupted = speaker_turns[2 if event.speaker == 1 else 1]
        active = [
            s for s in interrupted.segments if s.start < event.start < s.end
        ]
        if not active:
            int_no_active_span += 1
            continue
        # The span being interrupted is the latest-starting one still open.
        span = max(active, key=lambda s: s.start)
        int_yield.append(span.end - event.start)

    return {
        "eot": eot,
        "eot_smooth": eot_smooth,
        "eot_unanswered": eot_unanswered,
        "int_yield": int_yield,
        "int_no_active_span": int_no_active_span,
    }


def percentiles(values: list[float]) -> dict:
    arr = np.array(values)
    return {
        "n": int(arr.size),
        "p10": round(float(np.percentile(arr, 10)), 3),
        "p25": round(float(np.percentile(arr, 25)), 3),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p75": round(float(np.percentile(arr, 75)), 3),
        "p90": round(float(np.percentile(arr, 90)), 3),
        "mean": round(float(arr.mean()), 3),
    }


def summarize(eot: list[float], eot_smooth: list[float], eot_unanswered: int,
              int_yield: list[float], int_no_active_span: int) -> dict:
    gaps_only = [v for v in eot_smooth if v > 0]
    return {
        "eot_floor_transfer_s": percentiles(eot),
        "eot_smooth_transfer_s": percentiles(eot_smooth),
        "eot_gap_only_s": percentiles(gaps_only),
        "eot_smooth_share_overlap": round(
            float(np.mean(np.array(eot_smooth) < 0)), 3
        ),
        "eot_share_overlap_response": round(
            float(np.mean(np.array(eot) < 0)), 3
        ),
        "eot_unanswered_final": eot_unanswered,
        "int_yield_s": percentiles(int_yield),
        "int_no_active_span": int_no_active_span,
        "assumptions": "recall=1, fpr=0 by construction",
    }


def main() -> int:
    results: dict[str, dict] = {}
    pooled = {"eot": [], "eot_smooth": [], "eot_unanswered": 0, "int_yield": [], "int_no_active_span": 0}
    for name, source in (("dev", DEV_DATASET), ("test", GOLD_DATASET)):
        dataset = resolve_dataset(source=source, skip_audio=True)
        split = {"eot": [], "eot_smooth": [], "eot_unanswered": 0, "int_yield": [], "int_no_active_span": 0}
        for task_id in conversation_ids(dataset):
            row = conversation_latencies(conversation(dataset, task_id))
            for key in ("eot", "eot_smooth", "int_yield"):
                split[key].extend(row[key])
                pooled[key].extend(row[key])
            for key in ("eot_unanswered", "int_no_active_span"):
                split[key] += row[key]
                pooled[key] += row[key]
        results[name] = summarize(
            split["eot"], split["eot_smooth"], split["eot_unanswered"],
            split["int_yield"], split["int_no_active_span"],
        )
    results["all"] = summarize(
        pooled["eot"], pooled["eot_smooth"], pooled["eot_unanswered"],
        pooled["int_yield"], pooled["int_no_active_span"],
    )

    for split_name, r in results.items():
        e, m, g, y = (r["eot_floor_transfer_s"], r["eot_smooth_transfer_s"],
                      r["eot_gap_only_s"], r["int_yield_s"])
        print(
            f"{split_name:>5s}: EOT all p50 {e['p50']:+.3f}s (n={e['n']}) | "
            f"smooth p50 {m['p50']:+.3f}s (n={m['n']}, "
            f"{r['eot_smooth_share_overlap']:.1%} overlap) | "
            f"gap-only p50 {g['p50']:+.3f}s mean {g['mean']:+.3f}s (n={g['n']}) | "
            f"INT yield p50 {y['p50']:+.3f}s mean {y['mean']:+.3f}s "
            f"(n={y['n']}, skipped {r['int_no_active_span']})"
        )

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
