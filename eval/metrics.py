#!/usr/bin/env python3
"""Evaluate a baseline's predictions against the consensus gold.

Inputs:
    Gold:  stats_out/consensus/<task_id>.jsonl
    Pred:  predictions/<baseline>/<task_id>.jsonl, same schema:
               {"speaker": int, "time": float, "label": str}
           (predictions are points in time, not intervals)

Metrics:
    EOT
        Gold EOT time = `end` of every consensus `Turn` event.
        TP            : predicted EOT in (t_gold, t_gold + WINDOW_S]
        FP-premature  : predicted EOT in (t_gold - WINDOW_S, t_gold]
                        (model cut off the speaker)
        FP-spurious   : predicted EOT outside any gold EOT window
        FN            : gold EOT with no TP within (t_gold, t_gold + WINDOW_S]
        Latency       : t_pred - t_gold for each TP (always in [0, WINDOW_S])

    Interruption
        Same shape as EOT, on gold `Interruption` events.

    Confusion (predicted Interruption)
        For each predicted `Interruption`, what gold canonical label, if any,
        lies in (t_pred - WINDOW_S, t_pred + WINDOW_S]? Reported as a 6-way
        histogram: {Interruption, Backchannel, Overlap, Laughter,
        NonContent, None}. A model that confuses backchannels with
        interruptions will show up clearly in the Backchannel column.

All scoring is restricted to consensus events; regions without consensus are
ignored (no predictions in those spans count for or against the model).
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


WINDOW_S = 2.0  # grace window for EOT / Interruption detection
CANONICAL_LABELS = ("Turn", "Interruption", "Backchannel", "Overlap", "Laughter", "NonContent")
NON_INTERRUPTION = ("Backchannel", "Overlap", "Laughter", "NonContent")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def gold_eot_times(gold: list[dict]) -> dict[int, list[float]]:
    """Per speaker, the end-times of consensus Turn events."""
    out: dict[int, list[float]] = defaultdict(list)
    for ev in gold:
        if ev["label"] == "Turn":
            out[ev["speaker"]].append(ev["end"])
    for sp in out:
        out[sp].sort()
    return out


def gold_event_times(gold: list[dict], label: str) -> dict[int, list[float]]:
    """Per speaker, the start-times of consensus events with `label`."""
    out: dict[int, list[float]] = defaultdict(list)
    for ev in gold:
        if ev["label"] == label:
            out[ev["speaker"]].append(ev["start"])
    for sp in out:
        out[sp].sort()
    return out


def score_point_event(pred_times: list[float], gold_times: list[float]
                      ) -> dict[str, float | list[float]]:
    """Greedy 1:1 matching. Each gold event consumes at most one prediction in
    its (t_g, t_g + WINDOW_S] window; the earliest unused prediction in that
    window wins (lowest latency)."""
    pred_sorted = sorted(pred_times)
    used = [False] * len(pred_sorted)
    tp_latencies: list[float] = []
    fn = 0
    for t_g in gold_times:
        match_i = None
        for i, t_p in enumerate(pred_sorted):
            if used[i]:
                continue
            if t_p <= t_g:
                continue
            if t_p > t_g + WINDOW_S:
                break  # sorted — no later pred can match either
            match_i = i
            break
        if match_i is None:
            fn += 1
        else:
            used[match_i] = True
            tp_latencies.append(pred_sorted[match_i] - t_g)

    tp = len(tp_latencies)
    fp_premature = 0
    fp_spurious = 0
    for i, t_p in enumerate(pred_sorted):
        if used[i]:
            continue
        # was it premature (in (t_g - WINDOW_S, t_g] for some gold)?
        if any(t_g - WINDOW_S < t_p <= t_g for t_g in gold_times):
            fp_premature += 1
        else:
            fp_spurious += 1

    return {
        "tp": tp,
        "fn": fn,
        "fp_premature": fp_premature,
        "fp_spurious": fp_spurious,
        "fp_total": fp_premature + fp_spurious,
        "latencies_s": tp_latencies,
    }


def confusion_for_label(pred: list[dict], gold: list[dict], target: str
                        ) -> dict[str, int]:
    """For each prediction with label==target, find what gold canonical event
    (if any) is within (t_pred - WINDOW_S, t_pred + WINDOW_S]. Reports counts."""
    by_sp: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    for ev in gold:
        by_sp[ev["speaker"]].append((ev["start"], ev["end"], ev["label"]))
    for sp in by_sp:
        by_sp[sp].sort()

    cm: Counter = Counter()
    for p in pred:
        if p["label"] != target:
            continue
        t_p, sp = p["time"], p["speaker"]
        match = "None"
        for s, e, lbl in by_sp.get(sp, []):
            if t_p - WINDOW_S < s <= t_p + WINDOW_S or s <= t_p <= e:
                match = lbl
                break
        cm[match] += 1
    return dict(cm)


def aggregate(scores: dict, key: str) -> dict:
    total = {"tp": 0, "fn": 0, "fp_premature": 0, "fp_spurious": 0, "fp_total": 0}
    latencies: list[float] = []
    for s in scores.values():
        for k in total:
            total[k] += s[key][k]
        latencies.extend(s[key]["latencies_s"])
    p = total["tp"] / max(total["tp"] + total["fp_total"], 1)
    r = total["tp"] / max(total["tp"] + total["fn"], 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    lat = sorted(latencies)
    pct = lambda q: lat[int(q * (len(lat) - 1))] if lat else float("nan")
    return {
        **total,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "latency_mean_s": round(sum(latencies) / len(latencies), 4) if latencies else None,
        "latency_p50_s": round(pct(0.5), 4) if latencies else None,
        "latency_p95_s": round(pct(0.95), 4) if latencies else None,
        "n_predictions_matched": len(latencies),
    }


def evaluate_baseline(consensus_dir: Path, pred_dir: Path) -> dict:
    per_sample: dict[str, dict] = {}
    for gold_file in sorted(consensus_dir.glob("*.jsonl")):
        if gold_file.name.startswith("_"):
            continue
        task_id = gold_file.stem
        pred_file = pred_dir / f"{task_id}.jsonl"
        if not pred_file.exists():
            continue

        gold = load_jsonl(gold_file)
        pred = load_jsonl(pred_file)

        eot_score = {"tp": 0, "fn": 0, "fp_premature": 0, "fp_spurious": 0,
                     "fp_total": 0, "latencies_s": []}
        for sp, g_times in gold_eot_times(gold).items():
            p_times = [p["time"] for p in pred if p["speaker"] == sp and p["label"] == "EOT"]
            s = score_point_event(p_times, g_times)
            for k in ("tp", "fn", "fp_premature", "fp_spurious", "fp_total"):
                eot_score[k] += s[k]
            eot_score["latencies_s"].extend(s["latencies_s"])

        int_score = {"tp": 0, "fn": 0, "fp_premature": 0, "fp_spurious": 0,
                     "fp_total": 0, "latencies_s": []}
        for sp, g_times in gold_event_times(gold, "Interruption").items():
            p_times = [p["time"] for p in pred
                       if p["speaker"] == sp and p["label"] == "Interruption"]
            s = score_point_event(p_times, g_times)
            for k in ("tp", "fn", "fp_premature", "fp_spurious", "fp_total"):
                int_score[k] += s[k]
            int_score["latencies_s"].extend(s["latencies_s"])

        per_sample[task_id] = {
            "EOT": eot_score,
            "Interruption": int_score,
            "interruption_confusion": confusion_for_label(pred, gold, "Interruption"),
        }

    summary = {
        "n_samples_scored": len(per_sample),
        "EOT": aggregate(per_sample, "EOT"),
        "Interruption": aggregate(per_sample, "Interruption"),
        "interruption_confusion_total": dict(sum(
            (Counter(s["interruption_confusion"]) for s in per_sample.values()), Counter()
        )),
    }
    return {"summary": summary, "per_sample": per_sample}


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: metrics.py <consensus_dir> <predictions_dir>", file=sys.stderr)
        return 2
    result = evaluate_baseline(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
