#!/usr/bin/env python3
"""Sesame-aligned EOT metric for the tt-benchmark gold.

Two numbers, one head (`agent_should_speak` -> `eot_score_speaker_K`):

  1. EOT latency at Sesame's 8 operating points.
       For each target latency T in {80, 160, 240, 320, 400, 480, 560, 640} ms,
       find the highest threshold whose aggregate median TP latency is <= T.
       Report (target_ms, threshold, recall, actual median/p90/p95 latency).
       TP definition matches Sesame: first crossing of pred > thr at
       t >= t_gold within [t_gold, t_gold + 2.0s].

  2. False-EOT rate at the SAME operating points (Sesame `eot_fpr_seg`).
       Within-turn pauses = gaps between consecutive events inside the
       same reconciled turn region (same speaker, no floor change).
       Per-pause binary trigger: any frame in [pause_start, pause_end]
       with pred > thr -> 1 FP for that pause.
       fpr_seg = #FP_pauses / #total_pauses.

Inputs (defaults):
  --run-dir       predictions/<run>/   (manifest.json + traces/<task>.npz)
  --eot-dir       stats_out/consensus_eot
  --regions-dir   stats_out/turn_regions

The score channel evaluated is `eot_score_speaker_K` for both speakers.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from eval.submission_format import load_manifest, load_traces  # noqa: E402


# Operating points from sesame/ml/core/evals/cd/threshold_analysis.py:38
TARGET_LATENCIES_MS = (80, 160, 240, 320, 400, 480, 560, 640)
TAU_MAX_S = 2.00  # rightmost edge of the TP search window past t_gold


def load_gold_eots(eot_dir: Path, task_id: str
                   ) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {1: [], 2: []}
    p = eot_dir / f"{task_id}.jsonl"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        out[ev["speaker"]].append(float(ev["time"]))
    return out


def load_within_turn_pauses(regions_dir: Path, task_id: str
                             ) -> dict[int, list[tuple[float, float]]]:
    """For each speaker, return the [(pause_start, pause_end)] inside each
    reconciled turn region — i.e. the gaps between consecutive events of
    that region's `events` list. These are the USER_PAUSE-equivalents:
    silences mid-turn, no floor change."""
    out: dict[int, list[tuple[float, float]]] = {1: [], 2: []}
    p = regions_dir / f"{task_id}.jsonl"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sp = r["speaker"]
        events = r.get("events", [])
        for i in range(len(events) - 1):
            a, b = events[i], events[i + 1]
            ge, gs = float(a["end"]), float(b["start"])
            if gs > ge + 1e-6:
                out[sp].append((ge, gs))
    return out


@dataclass
class TaskAcc:
    # accumulators are per-threshold; index = threshold position in sweep
    tp_count: np.ndarray = field(default_factory=lambda: np.array([]))
    fn_count: np.ndarray = field(default_factory=lambda: np.array([]))
    # latencies_ms[k] is the list of TP latencies at threshold k across all tasks
    latencies_ms: list[list[float]] = field(default_factory=list)
    pause_fp_count: np.ndarray = field(default_factory=lambda: np.array([]))
    pause_total: int = 0


def score_task(
    gold_eots: dict[int, list[float]],
    pauses: dict[int, list[tuple[float, float]]],
    score_by_speaker: dict[int, np.ndarray],
    fps: float,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[list[float]], np.ndarray, int]:
    """Return (tp[T], fn[T], latencies[T][...], pause_fp[T], pause_total)."""
    T = len(thresholds)
    tp = np.zeros(T, dtype=np.int64)
    fn = np.zeros(T, dtype=np.int64)
    latencies: list[list[float]] = [[] for _ in range(T)]
    pause_fp = np.zeros(T, dtype=np.int64)
    pause_total = 0

    for sp in (1, 2):
        scores = score_by_speaker[sp]
        n = len(scores)
        # Per gold EOT, find first frame >= t_gold in [t_gold, t_gold + TAU_MAX_S]
        # where pred > thr (loop over thresholds).
        for t_gold in gold_eots[sp]:
            lo = max(0, int(np.ceil(t_gold * fps)))
            hi = min(n - 1, int(np.ceil((t_gold + TAU_MAX_S) * fps)))
            if lo > hi:
                fn += 1
                continue
            window = scores[lo:hi + 1]
            for k, thr in enumerate(thresholds):
                crossings = np.where(window > thr)[0]
                if len(crossings) == 0:
                    fn[k] += 1
                else:
                    fi = lo + int(crossings[0])
                    tp[k] += 1
                    latencies[k].append((fi / fps - t_gold) * 1000.0)

        # Per within-turn pause, binary trigger across thresholds.
        for ps, pe in pauses[sp]:
            lo = max(0, int(np.ceil(ps * fps)))
            hi = min(n - 1, int(np.floor(pe * fps)))
            pause_total += 1
            if lo > hi:
                continue
            window = scores[lo:hi + 1]
            wmax = float(window.max())
            for k, thr in enumerate(thresholds):
                if wmax > thr:
                    pause_fp[k] += 1
    return tp, fn, latencies, pause_fp, pause_total


def aggregate(run_dir: Path, eot_dir: Path, regions_dir: Path,
              thresholds: np.ndarray, only: set[str] | None = None
              ) -> dict:
    manifest = load_manifest(run_dir)
    fps = manifest.frame_rate_hz
    T = len(thresholds)
    tp_g = np.zeros(T, dtype=np.int64)
    fn_g = np.zeros(T, dtype=np.int64)
    pause_fp_g = np.zeros(T, dtype=np.int64)
    pause_total_g = 0
    lat_g: list[list[float]] = [[] for _ in range(T)]

    task_ids = manifest.task_ids if only is None else [
        t for t in manifest.task_ids if t in only]
    n_tasks = 0

    for tid in task_ids:
        if not (eot_dir / f"{tid}.jsonl").exists():
            continue
        gold = load_gold_eots(eot_dir, tid)
        pauses = load_within_turn_pauses(regions_dir, tid)
        traces = load_traces(run_dir, tid)
        scores = {1: traces["eot_score_speaker_1"],
                  2: traces["eot_score_speaker_2"]}
        tp, fn, lat, pfp, pt_total = score_task(gold, pauses, scores, fps, thresholds)
        tp_g += tp; fn_g += fn; pause_fp_g += pfp; pause_total_g += pt_total
        for k in range(T):
            lat_g[k].extend(lat[k])
        n_tasks += 1

    # Per-threshold summary
    out_rows = []
    for k, thr in enumerate(thresholds):
        lats = np.asarray(lat_g[k], dtype=np.float64) if lat_g[k] else np.array([])
        tp_k, fn_k = int(tp_g[k]), int(fn_g[k])
        fp_k = int(pause_fp_g[k])  # Sesame F1 uses per-segment FP, not frame FPs
        recall = tp_k / max(1, tp_k + fn_k)
        precision = tp_k / max(1, tp_k + fp_k)
        f1 = 2*precision*recall / (precision + recall) if (precision + recall) else 0.0
        fpr_seg = pause_fp_g[k] / max(1, pause_total_g)
        out_rows.append({
            "thr": round(float(thr), 4),
            "tp": tp_k, "fn": fn_k,
            "recall": round(float(recall), 4),
            "precision": round(float(precision), 4),
            "f1": round(float(f1), 4),
            "median_ms": round(float(np.median(lats)), 1) if lats.size else None,
            "p90_ms": round(float(np.percentile(lats, 90)), 1) if lats.size else None,
            "p95_ms": round(float(np.percentile(lats, 95)), 1) if lats.size else None,
            "pause_fp": int(pause_fp_g[k]),
            "pause_total": int(pause_total_g),
            "fpr_seg": round(float(fpr_seg), 4),
        })

    # Operating points: highest threshold whose aggregate median latency <= target.
    operating_points = []
    for target in TARGET_LATENCIES_MS:
        # eligible thresholds = those with a defined median <= target
        eligible = [(k, r) for k, r in enumerate(out_rows)
                    if r["median_ms"] is not None and r["median_ms"] <= target]
        if not eligible:
            operating_points.append({
                "target_ms": target, "threshold": None, "note": "unreachable"})
            continue
        # pick the highest threshold (tightest)
        k, r = max(eligible, key=lambda kr: kr[1]["thr"])
        operating_points.append({
            "target_ms": target, **r,
        })

    return {
        "run": manifest.run_name,
        "baseline": manifest.baseline,
        "split": manifest.split,
        "n_tasks": n_tasks,
        "operating_points": operating_points,
        "sweep": out_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--eot-dir", type=Path,
                    default=Path("stats_out/consensus_eot"))
    ap.add_argument("--regions-dir", type=Path,
                    default=Path("stats_out/turn_regions"))
    ap.add_argument("--split-file", type=Path, default=None,
                    help="Restrict to task_ids listed here (one per line).")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON.")
    args = ap.parse_args()

    only = None
    if args.split_file:
        only = {ln.strip() for ln in args.split_file.read_text().splitlines()
                if ln.strip() and not ln.startswith("#")}

    # Sesame's default sweep — matches DEFAULT_SWEEP_THRESHOLDS in
    # threshold_analysis.py:36 (0.01, 0.05 step from 0.05 to 0.95, 0.99).
    thresholds = np.array([0.01] + [round(0.05 * i, 2) for i in range(1, 20)] + [0.99],
                          dtype=np.float64)

    res = aggregate(args.run_dir, args.eot_dir, args.regions_dir,
                    thresholds, only=only)
    if args.json:
        print(json.dumps(res, indent=2))
        return 0

    print(f"== {res['run']}  ({res['baseline']}, split={res['split']}, "
          f"n_tasks={res['n_tasks']}) ==")
    print()
    # Headline columns match `compare_models.py` (lines 510-573): Speak F1,
    # Speak median (ms), Speak p95 (ms). We add the latency target column,
    # plus fpr_seg as an auxiliary column (the per-pause false-EOT rate)
    # since that's the failure mode the benchmark is designed to expose.
    print("EOT operating points (Sesame-aligned, compare_models.py format):")
    print(f"{'target':>7} {'thr':>6} {'Speak F1':>9} "
          f"{'Speak med':>10} {'Speak p95':>10}  "
          f"{'recall':>7} {'fpr_seg':>8}")
    for op in res["operating_points"]:
        if "thr" not in op:
            print(f"{op['target_ms']:>7d} {'-':>6} (unreachable: no threshold "
                  f"achieves median latency <= {op['target_ms']}ms)")
            continue
        print(f"{op['target_ms']:>7d} {op['thr']:>6.2f} "
              f"{op['f1']*100:>8.1f}% "
              f"{op['median_ms']:>9.0f}ms {op['p95_ms']:>9.0f}ms  "
              f"{op['recall']:>7.3f} {op['fpr_seg']:>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
