#!/usr/bin/env python3
"""Score EOT predictions against the new gold sets.

Gold sources (produced upstream by eval/consensus_eot.py and
eval/turn_regions.py):

  Positives (events to detect):
    stats_out/consensus_eot/<task>.jsonl
      {"speaker": int, "time": float, "raw": {"a","b","c": float}}
    -> 3-of-3 reconciled floor-handover times per speaker.

  No-fire zones (regions where a model prediction is a false positive):
    stats_out/turn_regions/<task>.jsonl
      {"speaker": int, "region_start": float, "region_end": float, ...}
    -> contiguous same-speaker turn regions.

For each speaker channel of each task:

  TP    a threshold-crossing of `eot_score_speaker_K` lies inside any
        EOT window [t_gold - TAU_PRE_S, t_gold + TAU_MAX_S]. Earliest
        crossing per gold event; latency = (t_pred - t_gold)*1000.

  FN    a gold EOT with no crossing inside its window.

  FP-failed   a crossing inside a turn region (same speaker) but NOT
              inside any EOT window. "Failed EOT" / hard false positive
              — the model said the speaker ended their turn but they
              didn't.

  FP-noise    a crossing OUTSIDE any turn region (same speaker) and
              not inside any EOT window. Softer false positive — fire
              outside any in-progress turn.

A 1 s refractory collapses consecutive above-threshold frames into one
FP so a single sustained high segment counts once.

Usage:
    python -m eval.metrics_eot \
        --run-dir   /path/to/predictions/<run>/ \
        --threshold 0.5
    python -m eval.metrics_eot --run-dir ... --sweep 0.05:0.95:0.05
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


TAU_PRE_S = 0.25
TAU_MAX_S = 2.00
FP_REFRACTORY_S = 1.0


@dataclass
class EotScore:
    tp: int = 0
    fn: int = 0
    fp_failed: int = 0   # crossings inside a turn region, outside EOT windows
    fp_noise: int = 0    # crossings outside any region, outside EOT windows
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def fp_total(self) -> int:
        return self.fp_failed + self.fp_noise

    @property
    def precision(self) -> float:
        d = self.tp + self.fp_total
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        lat = np.asarray(self.latencies_ms or [np.nan])
        return {
            "tp": self.tp, "fn": self.fn,
            "fp_failed": self.fp_failed, "fp_noise": self.fp_noise,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "latency_p10_ms": round(float(np.nanpercentile(lat, 10)), 1),
            "latency_p50_ms": round(float(np.nanpercentile(lat, 50)), 1),
            "latency_p90_ms": round(float(np.nanpercentile(lat, 90)), 1),
        }


def load_gold_eots(eot_dir: Path, task_id: str
                   ) -> dict[int, list[float]]:
    """{speaker: [t_eot, ...]}."""
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


def load_regions(regions_dir: Path, task_id: str
                 ) -> dict[int, list[tuple[float, float]]]:
    """{speaker: [(region_start, region_end), ...]}."""
    out: dict[int, list[tuple[float, float]]] = {1: [], 2: []}
    p = regions_dir / f"{task_id}.jsonl"
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["speaker"]].append((float(r["region_start"]),
                                   float(r["region_end"])))
    return out


def _frames_to_mask(intervals: list[tuple[float, float]],
                    n_frames: int, fps: float) -> np.ndarray:
    m = np.zeros(n_frames, dtype=bool)
    for s, e in intervals:
        lo = max(0, int(np.floor(s * fps)))
        hi = min(n_frames - 1, int(np.ceil(e * fps)))
        if lo <= hi:
            m[lo:hi + 1] = True
    return m


def score_task(
    gold_eots: dict[int, list[float]],
    regions: dict[int, list[tuple[float, float]]],
    score_by_speaker: dict[int, np.ndarray],
    fps: float,
    threshold: float,
    *,
    tau_pre_s: float = TAU_PRE_S,
    tau_max_s: float = TAU_MAX_S,
    fp_refractory_s: float = FP_REFRACTORY_S,
) -> EotScore:
    out = EotScore()
    refr = max(1, int(round(fp_refractory_s * fps)))

    for sp in (1, 2):
        scores = score_by_speaker[sp]
        n = len(scores)
        above = scores > threshold

        # 1. Match each gold EOT to the earliest in-window crossing.
        eot_windows: list[tuple[int, int]] = []
        for t_gold in gold_eots[sp]:
            lo = max(0, int(np.floor((t_gold - tau_pre_s) * fps)))
            hi = min(n - 1, int(np.ceil((t_gold + tau_max_s) * fps)))
            eot_windows.append((lo, hi))
            if lo > hi:
                out.fn += 1
                continue
            crossing = np.where(above[lo:hi + 1])[0]
            if len(crossing) == 0:
                out.fn += 1
            else:
                fi = lo + int(crossing[0])
                out.tp += 1
                out.latencies_ms.append((fi / fps - t_gold) * 1000.0)

        # 2. Classify every above-threshold frame (modulo refractory) as
        #    TP-window / FP-failed / FP-noise.
        in_eot = _frames_to_mask([(lo / fps, hi / fps)
                                   for lo, hi in eot_windows], n, fps)
        in_region = _frames_to_mask(regions[sp], n, fps)
        i = 0
        while i < n:
            if not above[i]:
                i += 1
                continue
            if in_eot[i]:
                # counted as TP already (or its window is empty); skip past refractory
                i += refr
                continue
            if in_region[i]:
                out.fp_failed += 1
            else:
                out.fp_noise += 1
            i += refr
    return out


def score_run(run_dir: Path, eot_dir: Path, regions_dir: Path,
              threshold: float,
              *, only_task_ids: set[str] | None = None) -> dict:
    manifest = load_manifest(run_dir)
    fps = manifest.frame_rate_hz
    total = EotScore()
    per_task: dict[str, dict] = {}

    task_ids = manifest.task_ids if only_task_ids is None else [
        t for t in manifest.task_ids if t in only_task_ids]

    for tid in task_ids:
        if not (eot_dir / f"{tid}.jsonl").exists():
            continue
        gold = load_gold_eots(eot_dir, tid)
        regs = load_regions(regions_dir, tid)
        traces = load_traces(run_dir, tid)
        scores = {1: traces["eot_score_speaker_1"],
                  2: traces["eot_score_speaker_2"]}
        s = score_task(gold, regs, scores, fps, threshold)
        per_task[tid] = s.as_dict()
        total.tp += s.tp; total.fn += s.fn
        total.fp_failed += s.fp_failed; total.fp_noise += s.fp_noise
        total.latencies_ms.extend(s.latencies_ms)

    return {
        "run": manifest.run_name,
        "baseline": manifest.baseline,
        "checkpoint": manifest.checkpoint,
        "split": manifest.split,
        "threshold": threshold,
        "n_tasks_scored": len(per_task),
        "EOT": total.as_dict(),
        "per_task": per_task,
    }


def parse_sweep(spec: str) -> list[float]:
    a, b, c = (float(x) for x in spec.split(":"))
    out, x = [], a
    while x <= b + 1e-9:
        out.append(round(x, 6))
        x += c
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--eot-dir", type=Path,
                    default=Path("stats_out/consensus_eot"))
    ap.add_argument("--regions-dir", type=Path,
                    default=Path("stats_out/turn_regions"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--sweep", type=str, default=None,
                    help="lo:hi:step, e.g. 0.05:0.95:0.05")
    ap.add_argument("--split-file", type=Path, default=None,
                    help="Restrict to task_ids listed here (one per line).")
    args = ap.parse_args()

    only = None
    if args.split_file:
        only = {ln.strip() for ln in args.split_file.read_text().splitlines()
                if ln.strip() and not ln.startswith("#")}

    if args.sweep:
        rows = []
        for thr in parse_sweep(args.sweep):
            r = score_run(args.run_dir, args.eot_dir, args.regions_dir, thr,
                          only_task_ids=only)
            rows.append({"threshold": thr,
                         "p": r["EOT"]["precision"], "r": r["EOT"]["recall"],
                         "f1": r["EOT"]["f1"], "tp": r["EOT"]["tp"],
                         "fn": r["EOT"]["fn"],
                         "fp_failed": r["EOT"]["fp_failed"],
                         "fp_noise": r["EOT"]["fp_noise"],
                         "lat_p50_ms": r["EOT"]["latency_p50_ms"],
                         "lat_p90_ms": r["EOT"]["latency_p90_ms"]})
        print(f"{'thr':>5} {'P':>6} {'R':>6} {'F1':>6} "
              f"{'TP':>5} {'FN':>5} {'FPf':>5} {'FPn':>5} "
              f"{'L50':>6} {'L90':>6}")
        best = max(rows, key=lambda r: r["f1"])
        for r in rows:
            mark = "  *" if r is best else ""
            print(f"{r['threshold']:>5.2f} {r['p']:>6.3f} {r['r']:>6.3f} "
                  f"{r['f1']:>6.3f} {r['tp']:>5d} {r['fn']:>5d} "
                  f"{r['fp_failed']:>5d} {r['fp_noise']:>5d} "
                  f"{r['lat_p50_ms']:>6.0f} {r['lat_p90_ms']:>6.0f}{mark}")
    else:
        r = score_run(args.run_dir, args.eot_dir, args.regions_dir,
                      args.threshold, only_task_ids=only)
        print(json.dumps({k: v for k, v in r.items() if k != "per_task"},
                         indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
