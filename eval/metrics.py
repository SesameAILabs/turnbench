#!/usr/bin/env python3
"""Score a baseline's unified-format predictions against the consensus gold.

Inputs:
    Predictions dir (`run_dir`) — `manifest.json` + `traces/<task_id>.npz`,
        produced by `baselines/runner.py` (see eval/submission_format.py and
        docs/SUBMISSION_FORMAT.md).
    Consensus dir — `stats_out/consensus/<task_id>.jsonl` produced by
        eval/consensus.py. Per-event lines: `{"speaker", "start", "end",
        "label", "raw": ...}`.

What we score (initial MVP, two tracks):

  EOT detection
      Gold event = the END time of a consensus `Turn` event for speaker K.
      Score array = `eot_score_speaker_K`.
      For each gold event, locate the first frame in the eval window
      [t_gold - tau_pre, t_gold + tau_max] whose score crosses `threshold`.
      Hit -> TP; miss -> FN.

  Interruption detection
      Gold event = the START time of a consensus `Interruption` event for
      speaker K (i.e., speaker K is interrupting). Score array =
      `interruption_score_speaker_K`. Same TP/FN scoring as EOT.

  False positives
      Each threshold-crossing in the score that does NOT lie inside any
      gold event window of the corresponding track is counted as one FP,
      with a 1 s refractory so a single sustained high segment is counted
      once rather than per frame.

Metrics produced per track:
    tp, fn, fp, precision, recall, f1
    latency_p10_ms / latency_p50_ms / latency_p90_ms over TP detections,
        where latency = (frame_time - t_gold) * 1000.

Usage:
    python -m eval.metrics \\
        --run-dir   <run dir with manifest.json + traces/> \\
        --gold-dir  stats_out/consensus \\
        --threshold 0.5

    # Sweep:
    python -m eval.metrics --run-dir ... --gold-dir ... --sweep 0.05:0.95:0.05
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

# allow `python eval/metrics.py` and `python -m eval.metrics`
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from eval.submission_format import load_manifest, load_traces  # noqa: E402


TAU_PRE_S = 0.25       # speculative-early tolerance
TAU_MAX_S = 2.00       # latency tolerance after gold event
FP_REFRACTORY_S = 1.0  # collapse repeated above-threshold detections


@dataclass
class TrackScore:
    tp: int = 0
    fn: int = 0
    fp: int = 0
    latencies_ms: list[float] = None  # populated lazily

    def __post_init__(self) -> None:
        if self.latencies_ms is None:
            self.latencies_ms = []

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def latency_percentiles_ms(self) -> dict[str, float]:
        if not self.latencies_ms:
            return {"latency_p10_ms": float("nan"),
                    "latency_p50_ms": float("nan"),
                    "latency_p90_ms": float("nan")}
        arr = np.asarray(self.latencies_ms, dtype=np.float64)
        return {
            "latency_p10_ms": float(np.percentile(arr, 10)),
            "latency_p50_ms": float(np.percentile(arr, 50)),
            "latency_p90_ms": float(np.percentile(arr, 90)),
        }

    def as_dict(self) -> dict:
        return {
            "tp": self.tp, "fn": self.fn, "fp": self.fp,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            **{k: round(v, 1) for k, v in self.latency_percentiles_ms().items()},
        }


def load_gold_events(consensus_path: Path) -> list[dict]:
    return [json.loads(line) for line in consensus_path.read_text().splitlines() if line.strip()]


def gold_eot_events(gold: Iterable[dict]) -> list[tuple[float, int]]:
    """Return (t_gold_seconds, speaker) for each consensus Turn event end."""
    return [(ev["end"], ev["speaker"]) for ev in gold if ev["label"] == "Turn"]


def gold_interruption_events(gold: Iterable[dict]) -> list[tuple[float, int]]:
    """Return (t_gold_seconds, speaker) for each consensus Interruption start."""
    return [(ev["start"], ev["speaker"]) for ev in gold if ev["label"] == "Interruption"]


def score_track(
    gold: list[tuple[float, int]],
    score_by_speaker: dict[int, np.ndarray],
    frame_rate_hz: float,
    threshold: float,
    *,
    tau_pre_s: float = TAU_PRE_S,
    tau_max_s: float = TAU_MAX_S,
    fp_refractory_s: float = FP_REFRACTORY_S,
) -> TrackScore:
    out = TrackScore()
    fps = frame_rate_hz

    # Match each gold event to the first frame in its eval window where the
    # score crosses the threshold.
    gold_windows: dict[int, list[tuple[int, int]]] = {1: [], 2: []}
    for t_gold, speaker in gold:
        scores = score_by_speaker[speaker]
        lo = max(0, int(np.floor((t_gold - tau_pre_s) * fps)))
        hi = min(len(scores) - 1, int(np.ceil((t_gold + tau_max_s) * fps)))
        gold_windows[speaker].append((lo, hi))
        if lo > hi:
            out.fn += 1
            continue
        window = scores[lo:hi + 1]
        crossing = np.where(window > threshold)[0]
        if len(crossing) == 0:
            out.fn += 1
        else:
            frame_idx = lo + int(crossing[0])
            t_pred = frame_idx / fps
            out.tp += 1
            out.latencies_ms.append((t_pred - t_gold) * 1000.0)

    # Count FPs: threshold-crossings outside any gold window, with refractory.
    refractory_frames = max(1, int(round(fp_refractory_s * fps)))
    for speaker in (1, 2):
        scores = score_by_speaker[speaker]
        in_gold = np.zeros(len(scores), dtype=bool)
        for lo, hi in gold_windows[speaker]:
            in_gold[lo:hi + 1] = True
        above = scores > threshold
        spurious = above & ~in_gold
        i = 0
        while i < len(spurious):
            if spurious[i]:
                out.fp += 1
                i += refractory_frames
            else:
                i += 1
    return out


def score_run(
    run_dir: Path,
    gold_dir: Path,
    threshold: float,
    *,
    only_task_ids: set[str] | None = None,
) -> dict:
    """Score a single run at a single threshold; return aggregate metrics
    plus a per-task breakdown."""
    manifest = load_manifest(run_dir)
    fps = manifest.frame_rate_hz
    eot_total = TrackScore()
    int_total = TrackScore()
    per_task: dict[str, dict] = {}

    task_ids = manifest.task_ids if only_task_ids is None else [
        t for t in manifest.task_ids if t in only_task_ids
    ]

    for tid in task_ids:
        gold_path = gold_dir / f"{tid}.jsonl"
        if not gold_path.exists():
            continue
        gold = load_gold_events(gold_path)
        traces = load_traces(run_dir, tid)
        eot_scores = {1: traces["eot_score_speaker_1"], 2: traces["eot_score_speaker_2"]}
        int_scores = {1: traces["interruption_score_speaker_1"],
                      2: traces["interruption_score_speaker_2"]}

        eot = score_track(gold_eot_events(gold), eot_scores, fps, threshold)
        intr = score_track(gold_interruption_events(gold), int_scores, fps, threshold)

        per_task[tid] = {"EOT": eot.as_dict(), "Interruption": intr.as_dict()}

        eot_total.tp += eot.tp; eot_total.fn += eot.fn; eot_total.fp += eot.fp
        eot_total.latencies_ms.extend(eot.latencies_ms)
        int_total.tp += intr.tp; int_total.fn += intr.fn; int_total.fp += intr.fp
        int_total.latencies_ms.extend(intr.latencies_ms)

    return {
        "run": manifest.run_name,
        "baseline": manifest.baseline,
        "checkpoint": manifest.checkpoint,
        "split": manifest.split,
        "threshold": threshold,
        "n_tasks_scored": len(per_task),
        "EOT": eot_total.as_dict(),
        "Interruption": int_total.as_dict(),
        "per_task": per_task,
    }


def parse_sweep(spec: str) -> list[float]:
    a, b, c = (float(x) for x in spec.split(":"))
    out = []
    x = a
    while x <= b + 1e-9:
        out.append(round(x, 6))
        x += c
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="predictions/<run-name>/ directory")
    ap.add_argument("--gold-dir", required=True, type=Path,
                    help="stats_out/consensus/")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--sweep", default=None,
                    help='Sweep thresholds, e.g. "0.1:0.9:0.1"')
    ap.add_argument("--split-file", type=Path, default=None,
                    help="Optional task_id list (one per line)")
    args = ap.parse_args()

    only = None
    if args.split_file is not None:
        only = {ln.strip() for ln in args.split_file.read_text().splitlines() if ln.strip()}

    thresholds = parse_sweep(args.sweep) if args.sweep else [args.threshold]
    summaries = []
    for thr in thresholds:
        r = score_run(args.run_dir, args.gold_dir, thr, only_task_ids=only)
        del r["per_task"]
        summaries.append(r)

    # Pretty-print
    print(f"\nRun: {summaries[0]['run']}  ({summaries[0]['n_tasks_scored']} tasks)")
    header = f"{'thr':>5s}  {'EOT P':>7s} {'EOT R':>7s} {'EOT F1':>7s} {'EOT p50ms':>10s}   {'Int P':>7s} {'Int R':>7s} {'Int F1':>7s} {'Int p50ms':>10s}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        e, i = s["EOT"], s["Interruption"]
        print(f"{s['threshold']:>5.2f}  "
              f"{e['precision']:>7.3f} {e['recall']:>7.3f} {e['f1']:>7.3f} "
              f"{e['latency_p50_ms']:>10.1f}   "
              f"{i['precision']:>7.3f} {i['recall']:>7.3f} {i['f1']:>7.3f} "
              f"{i['latency_p50_ms']:>10.1f}")

    print(json.dumps(summaries, indent=2)) if False else None
    return 0


if __name__ == "__main__":
    sys.exit(main())
