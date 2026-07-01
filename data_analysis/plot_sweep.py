#!/usr/bin/env python3
"""Render the dev threshold-sweep diagnostic straight from a probs JSON file.

The probs file is swept and scored in memory (eval.sweep) against the dev gold —
no intermediate CSV — into a per-model twin-axis diagnostic: median latency (left,
ms) and the two rate metrics recall + fp_rate (right, 0-1) vs the decision
threshold, with the operating point (highest recall at fp_rate <= budget) marked.

The three curves are read together: recall and fp_rate are the benefit/cost the
threshold trades off, latency is the price of the recall you keep. Latency is
plotted only where recall clears the floor — it is measured over true positives
alone, so where almost nothing fires the median is a meaningless average over a
handful of easy events (and can even go negative, from speculative pre-fires
inside the matching tolerance). Neither recall nor fp_rate is monotone in the
threshold under the rising-edge commit rule, so the curves need not be smooth.

    uv run --extra eval --extra plot python data_analysis/plot_sweep.py \
        baselines/<name>/probs-eot.json --out fig.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `eval` imports

from eval.data import DEV_DATASET, resolve_dataset  # noqa: E402
from eval.sweep import SweepRow, load_probs, operating_point, sweep  # noqa: E402

OLIVE = "#6e7b3d"    # latency (left axis)
STEEL = "#2b6cb0"    # recall (right axis)
CRIMSON = "#b23a48"  # fp_rate (right axis)
BG = "#eef0e0"


def _latency_masked(rows: list[SweepRow], recall_floor: float) -> list[float]:
    # latency is measured over TPs only — meaningless where almost nothing fires.
    return [r.lat_p50 if r.recall >= recall_floor else float("nan") for r in rows]


def plot_single(rows: list[SweepRow], task: str, out: Path, *, fp_budget: float, recall_floor: float, theta_max: float = 1.0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    op = operating_point(rows, fp_budget=fp_budget)  # picked on the full sweep
    rows = [r for r in rows if r.theta <= theta_max]  # zoom x to the live region
    theta = [r.theta for r in rows]

    fig, axL = plt.subplots(figsize=(7.2, 5.0))
    fig.patch.set_facecolor("white")
    axL.set_facecolor(BG)
    line_lat, = axL.plot(theta, _latency_masked(rows, recall_floor), "o-", color=OLIVE, lw=2, ms=5, label=f"{task} latency")
    axL.set_xlabel("Decision threshold")
    axL.set_ylabel(f"{task} median latency (ms)")
    axL.set_xlim(0.0, theta_max)
    axL.tick_params(axis="y", colors=OLIVE)
    axL.yaxis.label.set_color(OLIVE)

    axR = axL.twinx()
    line_rec, = axR.plot(theta, [r.recall for r in rows], "o-", color=STEEL, lw=2, ms=5, label="recall")
    line_fp, = axR.plot(theta, [r.fp_rate for r in rows], "s-", color=CRIMSON, lw=2, ms=5, label="fp_rate")
    axR.set_ylabel("recall / fp_rate")
    axR.set_ylim(0.0, 1.0)

    if op is not None:
        axL.axvline(op.theta, ls="--", lw=1.2, color="#888")
        axL.text(op.theta + 0.01, axL.get_ylim()[1] * 0.96, f"op θ={op.theta:.2f}", color="#555", fontsize=9, va="top")

    axL.legend(handles=[line_lat, line_rec, line_fp], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor="white")
    print(f"saved -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("probs", type=Path, help="a probs-{eot,int}.json file")
    ap.add_argument("--out", type=Path, required=True, help="output image path")
    ap.add_argument("--fp-budget", type=float, default=0.1, help="operating-point false-positive budget")
    ap.add_argument("--recall-floor", type=float, default=0.05, help="hide latency below this recall")
    ap.add_argument("--theta-max", type=float, default=1.0, help="zoom the x-axis to [0, theta-max] (drop the dead tail)")
    args = ap.parse_args()

    dataset = resolve_dataset(source=DEV_DATASET, skip_audio=True)  # gold, loaded once and reused
    probs = load_probs(args.probs)
    rows = sweep(probs, dataset)
    plot_single(rows, probs.task.upper(), args.out, fp_budget=args.fp_budget,
                recall_floor=args.recall_floor, theta_max=args.theta_max)


if __name__ == "__main__":
    main()
