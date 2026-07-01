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

# TurnBench site palette (site/src/app/globals.css): sage data hue, burgundy
# primary, bad-red, on the bone/card surface — so figures match the paper/site.
OLIVE = "#5b652a"    # latency (left axis) — sage
STEEL = "#43292e"    # recall (right axis) — burgundy (primary)
CRIMSON = "#b91c1c"  # fp_rate (right axis) — bad-red
BG = "#f6f5ef"       # card surface
INK = "#111111"      # foreground


def _latency_masked(rows: list[SweepRow], recall_floor: float) -> list[float]:
    # latency is measured over TPs only — meaningless where almost nothing fires.
    return [r.lat_p50 if r.recall >= recall_floor else float("nan") for r in rows]


def plot_single(rows: list[SweepRow], task: str, out: Path, *, fp_budget: float, recall_floor: float, theta_max: float = 1.0, criterion_arrows: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    op = operating_point(rows, fp_budget=fp_budget)  # picked on the full sweep
    rows = [r for r in rows if r.theta <= theta_max]  # zoom x to the live region
    theta = [r.theta for r in rows]

    fig, axL = plt.subplots(figsize=(7.2, 5.0))
    fig.patch.set_facecolor("#ecece8")  # bone surface
    axL.set_facecolor(BG)
    line_lat, = axL.plot(theta, _latency_masked(rows, recall_floor), "o-", color=OLIVE, lw=2, ms=5, label=f"{task} latency")
    axL.set_xlabel("Decision threshold")
    axL.set_ylabel(f"{task} median latency (ms)")
    axL.set_xlim(0.0, theta_max)
    axL.tick_params(axis="y", colors=OLIVE)
    axL.yaxis.label.set_color(OLIVE)

    axR = axL.twinx()
    line_rec, = axR.plot(theta, [r.recall for r in rows], "o-", color=STEEL, lw=2, ms=5, label="recall")
    line_fp, = axR.plot(theta, [r.fp_rate for r in rows], "s-", color=CRIMSON, lw=2, ms=5, label="FP rate")
    axR.set_ylabel("recall / FP rate")
    axR.set_ylim(0.0, 1.0)
    axR.axhline(fp_budget, ls=":", lw=1.2, color=CRIMSON, alpha=0.7)  # FP-rate budget
    axR.text(0.6, fp_budget - 0.015, f"FP budget = {fp_budget:g}", transform=axR.get_yaxis_transform(),
             ha="center", va="top", fontsize=8, color=CRIMSON)

    if op is not None:
        axL.axvline(op.theta, ls="--", lw=1.2, color="#888")
        axR.text(op.theta - 0.015, 0.6, f"op θ={op.theta:.2f}", color="#555", fontsize=9, va="center", ha="right")

    if criterion_arrows:  # low θ fires on weak evidence (eager); high θ requires strong evidence
        axL.text(0.0, -0.19, "← more eager", transform=axL.transAxes, ha="left", va="top",
                 fontsize=9, style="italic", color=INK)
        axL.text(1.0, -0.19, "more conservative →", transform=axL.transAxes, ha="right", va="top",
                 fontsize=9, style="italic", color=INK)

    axL.legend(handles=[line_lat, line_rec, line_fp], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="#ecece8", bbox_inches="tight")
    print(f"saved -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("probs", type=Path, help="a probs-{eot,int}.json file")
    ap.add_argument("--out", type=Path, required=True, help="output image path")
    ap.add_argument("--fp-budget", type=float, default=0.1, help="operating-point false-positive budget")
    ap.add_argument("--recall-floor", type=float, default=0.05, help="hide latency below this recall")
    ap.add_argument("--theta-max", type=float, default=1.0, help="zoom the x-axis to [0, theta-max] (drop the dead tail)")
    ap.add_argument("--criterion-arrows", action="store_true", help="annotate the θ axis with eager ↔ conservative")
    ap.add_argument("--step", type=float, default=0.05, help="threshold grid step for the sweep")
    args = ap.parse_args()

    dataset = resolve_dataset(source=DEV_DATASET, skip_audio=True)  # gold, loaded once and reused
    probs = load_probs(args.probs)
    thetas = [round(i * args.step, 4) for i in range(1, int(round(1.0 / args.step)))]
    rows = sweep(probs, dataset, thetas)
    plot_single(rows, probs.task.upper(), args.out, fp_budget=args.fp_budget,
                recall_floor=args.recall_floor, theta_max=args.theta_max,
                criterion_arrows=args.criterion_arrows)


if __name__ == "__main__":
    main()
