#!/usr/bin/env python3
"""Render the dev threshold-sweep figure(s) straight from probs JSON files.

Each probs file is swept and scored in memory (eval.sweep) against the dev gold —
no intermediate CSV. One file -> the per-model twin-axis (paper Fig. 1): EOT median
latency (left) and false-interruption rate (right) vs the decision threshold. Many
files -> all baselines on one graph: the latency vs false-interruption-rate
trade-off, one swept curve per model, each operating point (rule 2: lowest latency
at fp_rate <= budget) marked.

    # one baseline
    uv run --extra eval --extra plot python data_analysis/plot_sweep.py \
        baselines/<name>/probs-eot.json --out fig.png
    # every baseline on one graph
    uv run --extra eval --extra plot python data_analysis/plot_sweep.py \
        baselines/*/probs-eot.json --out sweep-all.png
"""
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `eval` imports

from eval.data import DEV_DATASET, resolve_dataset  # noqa: E402
from eval.sweep import SweepRow, load_probs, operating_point, sweep  # noqa: E402

OLIVE = "#6e7b3d"   # latency (single-model)
BLACK = "#111111"   # false-interruption rate (single-model)
BG = "#eef0e0"


def _latency_masked(rows: list[SweepRow], recall_floor: float) -> list[float]:
    # latency is unreliable where almost nothing fires — mask it there.
    return [r.lat_p50 if r.recall >= recall_floor else float("nan") for r in rows]


def plot_single(rows: list[SweepRow], task: str, out: Path, *, fp_budget: float, recall_floor: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    theta = [r.theta for r in rows]
    op = operating_point(rows, fp_budget=fp_budget)

    fig, axL = plt.subplots(figsize=(7.2, 5.0))
    fig.patch.set_facecolor("white")
    axL.set_facecolor(BG)
    line_lat, = axL.plot(theta, _latency_masked(rows, recall_floor), "o-", color=OLIVE, lw=2, ms=5, label=f"{task} latency")
    axL.set_xlabel("Decision threshold")
    axL.set_ylabel(f"{task} median latency (ms)")
    axL.set_xlim(0.0, 1.0)
    axL.tick_params(axis="y", colors=OLIVE)
    axL.yaxis.label.set_color(OLIVE)

    axR = axL.twinx()
    line_fir, = axR.plot(theta, [r.false_int_rate for r in rows], "s-", color=BLACK, lw=2, ms=5, label="False-interruption rate")
    axR.set_ylabel("False-interruption rate")
    axR.set_ylim(0.0, 1.0)
    axR.tick_params(axis="y", colors=BLACK)

    if op is not None:
        axL.axvline(op.theta, ls="--", lw=1.2, color="#888")
        axL.text(op.theta + 0.01, axL.get_ylim()[1] * 0.96, f"op θ={op.theta:.2f}", color="#555", fontsize=9, va="top")

    axL.legend(handles=[line_lat, line_fir], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor="white")
    print(f"saved -> {out}")


def plot_overlay(by_model: dict[str, list[SweepRow]], task: str, out: Path, *, fp_budget: float, recall_floor: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for i, (model, rows) in enumerate(by_model.items()):
        color = cmap(i % 10)
        ax.plot([r.false_int_rate for r in rows], _latency_masked(rows, recall_floor),
                "-", color=color, alpha=0.7, lw=1.8, label=model)
        op = operating_point(rows, fp_budget=fp_budget)
        if op is not None:
            ax.plot(op.false_int_rate, op.lat_p50, "o", color=color, ms=10,
                    markeredgecolor="black", markeredgewidth=0.8, zorder=5)
    ax.set_xlabel("False-interruption rate")
    ax.set_ylabel(f"{task} median latency (ms)")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(f"{task} latency vs false-interruption rate (● = operating point, fp ≤ {fp_budget})", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor="white")
    print(f"saved -> {out} ({len(by_model)} baselines)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("probs", type=Path, nargs="+", help="one or more probs-{eot,int}.json files")
    ap.add_argument("--out", type=Path, required=True, help="output image path")
    ap.add_argument("--fp-budget", type=float, default=0.1)
    ap.add_argument("--recall-floor", type=float, default=0.05)
    args = ap.parse_args()

    dataset = resolve_dataset(source=DEV_DATASET)  # gold, loaded once and reused
    by_model: dict[str, list[SweepRow]] = OrderedDict()
    tasks = set()
    for path in args.probs:
        probs = load_probs(path)
        tasks.add(probs.task)
        by_model[path.parent.name] = sweep(probs, dataset)
    if len(tasks) > 1:
        raise SystemExit(f"refusing to mix tasks on one graph: {sorted(tasks)} — plot eot and int separately")
    task = tasks.pop().upper()

    if len(by_model) == 1:
        plot_single(next(iter(by_model.values())), task, args.out, fp_budget=args.fp_budget, recall_floor=args.recall_floor)
    else:
        plot_overlay(by_model, task, args.out, fp_budget=args.fp_budget, recall_floor=args.recall_floor)


if __name__ == "__main__":
    main()
