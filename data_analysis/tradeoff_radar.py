#!/usr/bin/env python3
"""The turn-taking tradeoff triangle: Recall vs Precision (1-fp) vs Speed (1-latency).

Each committed baseline is scored on the whole split and placed on a 3-axis radar
per task (EOT, INT). All three axes point "outward = better", so a hypothetical
perfect model fills the triangle; the whole point of the figure is that none do —
the models that meet the false-positive budget are exactly the slow ones.

Latency has no natural [0,1] scale, so Speed = 1 - clamp(latency_p50, 0, L_MAX)/L_MAX
(a fire at/before the gold boundary scores 1; L_MAX defaults to 1300 ms, ~the slowest
model). Recall and Precision=1-fp_rate are already in [0,1].

    HF_TOKEN=<gold-token> uv run --extra eval --extra plot python \
        data_analysis/tradeoff_radar.py --dataset mundo-ai/turn-benchmark-test-golden --out radar.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from eval.data import DEV_DATASET, resolve_dataset  # noqa: E402
from eval.score import score_submission  # noqa: E402
from eval.submission import load_submission  # noqa: E402

BASELINES_DIR = Path(__file__).resolve().parent.parent / "baselines"
AXES = ["Recall", "Precision\n(1 − fp)", "Speed\n(1 − lat/1.3s)"]


def discover(split: str) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(BASELINES_DIR.glob(f"*/predictions-{split}*.json")):
        variant = path.stem[len(f"predictions-{split}"):].lstrip("-")
        found[path.parent.name + (f"/{variant}" if variant else "")] = path
    return found


def corner(recall: float, fp_rate: float, lat_ms: float, l_max: float) -> list[float]:
    speed = 1.0 - min(max(lat_ms, 0.0), l_max) / l_max
    return [recall, 1.0 - fp_rate, speed]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DEV_DATASET)
    ap.add_argument("--split", default=None, choices=["dev", "test"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--l-max", type=float, default=1300.0, help="latency (ms) mapped to Speed=0")
    ap.add_argument("--layout", choices=["overlay", "grid"], default="grid",
                    help="grid = one mini-triangle per model (clearest); overlay = all on two radars")
    args = ap.parse_args()
    split = args.split or ("test" if ("test" in args.dataset or "golden" in args.dataset) else "dev")

    dataset = resolve_dataset(source=args.dataset, skip_audio=True)
    baselines = discover(split)
    scored = {}
    for label, path in baselines.items():
        sc = score_submission(load_submission(path), dataset)
        scored[label] = {
            "EOT": (sc.task_eot.recall, sc.task_eot.fp_rate, sc.task_eot.latency().p50),
            "INT": (sc.task_int.recall, sc.task_int.fp_rate, sc.task_int.latency().p50),
        }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    angles = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False) + np.pi / 2
    loop = np.concatenate([angles, angles[:1]])
    EOT_C, INT_C = "#2b6cb0", "#b23a48"

    def style(ax, title, small):
        ax.set_xticks(angles)
        ax.set_xticklabels(AXES if not small else ["R", "P", "S"], fontsize=8 if small else 10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels([] if small else ["0.5", "1.0"], fontsize=8, color="#999")
        ax.set_title(title, fontsize=10 if small else 12, pad=10 if small else 18)
        ax.grid(alpha=0.3)

    if args.layout == "grid":
        names = sorted(scored, key=lambda n: scored[n]["EOT"][0], reverse=True)
        ncol = 5
        nrow = -(-len(names) // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.3 * nrow),
                                 subplot_kw={"polar": True})
        for ax, name in zip(axes.flat, names):
            for task, color in (("EOT", EOT_C), ("INT", INT_C)):
                v = corner(*scored[name][task], args.l_max)
                ax.plot(loop, v + v[:1], "-", lw=1.5, color=color, label=task)
                ax.fill(loop, v + v[:1], color=color, alpha=0.12)
            style(ax, name, small=True)
        for ax in axes.flat[len(names):]:
            ax.set_visible(False)
        axes.flat[0].legend(loc="upper left", bbox_to_anchor=(-0.35, 1.25), fontsize=8, frameon=False)
        fig.suptitle("Turn-taking tradeoff — no baseline fills the triangle "
                     "(R=Recall, P=Precision 1−fp, S=Speed 1−lat/1.3s; outward=better)",
                     fontsize=13, y=1.0)
    else:
        cmap = plt.get_cmap("tab10")
        fig, axaxes = plt.subplots(1, 2, figsize=(13, 6.5), subplot_kw={"polar": True})
        for ax, task in zip(axaxes, ("EOT", "INT")):
            for i, (label, tasks) in enumerate(scored.items()):
                v = corner(*tasks[task], args.l_max)
                ax.plot(loop, v + v[:1], "-o", lw=1.6, ms=3.5, color=cmap(i % 10), label=label, alpha=0.85)
            style(ax, f"{task}  (outward = better; ideal fills the triangle)", small=False)
        axaxes[1].legend(loc="center left", bbox_to_anchor=(1.15, 0.5), fontsize=9, frameon=False,
                         title=f"{args.dataset.split('/')[-1]} · {split}")
        fig.suptitle("Turn-taking tradeoff: no baseline fills the triangle", fontsize=14, y=1.02)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
