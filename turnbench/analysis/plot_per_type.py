#!/usr/bin/env python3
"""Generate per-conversation-type figures from stats_out/per_type_aggregate.json.

Outputs (under stats_out/figures/):
    heatmap_zscore.png        types x metrics, z-scored across types
    metric_bars.png           small multiples: one bar chart per metric, bar per type
    metric_bars.pdf           same as PDF for paper use
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLOT_METRICS = [
    "duration_min",
    "event_rate_per_min",
    "word_rate_wpm",
    "turn_dur_mean",
    "n_speaker_changes",
    "fto_median_s",
    "speaker_balance",
    "bc_rate_per_min",
    "int_rate_per_min",
    "int_competitive_per_min",
    "int_cooperative_per_min",
    "overlap_dur_s",
    "laughter_rate_per_min",
    "non_content_ratio",
    "silence_ratio",
    "question_rate_per_min",
    "iaa_fleiss_kappa",
    "boundary_f1_ab",
]

SHORT_NAMES = {
    "Argumentative/Deliberative": "Arg/Delib",
    "Casual/Spontaneous": "Casual",
    "Collaborative/Problem-Solving": "Collab",
    "Instructional": "Instruct",
    "Task-Oriented/Transactional": "Task",
    "Narrative/Storytelling": "Narrative",
}


def load_env(p: Path) -> dict:
    env = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    env = load_env(repo / ".env")
    stats_dir = Path(env.get("STATS_DIR", repo / "stats_out"))
    fig_dir = stats_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    agg = json.loads((stats_dir / "per_type_aggregate.json").read_text())
    types = [t for t in agg if t != "all"]
    short = [SHORT_NAMES.get(t, t) for t in types]

    # Matrix of mean values: rows=metrics, cols=types
    M = np.array([[agg[t][m]["mean"] for t in types] for m in PLOT_METRICS], dtype=float)
    stds = np.array([[agg[t][m]["std"] for t in types] for m in PLOT_METRICS], dtype=float)
    ns = np.array([agg[t]["n_conversations"] for t in types])

    # ---- z-score heatmap ----
    row_mu = M.mean(axis=1, keepdims=True)
    row_sd = M.std(axis=1, keepdims=True) + 1e-9
    Z = (M - row_mu) / row_sd

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(short, rotation=30, ha="right")
    ax.set_yticks(range(len(PLOT_METRICS)))
    ax.set_yticklabels(PLOT_METRICS)
    for i in range(len(PLOT_METRICS)):
        for j in range(len(types)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(Z[i, j]) < 1.2 else "white")
    ax.set_title("Per-conversation-type means (cells = mean; color = z-score across types)")
    fig.colorbar(im, ax=ax, label="z-score")
    fig.tight_layout()
    fig.savefig(fig_dir / "heatmap_zscore.png", dpi=160)
    fig.savefig(fig_dir / "heatmap_zscore.pdf")
    plt.close(fig)

    # ---- small-multiples bar chart with std error bars ----
    ncols = 4
    nrows = int(np.ceil(len(PLOT_METRICS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 2.6))
    axes = axes.flatten()
    colors = plt.get_cmap("tab10")(range(len(types)))
    se = stds / np.sqrt(ns[None, :])  # standard error

    for k, m in enumerate(PLOT_METRICS):
        ax = axes[k]
        ax.bar(range(len(types)), M[k], yerr=se[k], capsize=3, color=colors)
        ax.set_title(m, fontsize=10)
        ax.set_xticks(range(len(types)))
        ax.set_xticklabels(short, rotation=35, ha="right", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for k in range(len(PLOT_METRICS), len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"Per-conversation-type means (n_total={int(ns.sum())})", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "metric_bars.png", dpi=160)
    fig.savefig(fig_dir / "metric_bars.pdf")
    plt.close(fig)

    print(f"Wrote figures to {fig_dir}", file=sys.stderr)
    for p in sorted(fig_dir.iterdir()):
        print(f"  {p.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
