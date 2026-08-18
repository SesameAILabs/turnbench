#!/usr/bin/env python3
"""Model-comparison figures for the by-metadata breakdown (companion to
results_by_conversation_type.py / scores_by_metadata_findings.md).

Scores every committed baseline against the gold, pools into conversation_type
and gender-pairing groups (via results_by_conversation_type.compute), and renders:

    figures/int-fp-by-type.png       models x type heatmap of INT fp_rate
                                     (the §3 finding: fp concentrates in casual talk)
    figures/int-recall-by-gender.png FF/MM INT recall per model, sorted by the gap
                                     (the §4 finding: worse on female-female)

    # test (gold is private — set HF_TOKEN to the gold-repo token first)
    uv run --extra eval --extra plot python turnbench/analysis/plot_by_metadata.py \
        --dataset mundo-ai/turn-benchmark-test-golden
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

from turnbench.data import DEV_DATASET  # noqa: E402
from results_by_conversation_type import compute  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# TurnBench site palette — shared with plot_sweep.py.
CRIMSON = "#b91c1c"  # bad-red (false positives)
STEEL = "#43292e"    # burgundy (primary)
OLIVE = "#5b652a"    # sage
BONE = "#ecece8"
FIG_DIR = Path(__file__).resolve().parent / "figures"

SHORT_TYPE = {
    "Argumentative/Deliberative": "Argument",
    "Casual/Spontaneous": "Casual",
    "Collaborative/Problem-Solving": "Collab",
    "Instructional": "Instruct",
    "Narrative/Storytelling": "Narrative",
    "Task-Oriented/Transactional": "Task",
}


def _recall(ts) -> float:
    return ts.tp / (ts.tp + ts.fn) if ts.tp + ts.fn else float("nan")


def _fp(ts) -> float:
    return ts.fp / (ts.fp + ts.tn) if ts.fp + ts.tn else float("nan")


def plot_fp_by_type(scores, out: Path) -> None:
    """Heatmap: rows = baselines (sorted by mean INT fp), cols = conversation type,
    cell = INT fp_rate, redder = more false interruptions."""
    labels = sorted(scores.baselines, key=lambda l: np.nanmean(
        [_fp(scores.pooled[(l, "type", t, "INT")]) for t in scores.types]))
    M = np.array([[_fp(scores.pooled[(l, "type", t, "INT")]) for t in scores.types] for l in labels])

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    fig.patch.set_facecolor(BONE)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("bone_red", [BONE, "#e8b0b0", CRIMSON])
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0.0, vmax=0.6)
    ax.set_xticks(range(len(scores.types)))
    ax.set_xticklabels([SHORT_TYPE.get(t, t) for t in scores.types], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(scores.types)):
            v = M[i, j]
            ax.text(j, i, "—" if v != v else f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if v > 0.34 else "#333")
    ax.set_title("INT false-positive rate by conversation type\n(sorted by mean fp; redder = more false interruptions)",
                 fontsize=11, color=STEEL)
    fig.colorbar(im, ax=ax, label="INT fp_rate", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=BONE, bbox_inches="tight")
    print(f"saved -> {out}")


def plot_recall_by_gender(scores, out: Path) -> None:
    """Grouped bars: INT recall FF vs MM per baseline, sorted by (MM - FF) gap so the
    models that do worse on female-female pairs sink to the bottom."""
    labels = sorted(scores.baselines, key=lambda l: (
        _recall(scores.pooled[(l, "pairing", "MM", "INT")])
        - _recall(scores.pooled[(l, "pairing", "FF", "INT")])), reverse=True)
    ff = [_recall(scores.pooled[(l, "pairing", "FF", "INT")]) for l in labels]
    mm = [_recall(scores.pooled[(l, "pairing", "MM", "INT")]) for l in labels]

    y = np.arange(len(labels))
    h = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    fig.patch.set_facecolor(BONE)
    ax.barh(y + h / 2, ff, height=h, color=CRIMSON, label="female–female")
    ax.barh(y - h / 2, mm, height=h, color=STEEL, label="male–male")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("INT recall")
    ax.set_xlim(0, 1.0)
    ax.set_title("Interruption recall: female–female vs male–male\n(sorted by MM−FF gap; top = biggest FF deficit)",
                 fontsize=11, color=STEEL)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=BONE, bbox_inches="tight")
    print(f"saved -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DEV_DATASET, help="gold source (HF repo or local dir)")
    ap.add_argument("--split", default=None, choices=["dev", "test"])
    args = ap.parse_args()
    split = args.split or ("test" if ("test" in args.dataset or "golden" in args.dataset) else "dev")

    scores = compute(args.dataset, split)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_fp_by_type(scores, FIG_DIR / "int-fp-by-type.png")
    plot_recall_by_gender(scores, FIG_DIR / "int-recall-by-gender.png")


if __name__ == "__main__":
    main()
