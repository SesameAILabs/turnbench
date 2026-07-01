#!/usr/bin/env python3
"""Cross-model leaderboard + tradeoff figure — the paper's headline comparison.

Scores every committed baseline (`baselines/*/predictions-<split>.json`) on one
split and prints a leaderboard: per model, EOT and INT recall / fp_rate / latency,
sorted by EOT recall, with fp over the 0.1 operating-point budget shown in red.
With --figure it also writes the tradeoff scatter (latency vs recall, fp encoded
as the in/out-of-budget constraint, in-budget Pareto frontier drawn) — the figure
that shows no baseline is simultaneously in-budget, high-recall, and low-latency.

    HF_TOKEN=<gold-token> uv run --extra eval --extra plot python \
        data_analysis/leaderboard.py --dataset mundo-ai/turn-benchmark-test-golden --figure tradeoff.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich import box  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from eval.data import DEV_DATASET, resolve_dataset  # noqa: E402
from eval.score import score_submission  # noqa: E402
from eval.submission import load_submission  # noqa: E402

BASELINES_DIR = Path(__file__).resolve().parent.parent / "baselines"
BUDGET = 0.1
IN_C, OVER_C = "#2b7a3d", "#b23a48"


def discover(split: str) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(BASELINES_DIR.glob(f"*/predictions-{split}*.json")):
        variant = path.stem[len(f"predictions-{split}"):].lstrip("-")
        found[path.parent.name + (f"/{variant}" if variant else "")] = path
    return found


def score_all(dataset, split: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for label, path in discover(split).items():
        sc = score_submission(load_submission(path), dataset)
        out[label] = {
            "EOT": (sc.task_eot.recall, sc.task_eot.fp_rate, sc.task_eot.latency()),
            "INT": (sc.task_int.recall, sc.task_int.fp_rate, sc.task_int.latency()),
        }
    return out


def _fp(fp: float) -> str:
    if math.isnan(fp):
        return "[dim]—[/]"
    return (f"[red]{fp:.3f}[/]" if fp > BUDGET else f"[green]{fp:.3f}[/]")


def print_table(scored: dict, tag: str) -> None:
    table = Table(title=f"leaderboard — {tag}", box=box.SIMPLE_HEAVY, title_style="bold",
                  header_style="bold cyan",
                  caption=f"recall / fp_rate / latency-p50-ms · [red]red fp[/] over the {BUDGET:g} budget · sorted by EOT recall")
    table.add_column("#", justify="right")
    table.add_column("model", style="bold")
    for t in ("EOT recall", "EOT fp", "EOT lat", "INT recall", "INT fp", "INT lat"):
        table.add_column(t, justify="right")
    for i, (label, r) in enumerate(sorted(scored.items(), key=lambda kv: kv[1]["EOT"][0], reverse=True), 1):
        (er, ef, el), (ir, iff, il) = r["EOT"], r["INT"]
        table.add_row(str(i), label, f"{er:.3f}", _fp(ef), f"{el.p50:.0f}", f"{ir:.3f}", _fp(iff), f"{il.p50:.0f}")
    Console().print(table)


def pareto(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    front, best = [], -1.0
    for lat, rec in sorted(points):
        if rec > best:
            front.append((lat, rec)); best = rec
    return front


def short(label: str) -> str:
    return (label.replace("causal_wavlm_predictor", "cwlm").replace("espnet_turntaking_perchannel", "espnet_pc")
            .replace("espnet_turntaking", "espnet").replace("openai_", "").replace("_vad", "").replace("wavlm_", "wl_"))


def figure(scored: dict, out: Path, tag: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, task in zip(axes, ("EOT", "INT")):
        in_budget = []
        for label, r in scored.items():
            recall, fp, lat_obj = r[task]
            lat = lat_obj.p50
            ok = fp <= BUDGET
            ax.scatter(lat, recall, s=70, zorder=3, facecolor=IN_C if ok else "none",
                       edgecolor=IN_C if ok else OVER_C, linewidths=1.6)
            ax.annotate(short(label), (lat, recall), textcoords="offset points", xytext=(6, 4), fontsize=8, color="#333")
            if ok:
                in_budget.append((lat, recall))
        front = pareto(in_budget)
        if len(front) > 1:
            fx, fy = zip(*front)
            ax.plot(fx, fy, "--", color=IN_C, lw=1.3, alpha=0.7, zorder=2)
        ax.set_xlabel("latency p50 (ms)   →  slower"); ax.set_ylabel("recall")
        ax.set_ylim(0, 1); ax.set_title(task, fontsize=12); ax.grid(alpha=0.3)
    axes[0].legend(handles=[
        Line2D([], [], marker="o", ls="none", mfc=IN_C, mec=IN_C, ms=9, label=f"in budget (fp ≤ {BUDGET:g})"),
        Line2D([], [], marker="o", ls="none", mfc="none", mec=OVER_C, mew=1.6, ms=9, label="over budget"),
        Line2D([], [], ls="--", color=IN_C, label="in-budget Pareto frontier"),
    ], loc="lower right", fontsize=9, frameon=False)
    fig.suptitle(f"No baseline is in-budget, high-recall, and fast at once  ({tag})", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved figure -> {out}")


def _num(x: float) -> float | None:
    """JSON has no NaN; empty tasks (no positives / no negatives) become null."""
    return None if x is None or math.isnan(x) else round(x, 6)


def write_json(scored: dict, out: Path, dataset: str, split: str) -> None:
    """Emit the same rows the table shows as a committable artifact the website
    renders. Sorted by EOT recall (NaN last), so the file order is the ranking."""
    models = []
    for label, r in sorted(
        scored.items(),
        key=lambda kv: (kv[1]["EOT"][0] if not math.isnan(kv[1]["EOT"][0]) else -1.0),
        reverse=True,
    ):
        def task(recall: float, fp: float, lat) -> dict:
            return {
                "recall": _num(recall),
                "fp_rate": _num(fp),
                "latency_ms": {"p10": _num(lat.p10), "p50": _num(lat.p50), "p90": _num(lat.p90)},
            }

        (er, ef, el), (ir, iff, il) = r["EOT"], r["INT"]
        models.append({"model": label, "eot": task(er, ef, el), "int": task(ir, iff, il)})
    payload = {"split": split, "dataset": dataset, "fp_budget": BUDGET, "models": models}
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote leaderboard -> {out} ({len(models)} models)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DEV_DATASET)
    ap.add_argument("--split", default=None, choices=["dev", "test"])
    ap.add_argument("--figure", type=Path, default=None, help="also write the tradeoff scatter here")
    ap.add_argument("--json", type=Path, default=None, dest="json_out", help="also write the leaderboard as JSON here")
    args = ap.parse_args()
    split = args.split or ("test" if ("test" in args.dataset or "golden" in args.dataset) else "dev")

    scored = score_all(resolve_dataset(source=args.dataset, skip_audio=True), split)
    tag = f"{args.dataset.split('/')[-1]} · {split}"
    print_table(scored, tag)
    if args.json_out:
        write_json(scored, args.json_out, args.dataset, split)
    if args.figure:
        figure(scored, args.figure, tag)


if __name__ == "__main__":
    main()
