#!/usr/bin/env python3
"""Dual-objective threshold-sweep figure (EOT latency vs false-interruption rate)
for the individual-channel baseline, in the TurnBench Fig-1 style.

A single decision threshold theta is swept over the cached per-channel
probabilities (dev set); for each theta we score with the official eval.score and
read two opposing objectives:
  * EOT median latency (ms)        -- left axis  (lower = more responsive)
  * false-interruption rate        -- right axis (interruption fp_rate over the
                                       scorer's negative spans; lower = fewer
                                       spurious barge-ins)
They move in opposite directions as theta rises -> a Pareto trade-off.

No GPU / model run (reads the cache).

    python -m baselines.espnet_turntaking_perchannel.plot_pareto_dev \
        --out baselines/espnet_turntaking_perchannel/pareto_sweep_dev.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from baselines.espnet_turntaking_perchannel import predict as P  # noqa: E402
from eval.data import DEV_DATASET, conversation, conversation_ids, resolve_dataset  # noqa: E402
from eval.score import score_submission  # noqa: E402
from eval.submission import SCHEMA_VERSION, Submission  # noqa: E402

# committed per-track operating thresholds (the baseline tunes them separately)
EOT_OP = 0.20
INT_OP = 0.15


RECALL_FLOOR = 0.05   # below this, EOT median latency is over too few TPs to trust


def sweep(dataset, cache_dir, ids, thetas):
    convs = [conversation(dataset, i) for i in ids]
    for c in convs:
        P.channel_probs(c, cache_dir)
    eot_lat_p50, int_fpr, eot_recall = [], [], []
    for th in thetas:
        tl = round(th * 0.4, 4)
        sub = Submission(schema_version=SCHEMA_VERSION, predictions=[
            P.predict(c, cache_dir, eth=th, etl=tl, ith=th, itl=tl, refr=P.REFRACTORY_S)
            for c in convs])
        agg = score_submission(sub, dataset)
        eot_lat_p50.append(agg.task_eot.latency().p50)
        int_fpr.append(agg.task_int.fp_rate)
        eot_recall.append(agg.task_eot.recall)
    return eot_lat_p50, int_fpr, eot_recall


def plot(thetas, eot_lat, int_fpr, eot_recall, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    BG = "#eef0e0"            # light olive/cream
    OLIVE = "#6e7b3d"         # EOT latency
    BLACK = "#111111"         # false-interruption rate

    # mask EOT latency where recall is too low for a stable median
    eot_lat_m = [lat if r >= RECALL_FLOOR else float("nan")
                 for lat, r in zip(eot_lat, eot_recall)]
    last_valid = max((t for t, r in zip(thetas, eot_recall) if r >= RECALL_FLOOR),
                     default=thetas[-1])

    fig, axL = plt.subplots(figsize=(7.4, 5.2))
    fig.patch.set_facecolor("white")
    axL.set_facecolor(BG)

    lA, = axL.plot(thetas, eot_lat_m, "o-", color=OLIVE, lw=2, ms=6,
                   label="EOT latency")
    axL.set_xlabel("Decision threshold")
    axL.set_ylabel("EOT median latency (ms)")
    axL.set_xlim(0.0, 1.0)
    axL.tick_params(axis="y", colors=OLIVE)
    axL.yaxis.label.set_color(OLIVE)

    axR = axL.twinx()
    lB, = axR.plot(thetas, int_fpr, "s-", color=BLACK, lw=2, ms=6,
                   label="False-interruption rate")
    axR.set_ylabel("False-interruption rate")
    axR.set_ylim(0.0, 1.0)
    axR.tick_params(axis="y", colors=BLACK)

    # operating thresholds (per-track; the baseline tunes EOT and INT separately)
    axL.axvline(EOT_OP, ls="--", lw=1.2, color=OLIVE, alpha=0.8)
    axL.axvline(INT_OP, ls="--", lw=1.2, color=BLACK, alpha=0.5)
    ymax = axL.get_ylim()[1]
    axL.text(EOT_OP + 0.01, ymax * 0.96, "EOT op 0.20", color=OLIVE, fontsize=9,
             va="top")
    axL.text(INT_OP - 0.01, ymax * 0.96, "INT op 0.15", color=BLACK, fontsize=9,
             va="top", ha="right")

    # shade the region where EOT recall is too low to report a latency
    if last_valid < thetas[-1]:
        axL.axvspan(last_valid, thetas[-1], color="#cccccc", alpha=0.25, lw=0)
        axL.text((last_valid + thetas[-1]) / 2, ymax * 0.55,
                 "EOT recall < 5%\n(latency undefined)", color="#555555",
                 fontsize=8, ha="center", va="center")

    axL.legend(handles=[lA, lB], loc="upper center", ncol=2,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.10))
    for ax in (axL, axR):
        for s in ax.spines.values():
            s.set_color("#888888")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor="white")
    print(f"saved -> {out_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEV_DATASET)
    ap.add_argument("--cache-dir", default=str(P._DEFAULT_CACHE))
    ap.add_argument("--out", default=str(_REPO / "baselines" / "espnet_turntaking_perchannel" / "pareto_sweep_dev.png"))
    ap.add_argument("--grid", default="0.05:0.95:0.05", help="lo:hi:step")
    args = ap.parse_args()
    lo, hi, step = (float(x) for x in args.grid.split(":"))
    thetas = [round(lo + i * step, 3) for i in range(int(round((hi - lo) / step)) + 1)]

    dataset = resolve_dataset(source=args.dataset)
    ids = conversation_ids(dataset)
    eot_lat, int_fpr, eot_recall = sweep(dataset, Path(args.cache_dir), ids, thetas)

    print(f"{'theta':>6} {'EOT_lat_p50_ms':>15} {'false_int_rate':>15} {'EOT_recall':>11}")
    for th, lat, fpr, r in zip(thetas, eot_lat, int_fpr, eot_recall):
        print(f"{th:>6.2f} {lat:>15.0f} {fpr:>15.3f} {r:>11.3f}")
    plot(thetas, eot_lat, int_fpr, eot_recall, args.out)


if __name__ == "__main__":
    main()
