#!/usr/bin/env python3
"""Threshold sweep + metric plots for the individual-channel baseline.

Sweeps the commit threshold tau in {0.1,..,1.0} over the cached per-channel
probabilities (dev set), scoring each operating point with the official
eval.score, and saves a multi-panel figure plus a printed table.

Each track (EOT / interruption) is swept independently: the track's tau_high is
varied (tau_low = 0.4*tau, refractory 2 s, the baseline's hysteresis) while the
other track stays at its committed default.

    python -m baselines.espnet_turntaking_perchannel.plot_threshold_sweep \
        --out baselines/espnet_turntaking_perchannel/threshold_sweep_dev.png
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

THRESHOLDS = [round(0.1 * i, 1) for i in range(1, 11)]  # 0.1 .. 1.0


def _metrics(score):
    tp, fp, fn = score.tp, score.fp, score.fn
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = score.recall
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and (prec + rec) else float("nan")
    lat = score.latency()
    return dict(recall=rec, fp_rate=score.fp_rate, precision=prec, f1=f1,
                p10=lat.p10, p50=lat.p50, p90=lat.p90, tp=tp, fp=fp, fn=fn)


def sweep(dataset, cache_dir, ids):
    convs = [conversation(dataset, i) for i in ids]
    for c in convs:
        P.channel_probs(c, cache_dir)  # ensure cached
    rows = {"eot": [], "interruption": []}
    for tau in THRESHOLDS:
        tl = round(tau * 0.4, 4)
        for track in ("eot", "interruption"):
            kw = dict(refr=P.REFRACTORY_S)
            if track == "eot":
                kw.update(eth=tau, etl=tl)
            else:
                kw.update(ith=tau, itl=tl)
            sub = Submission(schema_version=SCHEMA_VERSION,
                             predictions=[P.predict(c, cache_dir, **kw) for c in convs])
            agg = score_submission(sub, dataset)
            s = agg.task_eot if track == "eot" else agg.task_int
            rows[track].append((tau, _metrics(s)))
    return rows


def print_table(rows):
    for track in ("eot", "interruption"):
        print(f"\n== {track.upper()} (individual-channel, dev) ==")
        print(f"{'tau':>5} {'recall':>7} {'fp_rate':>8} {'prec':>6} {'f1':>6} "
              f"{'lat p10/50/90':>16} {'tp/fp/fn':>14}")
        for tau, m in rows[track]:
            print(f"{tau:>5.1f} {m['recall']:>7.3f} {m['fp_rate']:>8.3f} "
                  f"{m['precision']:>6.3f} {m['f1']:>6.3f} "
                  f"{m['p10']:>5.0f}/{m['p50']:>4.0f}/{m['p90']:>4.0f} "
                  f"{m['tp']}/{m['fp']}/{m['fn']:>4}")


def plot(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    taus = [t for t, _ in rows["eot"]]

    def col(track, key):
        return [m[key] for _, m in rows[track]]

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Individual-channel baseline — commit-threshold sweep (dev, official eval.score)",
                 fontsize=13, fontweight="bold")

    # (0,0) EOT metrics vs tau
    a = ax[0, 0]
    a.plot(taus, col("eot", "recall"), "o-", label="recall", color="C0")
    a.plot(taus, col("eot", "fp_rate"), "s-", label="fp_rate", color="C3")
    a.plot(taus, col("eot", "precision"), "^-", label="precision", color="C2")
    a.plot(taus, col("eot", "f1"), "d--", label="F1", color="C4")
    a.set_title("EOT: metrics vs threshold"); a.set_xlabel("commit threshold τ"); a.set_ylabel("score")
    a.set_ylim(0, 1); a.grid(alpha=0.3); a.legend(fontsize=8)

    # (0,1) INT metrics vs tau
    a = ax[0, 1]
    a.plot(taus, col("interruption", "recall"), "o-", label="recall", color="C0")
    a.plot(taus, col("interruption", "fp_rate"), "s-", label="fp_rate", color="C3")
    a.plot(taus, col("interruption", "precision"), "^-", label="precision", color="C2")
    a.plot(taus, col("interruption", "f1"), "d--", label="F1", color="C4")
    a.set_title("Interruption: metrics vs threshold"); a.set_xlabel("commit threshold τ"); a.set_ylabel("score")
    a.set_ylim(0, 1); a.grid(alpha=0.3); a.legend(fontsize=8)

    # (1,0) recall vs fp_rate operating curves
    a = ax[1, 0]
    for track, color in (("eot", "C0"), ("interruption", "C1")):
        fr, rc = col(track, "fp_rate"), col(track, "recall")
        a.plot(fr, rc, "o-", color=color, label=track.upper())
        for tau, x, y in zip(taus, fr, rc):
            a.annotate(f"{tau:.1f}", (x, y), fontsize=6, alpha=0.7)
    a.set_title("Operating curve: recall vs fp_rate"); a.set_xlabel("fp_rate"); a.set_ylabel("recall")
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.grid(alpha=0.3); a.legend(fontsize=8)

    # (1,1) latency p50 vs tau
    a = ax[1, 1]
    a.plot(taus, col("eot", "p50"), "o-", label="EOT p50", color="C0")
    a.plot(taus, col("interruption", "p50"), "o-", label="INT p50", color="C1")
    a.axhline(0, color="k", lw=0.6, alpha=0.5)
    a.set_title("Median latency vs threshold"); a.set_xlabel("commit threshold τ"); a.set_ylabel("latency p50 (ms)")
    a.grid(alpha=0.3); a.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    print(f"\nsaved plot -> {out_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEV_DATASET)
    ap.add_argument("--cache-dir", default=str(P._DEFAULT_CACHE))
    ap.add_argument("--out", default=str(_REPO / "baselines" / "espnet_turntaking_perchannel" / "threshold_sweep_dev.png"))
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    dataset = resolve_dataset(source=args.dataset)
    ids = conversation_ids(dataset)
    rows = sweep(dataset, Path(args.cache_dir), ids)
    print_table(rows)
    plot(rows, args.out)
    if args.csv:
        with open(args.csv, "w") as f:
            f.write("track,tau,recall,fp_rate,precision,f1,lat_p10,lat_p50,lat_p90,tp,fp,fn\n")
            for track in ("eot", "interruption"):
                for tau, m in rows[track]:
                    f.write(f"{track},{tau},{m['recall']:.4f},{m['fp_rate']:.4f},"
                            f"{m['precision']:.4f},{m['f1']:.4f},{m['p10']:.0f},{m['p50']:.0f},"
                            f"{m['p90']:.0f},{m['tp']},{m['fp']},{m['fn']}\n")
        print(f"saved csv -> {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
