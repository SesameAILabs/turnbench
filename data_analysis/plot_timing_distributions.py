#!/usr/bin/env python3
"""Compare TurnBench and Switchboard timing distributions (FTO / gap / pause).

Inputs are the two distribution exports (each stores fto + pause; gap = the
positive FTOs, derived here):
  stats_out/timing_distributions.json                       (timing_distributions.py)
  data_analysis/swbd/results/swbd_timing_distributions.json (swbd/per_conversation_swbd.py)

Outputs:
  stats_out/figures/timing_distributions.{png,pdf}   3-panel figure: FTO density
                                                     (linear), gap and pause
                                                     densities (log time)
  stats_out/figures/timing_distributions_compact.{png,pdf}
                                                     single-column two-panel
                                                     (gap | pause), pooled
                                                     TurnBench vs Switchboard
  stats_out/timing_distributions_summary.json        quantiles + KS statistics
plus the same summary printed as a table.

The pause comparison carries one asymmetry: the Switchboard pauses have a floor
of --ipu-gap (0.2 s, shorter silences are merged into IPUs), while TurnBench
pauses are annotator-drawn. The summary therefore also reports pause KS
conditioned on >= 0.2 s, which compares like with like.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
MEASURES = ("fto", "gap", "pause")
QUANTS = (10, 25, 50, 75, 90)
PAUSE_FLOOR_S = 0.2
KDE_CHUNK = 8192


def measure(dists: dict, m: str) -> np.ndarray:
    """One measure's samples from a group's stored distributions.

    dists: {"fto": [...], "pause": [...]} as exported; m: "fto", "gap", or
    "pause". Gap is derived as the positive FTOs. Returns a float array.
    """
    x = np.array(dists["fto" if m == "gap" else m], dtype=float)
    return x[x > 0] if m == "gap" else x


def kde(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian KDE of samples `x` evaluated on `grid` (Silverman bandwidth).

    x: 1-D samples; grid: 1-D evaluation points. Returns densities on grid.
    The sample axis is processed in chunks to keep the grid x samples
    temporaries small (the Switchboard pause panel is ~255k samples).
    """
    x = np.asarray(x, dtype=float)
    sd = x.std()
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(sd, iqr / 1.34) if iqr > 0 else sd
    bw = 0.9 * sigma * len(x) ** (-1 / 5)
    if bw <= 0:
        bw = 0.05
    dens = np.zeros(len(grid))
    for i in range(0, len(x), KDE_CHUNK):
        z = (grid[:, None] - x[None, i:i + KDE_CHUNK]) / bw
        dens += np.exp(-0.5 * z * z).sum(axis=1)
    return dens / (len(x) * bw * np.sqrt(2 * np.pi))


def curve(vals: np.ndarray, grid: np.ndarray, logx: bool) -> np.ndarray:
    """Density curve for one group's samples on a panel's grid.

    vals: samples in seconds; grid: evaluation points (log10 seconds when logx).
    Returns densities on grid.
    """
    if logx:
        vals = np.log10(vals[vals > 0])
    return kde(vals, grid)


def ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic D = max |ECDF_a - ECDF_b|."""
    pts = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), pts, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), pts, side="right") / len(b)
    return float(np.abs(ca - cb).max())


def quantiles(x: np.ndarray) -> dict:
    q = np.percentile(x, QUANTS)
    return {f"p{p}": round(float(v), 3) for p, v in zip(QUANTS, q)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--turnbench", type=Path,
                    default=REPO / "stats_out/timing_distributions.json")
    ap.add_argument("--swbd", type=Path,
                    default=REPO / "data_analysis/swbd/results/swbd_timing_distributions.json")
    ap.add_argument("--out-dir", type=Path, default=REPO / "stats_out")
    args = ap.parse_args()

    tb = json.loads(args.turnbench.read_text())["groups"]
    sw_payload = json.loads(args.swbd.read_text())
    sw_label = sw_payload["corpus"]
    sw = sw_payload["groups"][sw_label]
    types = sorted(k for k in tb if k != "all")

    # ---- summary: quantiles per group, KS of TurnBench-pooled vs SWBD ------
    summary: dict = {"groups": {}, "ks_turnbench_all_vs_swbd": {}}
    for name, dists in [("TurnBench (all)", tb["all"]),
                        *[(t, tb[t]) for t in types],
                        (sw_label, sw)]:
        g: dict = {}
        for m in MEASURES:
            x = measure(dists, m)
            g[m] = {"n": len(x), **quantiles(x)}
            if m == "fto":
                g[m]["neg_share"] = round(float((x < 0).mean()), 3)
        summary["groups"][name] = g
    for m in MEASURES:
        summary["ks_turnbench_all_vs_swbd"][m] = round(
            ks_stat(measure(tb["all"], m), measure(sw, m)), 3)
    pa = measure(tb["all"], "pause")
    pb = measure(sw, "pause")
    summary["ks_turnbench_all_vs_swbd"]["pause_ge_floor"] = round(
        ks_stat(pa[pa >= PAUSE_FLOOR_S], pb[pb >= PAUSE_FLOOR_S]), 3)
    summary["pause_floor_s"] = PAUSE_FLOOR_S

    # Within-corpus yardstick for the cross-corpus KS values: the pairwise KS
    # between TurnBench's own conversation types. A cross-corpus D inside this
    # range means Switchboard differs from TurnBench no more than TurnBench's
    # registers differ from each other.
    summary["ks_turnbench_type_pairs"] = {
        m: {f"{a.split('/')[0]} vs {b.split('/')[0]}": round(
                ks_stat(measure(tb[a], m), measure(tb[b], m)), 3)
            for a, b in itertools.combinations(types, 2)}
        for m in MEASURES
    }

    out_json = args.out_dir / "timing_distributions_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))

    # ---- printed table ------------------------------------------------------
    hdr = ["group", "measure", "n"] + [f"p{p}" for p in QUANTS] + ["neg%"]
    print("  ".join(f"{h:>28}" if h == "group" else f"{h:>7}" for h in hdr))
    for name, g in summary["groups"].items():
        for m in MEASURES:
            row = [f"{name:>28}", f"{m:>7}", f"{g[m]['n']:>7}"]
            row += [f"{g[m][f'p{p}']:>7}" for p in QUANTS]
            row += [f"{g[m].get('neg_share', ''):>7}"]
            print("  ".join(row))
    print(f"\nKS (TurnBench all vs {sw_label}): " + ", ".join(
        f"{m}={v}" for m, v in summary["ks_turnbench_all_vs_swbd"].items()))

    # ---- figure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.7))
    panels = [
        ("fto", "Floor-transfer offset (s)", np.linspace(-2.5, 2.5, 400), False),
        ("gap", "Gap duration (s)", np.linspace(np.log10(0.02), np.log10(30), 400), True),
        ("pause", "Pause duration (s)", np.linspace(np.log10(0.05), np.log10(30), 400), True),
    ]
    for ax, (m, xlabel, grid, logx) in zip(axes, panels):
        for t in types:
            ax.plot(grid, curve(measure(tb[t], m), grid, logx), lw=0.8, alpha=0.45,
                    label=t.split("/")[0] if m == "fto" else None)
        ax.plot(grid, curve(measure(tb["all"], m), grid, logx), lw=2.0, color="black",
                label="TurnBench (all)" if m == "fto" else None)
        ax.plot(grid, curve(measure(sw, m), grid, logx), lw=2.0, ls="--", color="crimson",
                label=sw_label if m == "fto" else None)
        if m == "fto":
            ax.axvline(0.0, color="gray", lw=0.6, alpha=0.6)
        if logx:
            ticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30]
            ax.set_xticks([np.log10(t) for t in ticks])
            ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.set_xlabel(xlabel)
        ax.set_yticks([])
    axes[0].set_ylabel("Density")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()

    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"timing_distributions.{ext}", dpi=200,
                    bbox_inches="tight")

    # ---- compact single-column variant (gap | pause, pooled only) -----------
    cfig, caxes = plt.subplots(1, 2, figsize=(3.5, 1.3))
    for ax, (m, xlabel, grid, logx) in zip(caxes, panels[1:]):
        ax.plot(grid, curve(measure(tb["all"], m), grid, logx), lw=1.4,
                color="black", label="TurnBench")
        ax.plot(grid, curve(measure(sw, m), grid, logx), lw=1.4, ls="--",
                color="crimson", label="Switchboard")
        ticks = [0.05, 0.2, 1, 5, 30]
        ax.set_xticks([np.log10(t) for t in ticks])
        ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=6.5)
        ax.set_xlabel(xlabel, fontsize=7, labelpad=1.5)
        ax.set_yticks([])
    caxes[0].set_ylabel("Density", fontsize=7)
    caxes[0].legend(fontsize=6, frameon=False, borderaxespad=0.2,
                    handlelength=1.6)
    cfig.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        cfig.savefig(fig_dir / f"timing_distributions_compact.{ext}", dpi=200,
                     bbox_inches="tight")
    print(f"\nWrote {fig_dir}/timing_distributions[_compact].png/.pdf and {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
