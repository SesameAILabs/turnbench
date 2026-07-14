#!/usr/bin/env python3
"""Overall model scores split by audio status (clean vs noisy).

Scores every `baselines/*/predictions-<split>.json` against the real gold —
built on the fly by `eval/gold.py` from the dataset's annotation tracks (the
private golden repo for test; set HF_TOKEN) — and pools TP/FN/FP/TN and
latencies per baseline within {clean, noisy, all} using
`audio_status_metadata.csv` (keyed by prompt_id, which matches
conversation_id).

Committed predictions already sit at each model's tuned operating point
(`results/leaderboard-test.json`), so no thresholds are picked here.

`--type` restricts to one conversation_type — this controls for the
audio_status × conversation_type confound (noisy recordings over-index on
argumentative/collaborative talk, and INT false positives concentrate in
casual talk). The Instructional slice (10 clean / 9 noisy) is the
near-balanced type used in the paper's noise-robustness paragraph; every
number quoted there must come from this output — never hand-edit them.

    # test (gold is private — set HF_TOKEN to the gold-repo token first)
    uv run --extra eval python data_analysis/results_by_audio_status.py

    # the paper's slice
    uv run --extra eval python data_analysis/results_by_audio_status.py --type Instructional
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rich import box  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from data_analysis.results_by_conversation_type import load_metadata  # noqa: E402
from eval.data import (  # noqa: E402
    DEV_DATASET,
    GOLD_DATASET,
    conversation,
    conversation_ids,
    resolve_dataset,
)
from eval.score import TaskScore, merge, score_conversation  # noqa: E402
from eval.submission import load_submission  # noqa: E402

BASELINES_DIR = REPO / "baselines"
AUDIO_STATUS_CSV = REPO / "audio_status_metadata.csv"
SPLIT_SOURCE = {"test": GOLD_DATASET, "dev": DEV_DATASET}


def load_audio_status() -> dict[str, str]:
    """{conversation_id (str): 'clean' | 'noisy'} from audio_status_metadata.csv."""
    out: dict[str, str] = {}
    with AUDIO_STATUS_CSV.open() as fh:
        for row in csv.DictReader(fh):
            out[str(row["prompt_id"]).strip()] = row["audio_status"].strip()
    return out


def discover(split: str) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(BASELINES_DIR.glob(f"*/predictions-{split}*.json")):
        variant = path.stem[len(f"predictions-{split}"):].lstrip("-")
        label = path.parent.name + (f"/{variant}" if variant else "")
        found[label] = path
    return found


def fmt(ts: TaskScore) -> str:
    if ts.tp + ts.fn == 0:
        return "[dim]—[/]"
    recall = ts.tp / (ts.tp + ts.fn)
    if ts.fp + ts.tn:
        fp = ts.fp / (ts.fp + ts.tn)
        fp_cell = f"[red]{fp:.3f}[/]" if fp > 0.10 else f"[green]{fp:.3f}[/]"
    else:
        fp_cell = "[dim]—[/]"
    lat = statistics.median(ts.latencies_ms) if ts.latencies_ms else float("nan")
    lat_cell = "—" if lat != lat else f"{lat:.0f}"
    return f"{recall:.3f}/{fp_cell}/[dim]{lat_cell}[/]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--type", default=None,
                    help="restrict to this conversation_type (e.g. 'Instructional'). "
                         "Removes the audio_status × conversation_type confound.")
    args = ap.parse_args()

    source = SPLIT_SOURCE[args.split]
    status = load_audio_status()
    meta = load_metadata(source)
    dataset = resolve_dataset(source=source, skip_audio=True)
    durations = json.loads(
        (REPO / "eval" / f"durations-{args.split}.json").read_text()
    )["durations"]
    baselines = discover(args.split)
    if not baselines:
        raise SystemExit(f"no baselines/*/predictions-{args.split}*.json found")

    ids = sorted(
        (c for c in conversation_ids(dataset)
         if c in status and c in durations
         and (args.type is None or meta[c]["type"].startswith(args.type))),
        key=lambda s: int(s),
    )
    if not ids:
        raise SystemExit(f"no conversations match --type={args.type!r}")
    n_clean = sum(1 for c in ids if status[c] == "clean")
    n_noisy = sum(1 for c in ids if status[c] == "noisy")
    hours_by_group = {
        "clean": sum(durations[c] for c in ids if status[c] == "clean") / 3600.0,
        "noisy": sum(durations[c] for c in ids if status[c] == "noisy") / 3600.0,
        "all":   sum(durations[c] for c in ids) / 3600.0,
    }

    pooled: dict[tuple[str, str, str], TaskScore] = defaultdict(TaskScore)
    for label, path in baselines.items():
        by_conv = load_submission(path).by_conversation()
        for cid in ids:
            if cid not in by_conv:
                continue
            sc = score_conversation(by_conv[cid], conversation(dataset, cid))
            for group in (status[cid], "all"):
                merge(pooled[(label, group, "EOT")], sc.task_eot)
                merge(pooled[(label, group, "INT")], sc.task_int)

    console = Console()
    console.rule(
        f"[bold]{source}[/]  ·  {args.split}  ·  "
        + (f"type={args.type!r}  " if args.type else "")
        + f"clean={n_clean}  noisy={n_noisy}  all={len(ids)}  ·  {len(baselines)} baselines"
    )

    for task in ("EOT", "INT"):
        table = Table(
            title=f"{task} overall by audio_status",
            box=box.SIMPLE_HEAVY,
            title_style="bold",
            header_style="bold cyan",
            caption="recall / [green]fp[/] / [dim]lat_ms(p50)[/]   ·   "
                    "[red]red fp[/] = over 0.10 budget",
            caption_style="dim",
        )
        table.add_column("baseline", style="bold")
        for h in ("clean", "noisy", "all"):
            table.add_column(h, justify="right")
        for label in baselines:
            table.add_row(
                label,
                fmt(pooled[(label, "clean", task)]),
                fmt(pooled[(label, "noisy", task)]),
                fmt(pooled[(label, "all", task)]),
            )
        console.print(table)

    # False-positive volume, decoupled from negative-span density: raw FP count
    # and FP per hour of audio, per slice. This is the metric to trust when the
    # slice's negative-span distribution shifts (e.g. fewer backchannels on
    # noisy conversations shrinks the fp_rate denominator).
    for task in ("EOT", "INT"):
        fp_table = Table(
            title=f"{task} false positives — count and per-hour rate",
            box=box.SIMPLE_HEAVY, title_style="bold", header_style="bold cyan",
            caption=f"audio hours: clean={hours_by_group['clean']:.1f}  "
                    f"noisy={hours_by_group['noisy']:.1f}  all={hours_by_group['all']:.1f}",
            caption_style="dim",
        )
        fp_table.add_column("baseline", style="bold")
        for h in ("clean fp", "clean /hr", "noisy fp", "noisy /hr", "Δ /hr"):
            fp_table.add_column(h, justify="right")
        for label in baselines:
            clean_fp = pooled[(label, "clean", task)].fp
            noisy_fp = pooled[(label, "noisy", task)].fp
            clean_rate = clean_fp / hours_by_group["clean"]
            noisy_rate = noisy_fp / hours_by_group["noisy"]
            delta = noisy_rate - clean_rate
            color = "red" if delta > 0 else "green"
            fp_table.add_row(
                label,
                str(clean_fp), f"{clean_rate:.1f}",
                str(noisy_fp), f"{noisy_rate:.1f}",
                f"[{color}]{delta:+.1f}[/]",
            )
        console.print(fp_table)

    # Latency-tail comparison: for each baseline and task, print p10 / p50 / p90
    # on the clean vs noisy TP latency distribution. Noise pushing more evidence
    # into a model before it commits shows up here — usually at p90.
    def pct(ts, p):
        return statistics.quantiles(ts.latencies_ms, n=100)[p - 1] if len(ts.latencies_ms) >= 100 else (
            float("nan") if not ts.latencies_ms
            else sorted(ts.latencies_ms)[int(round((p / 100.0) * (len(ts.latencies_ms) - 1)))]
        )
    for task in ("EOT", "INT"):
        lat_table = Table(
            title=f"{task} latency (ms) percentiles — clean vs noisy",
            box=box.SIMPLE_HEAVY, title_style="bold", header_style="bold cyan",
            caption="p10/p50/p90 of TP latencies; Δp90 flags if noisy tail is worse",
            caption_style="dim",
        )
        lat_table.add_column("baseline", style="bold")
        for h in ("clean p10", "p50", "p90", "noisy p10", "p50", "p90", "Δ p90"):
            lat_table.add_column(h, justify="right")
        for label in baselines:
            cl = pooled[(label, "clean", task)]
            no = pooled[(label, "noisy", task)]
            if not cl.latencies_ms or not no.latencies_ms:
                lat_table.add_row(label, *(["[dim]—[/]"] * 7))
                continue
            cp10, cp50, cp90 = pct(cl, 10), pct(cl, 50), pct(cl, 90)
            np10, np50, np90 = pct(no, 10), pct(no, 50), pct(no, 90)
            d90 = np90 - cp90
            color = "red" if d90 > 0 else "green"
            lat_table.add_row(
                label,
                f"{cp10:.0f}", f"{cp50:.0f}", f"{cp90:.0f}",
                f"{np10:.0f}", f"{np50:.0f}", f"{np90:.0f}",
                f"[{color}]{d90:+.0f}[/]",
            )
        console.print(lat_table)


if __name__ == "__main__":
    main()
