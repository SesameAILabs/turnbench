#!/usr/bin/env python3
"""Overall model scores split by audio status (clean vs noisy) — offline.

Rebuilds the scorer's gold event sets from the committed consensus artifacts
under `stats_out/` — no HF dataset download needed — and replays every
`baselines/*/predictions-test.json` against them. Pools TP/FN/FP/TN and
latencies per baseline within {clean, noisy, all} using `audio_status_metadata.csv`
(keyed by prompt_id, which matches conversation_id).

Committed predictions already sit at each model's tuned operating point
(`results/leaderboard-test.json`), so no thresholds are picked here.

Approximation: the true turn-view consensus (Turn segments AND the turn-view
excluded intervals) is not committed — only the label-view is. This scorer
reconstructs the turn view from `stats_out/turn_regions/*.jsonl` (label-view
Turn segments) and reuses the label-view excluded intervals in the turn-view
slot as the closest offline stand-in. As a result the overall (clean+noisy)
numbers drift a few points below `results/leaderboard-test.json`. The
*clean/noisy delta* is still informative — the same approximation error applies
to both slices — but the absolute recall/fpr should not be quoted against the
leaderboard.

    uv run --extra eval python data_analysis/results_by_audio_status.py
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

from eval.gold import (  # noqa: E402
    ConsensusEvent,
    ConsensusViews,
    Interval,
    build_conversation_events,
)
from eval.score import TaskScore, merge, score_task  # noqa: E402
from eval.submission import load_submission, validate_event_times  # noqa: E402

BASELINES_DIR = REPO / "baselines"
STATS_DIR = REPO / "stats_out"
AUDIO_STATUS_CSV = REPO / "audio_status_metadata.csv"
DURATIONS_TEST = REPO / "eval" / "durations-test.json"


def load_audio_status() -> dict[str, str]:
    """{conversation_id (str): 'clean' | 'noisy'} from audio_status_metadata.csv."""
    out: dict[str, str] = {}
    with AUDIO_STATUS_CSV.open() as fh:
        for row in csv.DictReader(fh):
            out[str(row["prompt_id"]).strip()] = row["audio_status"].strip()
    return out


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def load_conversation_gold(cid: str):
    """Rebuild ConversationEvents for one conversation from stats_out artifacts.

    `stats_out/consensus/{cid}.jsonl`  — label-view consensus events
    `stats_out/consensus/{cid}_excluded.jsonl` — label-view excluded intervals
    `stats_out/turn_regions/{cid}.jsonl` — turn-view Turn segments, grouped by region

    The turn-view excluded intervals aren't committed separately; we approximate
    them with the label-view excluded intervals (see module docstring)."""
    label_events = [
        ConsensusEvent(
            speaker=row["speaker"],
            start=row["start"],
            end=row["end"],
            label=row["label"],
        )
        for row in read_jsonl(STATS_DIR / "consensus" / f"{cid}.jsonl")
    ]
    label_excluded = [
        Interval(speaker=row["speaker"], start=row["start"], end=row["end"])
        for row in read_jsonl(STATS_DIR / "consensus" / f"{cid}_excluded.jsonl")
    ]
    turn_events: list[ConsensusEvent] = []
    for region in read_jsonl(STATS_DIR / "turn_regions" / f"{cid}.jsonl"):
        speaker = region["speaker"]
        for ev in region["events"]:
            if ev["label"] == "Turn":
                turn_events.append(
                    ConsensusEvent(speaker=speaker, start=ev["start"], end=ev["end"], label="Turn")
                )
    views = ConsensusViews(
        turn_events=turn_events,
        turn_excluded=label_excluded,   # approx: see module docstring
        events=label_events,
        excluded=label_excluded,
    )
    return build_conversation_events(views)


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


def load_conversation_types() -> dict[str, str]:
    """{conversation_id: conversation_type} from stats_out/per_conversation.csv."""
    out: dict[str, str] = {}
    with (STATS_DIR / "per_conversation.csv").open() as fh:
        for row in csv.DictReader(fh):
            out[row["task_id"]] = row["conversation_type"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--type", default=None,
                    help="restrict to this conversation_type (e.g. 'Instructional'). "
                         "Removes the audio_status × conversation_type confound.")
    args = ap.parse_args()

    status = load_audio_status()
    types = load_conversation_types() if args.type else {}
    durations = json.loads(DURATIONS_TEST.read_text())["durations"]
    baselines = discover(args.split)
    if not baselines:
        raise SystemExit(f"no baselines/*/predictions-{args.split}*.json found")

    # scored conversations = intersection of (has status entry) ∩ (has stats_out gold)
    ids = sorted(
        {c for c in status if (STATS_DIR / "consensus" / f"{c}.jsonl").exists()
         and c in durations
         and (args.type is None or types.get(c) == args.type)},
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

    # Build gold once (cache per conversation)
    gold_by_cid = {cid: load_conversation_gold(cid) for cid in ids}

    pooled: dict[tuple[str, str, str], TaskScore] = defaultdict(TaskScore)
    for label, path in baselines.items():
        submission = load_submission(path)
        by_conv = submission.by_conversation()
        for cid in ids:
            if cid not in by_conv:
                continue
            pred = by_conv[cid]
            validate_event_times(pred, durations[cid])
            gold = gold_by_cid[cid]
            eot = score_task(
                gold.eot_positive_events,
                gold.eot_negative_spans,
                {1: pred.speaker_1.eot, 2: pred.speaker_2.eot},
                gold.eot_excluded,
            )
            intv = score_task(
                gold.int_positive_events,
                gold.int_negative_spans,
                {1: pred.speaker_1.interruption, 2: pred.speaker_2.interruption},
                gold.int_excluded,
            )
            for group in (status[cid], "all"):
                merge(pooled[(label, group, "EOT")], eot)
                merge(pooled[(label, group, "INT")], intv)

    console = Console()
    console.rule(
        f"[bold]offline scorer (stats_out gold)[/]  ·  {args.split}  ·  "
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
