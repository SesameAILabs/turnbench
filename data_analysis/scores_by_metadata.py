#!/usr/bin/env python3
"""Model scores broken down by conversation metadata.

Complements `per_conversation.py` (which characterizes the *dataset* — event/
backchannel/interruption rates, IAA — per conversation type). This one scores the
committed *baselines* (`baselines/*/predictions-<split>.json`) per conversation
against the gold, then pools TP/FN/FP/TN + latencies within metadata groups —
conversation_type and speaker-gender pairing — and prints group-level recall /
fp_rate / median-latency. It also prints the baseline-independent gold event density
per type. Pooling counts within a group (not averaging per-conversation rates) is
the statistically correct aggregation.

Metadata (conversation_type, per-speaker gender) rides in the dataset's `metadata`
column; it is present even in the public test parquet (labels stripped), so the
metadata axis needs no gold token — only scoring does.

    # dev (public, no token)
    uv run --extra eval python data_analysis/scores_by_metadata.py

    # test (gold is private — set HF_TOKEN to the gold-repo token first)
    uv run --extra eval python data_analysis/scores_by_metadata.py \
        --dataset mundo-ai/turn-benchmark-test-golden

The three-axis regime (acoustic high-recall/high-fp, semantic low/low, learned
balanced) holds within every type; the interesting movement is in *where* each
model's false positives concentrate — see the accompanying findings note.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import pyarrow.parquet as pq  # noqa: E402
from rich import box  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from eval.data import (  # noqa: E402
    DEV_DATASET,
    GOLD_DATASET,
    conversation,
    conversation_ids,
    resolve_dataset,
)
from eval.gold import events_for_conversation  # noqa: E402
from eval.score import TaskScore, merge, score_conversation  # noqa: E402
from eval.submission import load_submission  # noqa: E402

PAIRING = {("female", "female"): "FF", ("male", "male"): "MM"}
BASELINES_DIR = Path(__file__).resolve().parent.parent / "baselines"


def metadata_repo(source: str) -> str:
    """A source whose parquet carries the `metadata` column without the gold token:
    the private gold repo publishes none, so read metadata from the public test
    parquet (same conversation ids, metadata intact)."""
    return "mundo-ai/turn-benchmark-test" if source == GOLD_DATASET else source


def load_metadata(source: str) -> dict[str, dict[str, str]]:
    """{conversation_id: {"type": ..., "pairing": FF|MM|mixed}} from the parquet."""
    from baselines.openai_realtime import _shard_files

    out: dict[str, dict[str, str]] = {}
    for shard in _shard_files(metadata_repo(source), None):
        table = pq.ParquetFile(shard).read(columns=["conversation_id", "metadata"])
        for cid, m in zip(table["conversation_id"].to_pylist(), table["metadata"].to_pylist()):
            genders = (m["speaker_1_actor_gender"], m["speaker_2_actor_gender"])
            out[cid] = {
                "type": m["conversation_type"],
                "pairing": PAIRING.get(tuple(sorted(genders)), "mixed"),
            }
    return out


def discover(split: str) -> dict[str, Path]:
    """{label: predictions_path} for every baseline with a committed file for `split`,
    including `-base`/`-large`-style variants (the label carries the suffix)."""
    found: dict[str, Path] = {}
    for path in sorted(BASELINES_DIR.glob(f"*/predictions-{split}*.json")):
        variant = path.stem[len(f"predictions-{split}"):].lstrip("-")
        label = path.parent.name + (f"/{variant}" if variant else "")
        found[label] = path
    return found


def fmt(ts: TaskScore, *, with_lat: bool = True) -> str:
    """Rich-markup cell `recall / fp [/ lat_ms]`; fp over the 0.10 budget shows red."""
    if ts.tp + ts.fn == 0:
        return "[dim]—[/]"
    recall = ts.tp / (ts.tp + ts.fn)
    if ts.fp + ts.tn:
        fp = ts.fp / (ts.fp + ts.tn)
        fp_str = f"[red]{fp:.2f}[/]" if fp > 0.10 else f"[green]{fp:.2f}[/]"
    else:
        fp_str = "[dim]—[/]"
    out = f"{recall:.2f}/{fp_str}"
    if with_lat:
        lat = statistics.median(ts.latencies_ms) if ts.latencies_ms else float("nan")
        out += f"/[dim]{'—' if lat != lat else f'{lat:.0f}'}[/]"
    return out


class MetadataScores(NamedTuple):
    """Everything the table-printer and the plotter both need.

    ids: scored conversation ids (present in both meta and the dataset).
    types: sorted conversation_type labels.
    baselines: {label: predictions_path}.
    density: {type: [EOT+, EOT-, INT+, INT-]} gold-event counts (baseline-independent).
    pooled: {(label, axis, key, task): TaskScore} — axis in {"type","pairing"},
        key a type name or FF/MM/mixed, task in {"EOT","INT"}; TP/FN/FP/TN pooled within group.
    n_type: {type: n_conversations}.
    """
    ids: list[str]
    types: list[str]
    baselines: dict[str, Path]
    density: dict[str, list[int]]
    pooled: dict[tuple, TaskScore]
    n_type: dict[str, int]


def compute(dataset_source: str, split: str) -> MetadataScores:
    """Score every committed baseline for `split` against the gold at `dataset_source`,
    pooling TP/FN/FP/TN + latencies into conversation_type and gender-pairing groups.
    Returns a MetadataScores; raises SystemExit if no predictions files are found."""
    meta = load_metadata(dataset_source)
    dataset = resolve_dataset(source=dataset_source, skip_audio=True)
    ids = [c for c in conversation_ids(dataset) if c in meta]
    types = sorted({meta[c]["type"] for c in ids})
    baselines = discover(split)
    if not baselines:
        raise SystemExit(f"no baselines/*/predictions-{split}*.json found")

    # gold event density per type (baseline-independent)
    density: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for cid in ids:
        ev = events_for_conversation(conversation(dataset, cid))
        d = density[meta[cid]["type"]]
        d[0] += len(ev.eot_positive_events); d[1] += len(ev.eot_negative_spans)
        d[2] += len(ev.int_positive_events); d[3] += len(ev.int_negative_spans)

    # per-baseline, per-conversation scoring pooled into metadata groups
    pooled: dict[tuple, TaskScore] = defaultdict(TaskScore)
    for label, path in baselines.items():
        by_conv = load_submission(path).by_conversation()
        for cid in ids:
            if cid not in by_conv:
                continue
            sc = score_conversation(by_conv[cid], conversation(dataset, cid))
            for axis, key in (("type", meta[cid]["type"]), ("pairing", meta[cid]["pairing"])):
                merge(pooled[(label, axis, key, "EOT")], sc.task_eot)
                merge(pooled[(label, axis, key, "INT")], sc.task_int)

    n_type = {t: sum(meta[c]["type"] == t for c in ids) for t in types}
    return MetadataScores(ids, types, baselines, density, pooled, n_type)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DEV_DATASET, help="gold source (HF repo or local dir)")
    ap.add_argument("--split", default=None, choices=["dev", "test"],
                    help="which predictions-<split>.json to score (default: inferred from --dataset)")
    args = ap.parse_args()
    split = args.split or ("test" if ("test" in args.dataset or "golden" in args.dataset) else "dev")

    ids, types, baselines, density, pooled, n_type = compute(args.dataset, split)
    console = Console()
    console.rule(
        f"[bold]{args.dataset}[/]  ·  {split}  ·  {len(ids)} conversations  ·  {len(baselines)} baselines"
    )

    density_table = Table(
        title="gold event density by conversation type", box=box.SIMPLE_HEAVY,
        title_style="bold", header_style="bold cyan",
    )
    density_table.add_column("conversation_type", style="cyan")
    for col in ("n", "EOT+", "EOT-", "INT+", "INT-", "INT+/conv"):
        density_table.add_column(col, justify="right")
    for t in types:
        d = density[t]
        density_table.add_row(t, str(n_type[t]), str(d[0]), str(d[1]), str(d[2]), str(d[3]),
                              f"[bold]{d[2] / n_type[t]:.1f}[/]")
    console.print(density_table)

    def block(axis: str, keys: list[str], headers: list[str], *, with_lat: bool) -> None:
        metrics = "recall / [green]fp[/] / [dim]lat_ms[/]" if with_lat else "recall / [green]fp[/]"
        for task in ("EOT", "INT"):
            table = Table(
                title=f"{task} by {axis}", box=box.SIMPLE_HEAVY, title_style="bold",
                header_style="bold cyan",
                caption=f"{metrics}   ·   [red]red fp[/] = over 0.10 budget",
                caption_style="dim",
            )
            table.add_column("baseline", style="bold")
            for h in headers:
                table.add_column(h, justify="right")
            for label in baselines:
                table.add_row(label, *(fmt(pooled[(label, axis, k, task)], with_lat=with_lat) for k in keys))
            console.print(table)

    # by-type: 6 columns → drop latency for width (it's ~model-constant across types);
    # by-pairing: 3 columns → room for full recall/fp/latency.
    block("type", types, [t.split("/")[0][:8] for t in types], with_lat=False)  # full names in density table
    block("pairing", ["FF", "MM", "mixed"], ["FF", "MM", "mixed"], with_lat=True)


if __name__ == "__main__":
    main()
