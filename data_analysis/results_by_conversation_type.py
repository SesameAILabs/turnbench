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
    uv run --extra eval python data_analysis/results_by_conversation_type.py

    # test (gold is private — set HF_TOKEN to the gold-repo token first)
    uv run --extra eval python data_analysis/results_by_conversation_type.py \
        --dataset mundo-ai/turn-benchmark-test-golden

    # the paper's Table IV (tab:by-type): the complete LaTeX table* environment,
    # regenerated from the committed predictions — paste the output verbatim.
    # Run on latest main; never hand-edit the numbers.
    uv run --extra eval python data_analysis/results_by_conversation_type.py \
        --dataset mundo-ai/turn-benchmark-test-golden --latex

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

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from rich import box  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from eval.data import (  # noqa: E402
    DEV_DATASET,
    GOLD_DATASET,
    conversation,
    conversation_ids,
    read_columns_projected,
    resolve_dataset,
)
from eval.gold import events_for_conversation  # noqa: E402
from eval.score import TaskScore, merge, score_conversation  # noqa: E402
from eval.submission import load_submission  # noqa: E402

PAIRING = {("female", "female"): "FF", ("male", "male"): "MM"}
BASELINES_DIR = Path(__file__).resolve().parent.parent / "baselines"
LEADERBOARD_JSON = Path(__file__).resolve().parent.parent / "results" / "leaderboard-test.json"

# Paper display names for --latex, in table order; None = \midrule group break.
# Baselines missing from this list are appended (unmapped) at the end so new
# models are never silently dropped.
PAPER_ROWS: list[tuple[str, str] | None] = [
    ("rms_vad", "RMS VAD"),
    None,
    ("openai_server_vad", "OpenAI Realtime (Server VAD)"),
    ("openai_semantic_vad", "OpenAI Realtime (Semantic VAD)"),
    ("kyutai_semantic_vad", "Kyutai SVAD"),
    ("smart_turn_v3", "SmartTurn v3"),
    None,
    ("vap", "VAP"),
    ("mimi_endpointer", "Mimi-EP"),
    ("espnet_turntaking", "ESPnet TT-pred.\\"),
    ("espnet_turntaking_perchannel", "ESPnet TT-pred.\\ (per-ch.)"),
    None,
    ("wavlm_base_causal", "WavLM-Base (causal)"),
    ("wavlm_large_causal", "WavLM-Large (causal)"),
    ("wavlm_large_anchor", "WavLM-Large (anchor)"),
    None,
    ("gemini_vad", "Gemini 3.1 Live"),
]


def metadata_repo(source: str) -> str:
    """A source whose parquet carries the `metadata` column without the gold token:
    the private gold repo publishes none, so read metadata from the public test
    parquet (same conversation ids, metadata intact)."""
    return "mundo-ai/turn-benchmark-test" if source == GOLD_DATASET else source


def load_metadata(source: str) -> dict[str, dict[str, str]]:
    """{conversation_id: {"type": ..., "pairing": FF|MM|mixed}} from the parquet.
    Read column-projected over HTTP range requests — never snapshotting the shards,
    whose audio would dominate the download and memory."""
    repo = metadata_repo(source)
    if Path(repo).is_dir():
        table = pa.concat_tables([
            pq.ParquetFile(shard).read(columns=["conversation_id", "metadata"])
            for shard in sorted(Path(repo).glob("*.parquet"))
        ])
    else:
        table = read_columns_projected(repo, None, ["conversation_id", "metadata"])
    out: dict[str, dict[str, str]] = {}
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


def latex_table(ms: MetadataScores) -> str:
    """The paper's complete per-conversation-type table (tab:by-type): the full
    table* environment, one row per baseline in PAPER_ROWS order — per-type
    `recall/fpr` cells for EOT and INT (leading zeros stripped, 1.00 shown as
    1.0), then the overall median-latency Δt columns read from
    results/leaderboard-test.json so the two committed artifacts cannot disagree."""
    import json

    leaderboard = {m["model"]: m for m in json.loads(LEADERBOARD_JSON.read_text())["models"]}
    p50 = {
        label: (m["eot"]["latency_ms"]["p50"], m["int"]["latency_ms"]["p50"])
        for label, m in leaderboard.items()
    }
    # A track a model does not support (e.g. an EOT-only baseline) is null in
    # the leaderboard; render its cells as em-dashes rather than 0-recall.
    supported = {
        label: {task for task in ("EOT", "INT") if m[task.lower()]["recall"] is not None}
        for label, m in leaderboard.items()
    }

    def num(x: float) -> str:
        s = f"{x:.2f}"
        return "1.0" if s == "1.00" else s.lstrip("0")

    def cell(ts: TaskScore) -> str:
        return f"{num(ts.tp / (ts.tp + ts.fn))}/{num(ts.fp / (ts.fp + ts.tn))}"

    def lat(x: float | None) -> str:
        if x is None:
            return "---"
        v = round(x)
        return f"$-${abs(v)}" if v < 0 else str(v)

    mapped = [r[0] for r in PAPER_ROWS if r]
    rows: list[tuple[str, str] | None] = [
        r for r in PAPER_ROWS if r is None or r[0] in ms.baselines
    ] + [(label, label) for label in ms.baselines if label not in mapped]

    width = max(len(name) for row in rows if row for _, name in [row])
    lines = []
    for row in rows:
        if row is None:
            lines.append("\\midrule")
            continue
        label, name = row
        cells = [
            cell(ms.pooled[(label, "type", t, task)]) if task in supported[label] else "---"
            for t in ms.types
            for task in ("EOT", "INT")
        ]
        le, li = p50[label]
        lines.append(f"{name:<{width}} & " + " & ".join(cells) + f" & {lat(le)} & {lat(li)} \\\\")

    n = len(ms.types)
    type_heads = " & ".join(
        f"\\multicolumn{{2}}{{c{'|' if i < n - 1 else '|'}}}{{\\ctype{{{t.split('/')[0]}}}}}"
        for i, t in enumerate(ms.types)
    )
    cmidrules = "".join(f"\\cmidrule(lr){{{2 * i + 2}-{2 * i + 3}}}" for i in range(n + 1))
    return "\n".join([
        "\\begin{table*}[!tbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\caption{Per-conversation-type results, per baseline (test set). Per-type sub-columns "
        "EOT and INT (Interruption) report recall\\,/\\,fpr (leading zeros omitted); the "
        "\\textsc{Overall} column reports median latency (ms), "
        "$\\Delta t = t_\\text{pred}-t_\\text{gold}$, for each track. fpr is the false-positive rate; "
        "--- marks a track the baseline does not support.}",
        "\\label{tab:by-type}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "|cc" * (n + 1) + "@{}}",
        "\\toprule",
        f"& {type_heads} & \\multicolumn{{2}}{{c}}{{Overall $\\Delta t$}} \\\\",
        cmidrules,
        "Baseline & " + " & ".join(["EOT & INT"] * (n + 1)) + " \\\\",
        "\\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabular*}",
        "\\end{table*}",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DEV_DATASET, help="gold source (HF repo or local dir)")
    ap.add_argument("--split", default=None, choices=["dev", "test"],
                    help="which predictions-<split>.json to score (default: inferred from --dataset)")
    ap.add_argument("--latex", action="store_true",
                    help="print the paper's tab:by-type LaTeX rows instead of the rich tables")
    args = ap.parse_args()
    split = args.split or ("test" if ("test" in args.dataset or "golden" in args.dataset) else "dev")

    scores = compute(args.dataset, split)
    if args.latex:
        print(latex_table(scores))
        return
    ids, types, baselines, density, pooled, n_type = scores
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
