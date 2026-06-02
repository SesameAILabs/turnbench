#!/usr/bin/env python3
"""Summarize consensus-construction stats, broken down by conversation type.

Reads:
    stats_out/consensus/_summary.json
    DATA_ROOT/BATCH/<task_id>/metadata.json

Writes (and prints):
    stats_out/consensus/_per_type.csv     plain TSV-ish table
    stats_out/consensus/_per_type.tex     LaTeX table
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


CANONICAL = ("Turn", "Interruption", "Backchannel", "Overlap", "Laughter", "NonContent")


def load_env(p: Path) -> dict[str, str]:
    env = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    env = load_env(repo / ".env")
    data_root = (Path(env["TT_BENCHMARK_DATA"]) if env.get("TT_BENCHMARK_DATA") else Path(env["DATA_ROOT"]) / env["BATCH"])
    stats_dir = Path(env.get("STATS_DIR", repo / "stats_out"))
    consensus_dir = stats_dir / "consensus"

    summary = json.loads((consensus_dir / "_summary.json").read_text())

    # task_id -> conversation_type
    ctype: dict[str, str] = {}
    for tid in summary:
        m = data_root / tid / "metadata.json"
        if m.exists():
            ctype[tid] = json.loads(m.read_text()).get("conversation_type", "unknown")
        else:
            ctype[tid] = "unknown"

    # consensus event counts per type per canonical label
    per_type_label: dict[str, Counter] = defaultdict(Counter)
    # aggregated drops + canonical-event totals (input) per type
    per_type_drops: dict[str, Counter] = defaultdict(Counter)
    per_type_input: dict[str, Counter] = defaultdict(int)  # sum across a/b/c
    per_type_consensus: dict[str, int] = defaultdict(int)
    per_type_excluded_intervals: dict[str, int] = defaultdict(int)
    per_type_samples: Counter = Counter()

    for tid, info in summary.items():
        t = ctype.get(tid, "unknown")
        per_type_samples[t] += 1
        counts = info["counts"]
        drops = info["drops"]
        per_type_consensus[t] += info["n_events"]
        per_type_excluded_intervals[t] += info["n_excluded_intervals"]
        # number of canonical events seen across all three annotators
        for ann in ("a", "b", "c"):
            for sp in (1, 2):
                per_type_input[t] += counts.get(f"sp{sp}_ann_{ann}_canonical", 0)
        for k, v in drops.items():
            # k looks like "sp1_label_mismatch", strip the speaker prefix
            reason = "_".join(k.split("_")[1:])
            per_type_drops[t][reason] += v
        # per-label counts: read each consensus jsonl
        for line in (consensus_dir / f"{tid}.jsonl").read_text().splitlines():
            if line.strip():
                per_type_label[t][json.loads(line)["label"]] += 1

    types = sorted(per_type_samples.keys())

    # Add aggregate "ALL" row
    types_with_all = types + ["ALL"]
    all_label = Counter()
    all_drops = Counter()
    all_input = 0
    all_consensus = 0
    all_excluded = 0
    all_samples = 0
    for t in types:
        all_label.update(per_type_label[t])
        all_drops.update(per_type_drops[t])
        all_input += per_type_input[t]
        all_consensus += per_type_consensus[t]
        all_excluded += per_type_excluded_intervals[t]
        all_samples += per_type_samples[t]
    per_type_label["ALL"] = all_label
    per_type_drops["ALL"] = all_drops
    per_type_input["ALL"] = all_input
    per_type_consensus["ALL"] = all_consensus
    per_type_excluded_intervals["ALL"] = all_excluded
    per_type_samples["ALL"] = all_samples

    # ---- TSV-ish table ----
    rows = []
    header = ["type", "n_samples", "input_events_3x", "consensus", "kept_pct",
              "excluded_intervals"] + list(CANONICAL) + [
              "drop_no_b_match", "drop_no_c_match",
              "drop_label_mismatch", "drop_endpoint_spread"]
    header[2] = "avg_annot_events"
    rows.append(header)
    for t in types_with_all:
        inp = per_type_input[t]
        kept = per_type_consensus[t]
        # avg per-annotator event count = inp / 3; kept_pct is the average
        # share of an annotator's events that reaches consensus
        pct = round(100 * kept / max(inp / 3, 1), 2)
        row = [t, per_type_samples[t], inp // 3, kept, pct,
               per_type_excluded_intervals[t]]
        for lbl in CANONICAL:
            row.append(per_type_label[t].get(lbl, 0))
        for reason in ("no_b_match", "no_c_match", "label_mismatch", "endpoint_spread"):
            row.append(per_type_drops[t].get(reason, 0))
        rows.append(row)

    out_csv = consensus_dir / "_per_type.csv"
    with out_csv.open("w", newline="") as f:
        csv.writer(f).writerows(rows)

    # Pretty print
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    for ri, row in enumerate(rows):
        print(" | ".join(f"{str(c):>{w}}" for c, w in zip(row, widths)))
        if ri == 0:
            print("-+-".join("-" * w for w in widths))

    # ---- LaTeX table ----
    short = {
        "Argumentative/Deliberative": "Arg/Delib",
        "Casual/Spontaneous": "Casual",
        "Collaborative/Problem-Solving": "Collab",
        "Instructional": "Instruct",
        "Task-Oriented/Transactional": "Task",
        "Narrative/Storytelling": "Narrative",
        "ALL": "\\textbf{All}",
    }
    tex = []
    tex.append("% Auto-generated by eval/consensus_stats.py")
    tex.append("\\begin{table}[t]")
    tex.append("\\centering")
    tex.append("\\small")
    tex.append("\\setlength{\\tabcolsep}{4pt}")
    tex.append("\\begin{tabular}{l|r|rr|rrrrrr}")
    tex.append("\\toprule")
    tex.append("Conversation type & $N$ & Consensus & Kept (\\%) & "
               "Turn & Interrupt. & Backchnl. & Overlap & Laughter & NonCont. \\\\")
    tex.append("\\midrule")
    for t in types_with_all:
        name = short.get(t, t)
        inp = per_type_input[t]
        kept = per_type_consensus[t]
        pct = 100 * kept / max(inp / 3, 1)
        cols = [name, per_type_samples[t], kept, f"{pct:.1f}"]
        for lbl in CANONICAL:
            cols.append(per_type_label[t].get(lbl, 0))
        if t == "ALL":
            tex.append("\\midrule")
        tex.append(" & ".join(str(c) for c in cols) + " \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\caption{Consensus events per conversation type. "
               "Consensus events are those where all three annotators agree on the "
               "canonical label and start/end times (within $\\pm200$\\,ms after the "
               "label-map collapse). $N$ is the number of dialogues; Kept (\\%) is "
               "consensus events as a fraction of the average per-annotator event count.}")
    tex.append("\\label{tab:consensus-by-type}")
    tex.append("\\end{table}")

    (consensus_dir / "_per_type.tex").write_text("\n".join(tex) + "\n")

    print()
    print("LaTeX written to:", consensus_dir / "_per_type.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
