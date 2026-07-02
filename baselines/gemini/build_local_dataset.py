#!/usr/bin/env python3
"""Build local parquet shards matching the mundo-ai/turn-benchmark-dev schema
from the SRTs + WAVs under a Mundo delivery root (--data). Lets eval.score
run without HF access:

    python -m eval.score baselines/gemini/gemini_pred.json \
        --dataset baselines/gemini/.local_dev_parquet

Columns produced (per eval/data.py):
  conversation_id : string
  speaker_{1,2}_audio : binary (raw WAV bytes)
  speaker_{1,2}_annotation_{a,b,c} : list<struct<start_s: f64, end_s: f64, label: str>>
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SPEAKERS = (1, 2)
ANNOTATORS = ("a", "b", "c")

_SRT_TIME = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")
_LABEL = re.compile(r"\[([^\]]+)\]")


def _t(stamp: str) -> float:
    h, m, s, ms = _SRT_TIME.match(stamp).groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    events: list[dict] = []
    for block in text.strip().split("\n\n"):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        time_line = lines[1]
        if " --> " not in time_line:
            continue
        a, b = time_line.split(" --> ")
        start_s = _t(a.strip())
        end_s = _t(b.strip())
        body = " ".join(lines[2:]) if len(lines) > 2 else ""
        m = _LABEL.search(body)
        label = m.group(1) if m else ""
        events.append({"start_s": start_s, "end_s": end_s, "label": label})
    return events


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True,
                    help="Mundo delivery root (numbered folders with "
                         "speaker audio + metadata)")
    ap.add_argument("--split", type=Path,
                    default=Path(__file__).resolve().parents[2] / "eval" / "splits" / "dev.txt")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / ".local_dev_parquet")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    ids = [ln.strip() for ln in args.split.read_text().splitlines()
           if ln.strip() and not ln.startswith("#")]
    ids = sorted([t for t in ids if (args.data / t).is_dir()], key=int)
    print(f"[parquet] {len(ids)} conversations from {args.split.name}")

    rows: list[dict] = []
    for tid in ids:
        row: dict = {"conversation_id": tid}
        for k in SPEAKERS:
            row[f"speaker_{k}_audio"] = (args.data / tid / f"speaker_{k}_audio.wav").read_bytes()
            for ann in ANNOTATORS:
                row[f"speaker_{k}_annotation_{ann}"] = parse_srt(
                    args.data / tid / f"speaker_{k}_annotation_{ann}.srt")
        rows.append(row)

    annotation_type = pa.list_(pa.struct([
        ("start_s", pa.float64()),
        ("end_s", pa.float64()),
        ("label", pa.string()),
    ]))
    schema = pa.schema(
        [("conversation_id", pa.string())]
        + [(f"speaker_{k}_audio", pa.binary()) for k in SPEAKERS]
        + [(f"speaker_{k}_annotation_{ann}", annotation_type)
           for k in SPEAKERS for ann in ANNOTATORS]
    )

    table = pa.Table.from_pylist(rows, schema=schema)
    shard_path = args.out / "dev-00000-of-00001.parquet"
    pq.write_table(table, shard_path)
    print(f"[parquet] wrote {shard_path} "
          f"({table.num_rows} rows, {shard_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
