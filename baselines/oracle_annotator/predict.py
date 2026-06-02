#!/usr/bin/env python3
"""Oracle baseline: pretend annotator X (a/b/c) is the model.

Used as a sanity check on the eval pipeline. Scores should be near-perfect
(but not perfect, because gold is the median of all three annotators).

Usage:
    python3 baselines/oracle_annotator/predict.py --annotator a
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml


SRT_TIME = re.compile(r"(\d+):(\d{2}):(\d{2}),(\d{3})")
LABEL = re.compile(r"\[([^\]]+)\]")


def load_env(p: Path) -> dict[str, str]:
    env = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def srt_seconds(ts: str) -> float:
    h, m, s, ms = SRT_TIME.match(ts).groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt(path: Path) -> list[tuple[float, float, str]]:
    out = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        idx = 1 if lines[0].strip().isdigit() else 0
        if idx >= len(lines) or "-->" not in lines[idx]:
            continue
        a, b = [t.strip() for t in lines[idx].split("-->")]
        body = " ".join(lines[idx + 1:]).strip()
        m = LABEL.match(body)
        if m:
            out.append((srt_seconds(a), srt_seconds(b), m.group(1)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", choices=("a", "b", "c"), default="a")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent.parent
    env = load_env(repo / ".env")
    root = (Path(env["TT_BENCHMARK_DATA"]) if env.get("TT_BENCHMARK_DATA") else Path(env["DATA_ROOT"]) / env["BATCH"])
    out_dir = repo / "predictions" / f"oracle_annotator_{args.annotator}"
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical = {}
    for canon, fine_list in yaml.safe_load((repo / "eval" / "label_map.yaml").read_text()).items():
        for f in fine_list:
            canonical[f] = canon

    sample_dirs = sorted([p for p in root.iterdir() if p.is_dir()],
                         key=lambda p: int(p.name) if p.name.isdigit() else p.name)
    total = 0
    for d in sample_dirs:
        preds = []
        for sp in (1, 2):
            srt = d / f"speaker_{sp}_annotation_{args.annotator}.srt"
            for s, e, lbl in parse_srt(srt):
                canon = canonical.get(lbl)
                if canon == "Turn":
                    preds.append({"speaker": sp, "time": round(e, 4), "label": "EOT"})
                elif canon == "Interruption":
                    preds.append({"speaker": sp, "time": round(s, 4), "label": "Interruption"})
                elif canon in ("Backchannel", "Overlap", "Laughter", "NonContent"):
                    preds.append({"speaker": sp, "time": round(s, 4), "label": canon})
        with (out_dir / f"{d.name}.jsonl").open("w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        total += len(preds)
    print(f"Wrote {total} predictions for annotator {args.annotator} to {out_dir}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
