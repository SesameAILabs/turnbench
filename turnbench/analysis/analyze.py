#!/usr/bin/env python3
"""Compute dataset statistics for the turn-taking benchmark.

Reads DATA_ROOT/BATCH from .env, scans every sample dir, and writes:
    stats_out/summary.json    aggregate numbers
    stats_out/per_sample.csv  per-sample row
    stats_out/labels.csv      label x annotator counts
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf


SRT_TIME = re.compile(r"(\d+):(\d{2}):(\d{2}),(\d{3})")
LABEL = re.compile(r"\[([^\]]+)\]")
ANNOTATORS = ("a", "b", "c")
SPEAKERS = (1, 2)


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def srt_seconds(ts: str) -> float:
    h, m, s, ms = SRT_TIME.match(ts).groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt(path: Path) -> list[tuple[float, float, str, str]]:
    """Returns list of (start, end, label, text). label is '' if missing."""
    out = []
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip())
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        # skip index line if numeric
        idx = 1 if lines[0].strip().isdigit() else 0
        if "-->" not in lines[idx]:
            continue
        start_s, end_s = [t.strip() for t in lines[idx].split("-->")]
        start, end = srt_seconds(start_s), srt_seconds(end_s)
        body = " ".join(lines[idx + 1:]).strip()
        m = LABEL.match(body)
        label = m.group(1) if m else ""
        text = body[m.end():].strip() if m else body
        out.append((start, end, label, text))
    return out


def wav_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    env = load_env(repo / ".env")
    root = (Path(env["TT_BENCHMARK_DATA"]) if env.get("TT_BENCHMARK_DATA") else Path(env["DATA_ROOT"]) / env["BATCH"])
    out_dir = Path(env.get("STATS_DIR", repo / "stats_out"))
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted([p for p in root.iterdir() if p.is_dir()],
                         key=lambda p: int(p.name) if p.name.isdigit() else p.name)
    print(f"Found {len(sample_dirs)} samples under {root}", file=sys.stderr)

    per_sample_rows = []
    # label_counts[annotator][label] = count
    label_counts: dict[str, Counter] = {a: Counter() for a in ANNOTATORS}
    label_dur: dict[str, dict[str, float]] = {a: defaultdict(float) for a in ANNOTATORS}
    actor_ids: set[str] = set()
    actor_sample_count: Counter = Counter()

    total_combined_dur = 0.0
    total_speaker_dur = 0.0
    missing = []

    for d in sample_dirs:
        task_id = d.name
        meta_p = d / "metadata.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        for k in ("speaker_1_actor_id", "speaker_2_actor_id"):
            if meta.get(k):
                actor_ids.add(meta[k])
                actor_sample_count[meta[k]] += 1

        combined = d / "combined_audio.wav"
        comb_dur = wav_duration(combined) if combined.exists() else 0.0
        total_combined_dur += comb_dur

        row: dict = {"task_id": task_id, "combined_dur_s": round(comb_dur, 3)}
        for sp in SPEAKERS:
            wav_p = d / f"speaker_{sp}_audio.wav"
            dur = wav_duration(wav_p) if wav_p.exists() else 0.0
            total_speaker_dur += dur
            row[f"speaker_{sp}_dur_s"] = round(dur, 3)
            for ann in ANNOTATORS:
                srt_p = d / f"speaker_{sp}_annotation_{ann}.srt"
                if not srt_p.exists():
                    missing.append(str(srt_p))
                    row[f"sp{sp}_ann_{ann}_events"] = 0
                    continue
                events = parse_srt(srt_p)
                row[f"sp{sp}_ann_{ann}_events"] = len(events)
                for start, end, label, _ in events:
                    if label:
                        label_counts[ann][label] += 1
                        label_dur[ann][label] += max(0.0, end - start)
        per_sample_rows.append(row)

    # Write per_sample.csv
    csv_path = out_dir / "per_sample.csv"
    fieldnames = ["task_id", "combined_dur_s",
                  "speaker_1_dur_s", "speaker_2_dur_s"]
    for sp in SPEAKERS:
        for ann in ANNOTATORS:
            fieldnames.append(f"sp{sp}_ann_{ann}_events")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_sample_rows)

    # Write labels.csv
    all_labels = sorted({lbl for c in label_counts.values() for lbl in c})
    labels_path = out_dir / "labels.csv"
    with labels_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "count_a", "count_b", "count_c",
                    "dur_s_a", "dur_s_b", "dur_s_c"])
        for lbl in all_labels:
            w.writerow([lbl,
                        label_counts["a"][lbl], label_counts["b"][lbl], label_counts["c"][lbl],
                        round(label_dur["a"][lbl], 2),
                        round(label_dur["b"][lbl], 2),
                        round(label_dur["c"][lbl], 2)])

    summary = {
        "data_root": str(root),
        "n_samples": len(sample_dirs),
        "n_unique_actors": len(actor_ids),
        "total_combined_audio_hours": round(total_combined_dur / 3600, 3),
        "total_per_speaker_audio_hours": round(total_speaker_dur / 3600, 3),
        "events_per_annotator": {a: sum(label_counts[a].values()) for a in ANNOTATORS},
        "unique_labels_per_annotator": {a: len(label_counts[a]) for a in ANNOTATORS},
        "missing_srt_files": len(missing),
        "missing_examples": missing[:10],
        "top_labels": {
            a: label_counts[a].most_common(15) for a in ANNOTATORS
        },
        "actor_repeat_distribution": dict(Counter(actor_sample_count.values())),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
