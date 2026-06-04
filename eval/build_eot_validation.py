#!/usr/bin/env python3
"""Build a binary-validation labeling package for the derived EOT gold.

For each sampled EOT in `stats_out/consensus_eot/<task_id>.jsonl`, emits a
short stereo clip (L=speaker_1, R=speaker_2) centered on the EOT moment
plus a CSV row asking the annotator a single yes/no question:

    "Is this a real end-of-turn for speaker {speaker}?"

This is the cheap validation pass — annotators do NOT re-label, they just
confirm that the 3-of-3-derived EOTs are real turn-ends. Aggregating the
binary answers gives you an empirical confirmation rate that you can
quote in the paper as evidence that the derivation rule produces valid
EOTs.

Sampling:
    Stratified by task_id (uniform across conversations) and speaker, up
    to --n-samples events (default 200). Use --all to emit clips for
    every EOT in the gold set.

Outputs (under stats_out/eot_validation/):
    clips/<task_id>_<idx>_s{speaker}_<time>.wav   stereo, 24 kHz PCM_16
    labels.csv                                     one row per clip with
        columns task_id, eot_idx, speaker, eot_time_s, clip_start_s,
        clip_end_s, clip_path, is_real_eot (empty — annotator fills),
        notes (optional)
    README.md                                      annotator instructions
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from eval.consensus import load_env


TARGET_SR = 24_000
CLIP_BEFORE_S = 4.0
CLIP_AFTER_S = 3.0


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Read WAV (any sample rate / dtype), return (float32 mono, sample_rate)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32, copy=False), sr


def resample(data: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    if src_sr == tgt_sr:
        return data
    from math import gcd
    g = gcd(src_sr, tgt_sr)
    return resample_poly(data, tgt_sr // g, src_sr // g).astype(np.float32)


def extract_clip(s1: np.ndarray, s2: np.ndarray, sr: int,
                 t_eot: float, out: Path) -> tuple[float, float]:
    """Write stereo clip centered on t_eot (L=s1, R=s2), return (start, end)."""
    n = min(len(s1), len(s2))
    lo = max(0, int((t_eot - CLIP_BEFORE_S) * sr))
    hi = min(n, int((t_eot + CLIP_AFTER_S) * sr))
    clip_s1 = s1[lo:hi]
    clip_s2 = s2[lo:hi]
    m = min(len(clip_s1), len(clip_s2))
    stereo = np.stack([clip_s1[:m], clip_s2[:m]], axis=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), stereo, sr, subtype="PCM_16")
    return lo / sr, hi / sr


def stratified_sample(eots: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Pick `n` events with rough balance across task_id and speaker."""
    rng = random.Random(seed)
    by_key: dict[tuple[str, int], list[dict]] = {}
    for ev in eots:
        by_key.setdefault((ev["task_id"], ev["speaker"]), []).append(ev)
    # Round-robin pull one from each (task, speaker) bucket until we have n.
    buckets = list(by_key.values())
    for b in buckets:
        rng.shuffle(b)
    out: list[dict] = []
    i = 0
    while len(out) < n and any(buckets):
        b = buckets[i % len(buckets)]
        if b:
            out.append(b.pop())
        else:
            buckets.pop(i % len(buckets))
            continue
        i += 1
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=200,
                    help="Number of EOTs to include in the validation set.")
    ap.add_argument("--all", action="store_true",
                    help="Emit clips for every EOT (overrides --n-samples).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Override default stats_out/eot_validation/.")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    env = load_env(repo / ".env")
    root = (Path(env["TT_BENCHMARK_DATA"]) if env.get("TT_BENCHMARK_DATA")
            else Path(env["DATA_ROOT"]) / env["BATCH"])
    stats_dir = Path(env.get("STATS_DIR", repo / "stats_out"))
    eot_dir = stats_dir / "consensus_eot"
    out_dir = args.out_dir or (stats_dir / "eot_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clips").mkdir(parents=True, exist_ok=True)

    # Load all EOTs.
    all_eots: list[dict] = []
    for f in sorted(eot_dir.glob("*.jsonl")):
        if f.name.startswith("_"):
            continue
        task_id = f.stem
        for i, line in enumerate(f.read_text().splitlines()):
            if not line.strip():
                continue
            ev = json.loads(line)
            ev["task_id"] = task_id
            ev["eot_idx"] = i
            all_eots.append(ev)

    if not all_eots:
        print(f"No EOTs found under {eot_dir}", file=sys.stderr)
        return 1

    sample = all_eots if args.all else stratified_sample(
        all_eots, args.n_samples, seed=args.seed)
    print(f"Sampling {len(sample)} of {len(all_eots)} EOTs for validation",
          file=sys.stderr)

    # Group by task_id to amortize audio loading.
    rows = []
    by_task: dict[str, list[dict]] = {}
    for ev in sample:
        by_task.setdefault(ev["task_id"], []).append(ev)

    for task_id, evs in sorted(by_task.items()):
        s1_raw, src_sr = load_audio(root / task_id / "speaker_1_audio.wav")
        s2_raw, _ = load_audio(root / task_id / "speaker_2_audio.wav")
        s1 = resample(s1_raw, src_sr, TARGET_SR)
        s2 = resample(s2_raw, src_sr, TARGET_SR)
        for ev in evs:
            t = ev["time"]
            sp = ev["speaker"]
            clip_name = f"{task_id}_{ev['eot_idx']:03d}_s{sp}_{t:.2f}s.wav"
            clip_path = out_dir / "clips" / clip_name
            cs, ce = extract_clip(s1, s2, TARGET_SR, t, clip_path)
            rows.append({
                "task_id": task_id,
                "eot_idx": ev["eot_idx"],
                "speaker": sp,
                "eot_time_s": round(t, 4),
                "clip_start_s": round(cs, 3),
                "clip_end_s": round(ce, 3),
                "clip_path": f"clips/{clip_name}",
                "is_real_eot": "",
                "notes": "",
            })
        print(f"  {task_id}: {len(evs)} clips", file=sys.stderr)

    rows.sort(key=lambda r: (r["task_id"], r["eot_idx"]))
    csv_path = out_dir / "labels.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    (out_dir / "README.md").write_text(
        "# EOT Validation Labeling\n\n"
        f"{len(rows)} stereo audio clips, each centered on a 3-of-3-derived "
        "end-of-turn moment from `stats_out/consensus_eot/`.\n\n"
        f"Each clip is {CLIP_BEFORE_S:.0f} s before + {CLIP_AFTER_S:.0f} s "
        "after the EOT, mixed as L=speaker_1 / R=speaker_2, "
        "24 kHz mono-per-channel PCM_16.\n\n"
        "## Task\n\n"
        "For each row in `labels.csv`, listen to the clip and answer:\n\n"
        "**Is this a real end-of-turn for the listed `speaker`?**\n\n"
        "Fill `is_real_eot` with `yes` or `no`. Optionally use `notes` for\n"
        "any reasoning (e.g. \"speaker continued\", \"backchannel, not EOT\",\n"
        "\"laughter ambiguous\").\n\n"
        "Notes on the derivation rule (so you know what we're testing):\n\n"
        "- EOT for speaker A is fired at A's last floor-holding event END\n"
        "  whenever the next chronological turn belongs to the OTHER speaker.\n"
        "- Floor-holding events include Normal Turn, Regular Turn, Strong\n"
        "  Floor Hold, Bounded Response, Filler, Overlap, Awkward Silence,\n"
        "  Laughter.\n"
        "- All three annotators (a/b/c) had to agree on the EOT existence,\n"
        "  speaker, and time within 200 ms; gold time is the median.\n"
    )
    print(f"\nWrote {len(rows)} clips + labels.csv + README.md under {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
