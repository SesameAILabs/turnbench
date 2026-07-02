#!/usr/bin/env python3
"""Stereo inspection mixes of Gemini Live recordings: left = the human
speaker's channel (from the dataset parquet), right = the agent's recorded
output — time-aligned, so barge-ins, yields, and response latency are
audible directly.

    uv run --extra eval --with google-genai --with python-dotenv --with scipy \
        python baselines/gemini/pipeline/make_inspection_mix.py \
        --runs baselines/gemini/sample_runs --task 6        # one conversation
    ... --all                              # every direction with a recording present

Writes <runs>/inspection/<task_id>_speaker_K.wav (24 kHz stereo).
"""
from __future__ import annotations

import argparse
import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent.parent))

from run_split import TEST_DATASET, _materialize_input, dataset_index  # noqa: E402

OUT_SR = 24_000


def _mono_24k(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != OUT_SR:
        g = gcd(OUT_SR, sr)
        audio = resample_poly(audio, OUT_SR // g, sr // g)
    return audio


def mix_direction(tid: str, speaker: int, runs: Path, index, work: Path) -> Path | None:
    agent_flac = runs / tid / f"speaker_{speaker}" / "output.flac"
    if not agent_flac.exists():
        return None
    shard, row_group = index[tid]
    user_wav = _materialize_input(shard, row_group, speaker,
                                  work / f"{tid}_speaker_{speaker}.wav")
    user = _mono_24k(user_wav)
    agent = _mono_24k(agent_flac)
    n = max(len(user), len(agent))
    stereo = np.zeros((n, 2), dtype=np.float32)
    stereo[: len(user), 0] = user     # left: human
    stereo[: len(agent), 1] = agent   # right: Gemini
    out = runs / "inspection" / f"{tid}_speaker_{speaker}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), stereo, OUT_SR, subtype="PCM_16")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=_HERE.parent / "sample_runs")
    ap.add_argument("--dataset", default=TEST_DATASET)
    ap.add_argument("--task", action="append", default=None,
                    help="conversation id (repeatable)")
    ap.add_argument("--all", action="store_true",
                    help="mix every direction with a recording present")
    args = ap.parse_args()

    index = dataset_index(args.dataset)
    work = args.runs / ".batch"
    work.mkdir(parents=True, exist_ok=True)
    tids = (args.task if args.task else
            sorted((p.name for p in args.runs.iterdir()
                    if p.is_dir() and p.name in index), key=int) if args.all else [])
    if not tids:
        sys.exit("pass --task <id> (repeatable) or --all")
    for tid in tids:
        for k in (1, 2):
            out = mix_direction(tid, k, args.runs, index, work)
            print(f"{tid}/speaker_{k}: {out if out else 'no recording yet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
