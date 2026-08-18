#!/usr/bin/env python3
"""Export a benchmark split from HF into the flac layout the Moshi client reads.

`inference_moshi_dev_release.py` (and the ASR stage) expect a Mundo-style
delivery tree:

    <out>/<conversation_id>/speaker_1_audio.flac
    <out>/<conversation_id>/speaker_2_audio.flac

This dumps `mundo-ai/turn-benchmark-{dev,test}` into that layout so the
generative pipeline can run without a local delivery.

    uv run --extra eval python baselines/moshi/pipeline/export_flac_dataset.py \
        --split dev --out /path/to/turnbench_audio/dev
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from turnbench.data import DEV_DATASET, conversation, conversation_ids, resolve_dataset

SPLIT_SOURCES = {"dev": DEV_DATASET, "test": "mundo-ai/turn-benchmark-test"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), required=True,
                    help="benchmark split to export")
    ap.add_argument("--out", type=Path, required=True,
                    help="output root; one numbered folder per conversation")
    args = ap.parse_args()

    dataset = resolve_dataset(source=SPLIT_SOURCES[args.split])
    ids = conversation_ids(dataset)
    args.out.mkdir(parents=True, exist_ok=True)
    for n, cid in enumerate(sorted(ids, key=int), 1):
        conv_dir = args.out / cid
        conv_dir.mkdir(exist_ok=True)
        for speaker in (1, 2):
            dst = conv_dir / f"speaker_{speaker}_audio.flac"
            if dst.exists():
                continue
            samples, sample_rate = conversation(dataset, cid).audio(speaker)
            sf.write(dst, samples, sample_rate, format="FLAC")
        print(f"[{n}/{len(ids)}] {conv_dir}", flush=True)


if __name__ == "__main__":
    main()
