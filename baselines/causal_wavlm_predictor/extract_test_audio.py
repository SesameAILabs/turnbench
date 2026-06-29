#!/usr/bin/env python3
"""Extract audio from HF test dataset to wav files + wav.scp for predictor inference.

Must run on a compute node with >60GB RAM (the test Arrow table is ~13 GB).
Uses the ttbench Python 3.11 env.

    python extract_test_audio.py --output-dir /path/to/test_audio
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from eval.data import conversation, conversation_ids, resolve_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mundo-ai/turn-benchmark-test")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write wav files and wav.scp")
    args = parser.parse_args()

    out = Path(args.output_dir)
    wav_dir = out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {args.dataset} ...")
    dataset = resolve_dataset(source=args.dataset)
    ids = conversation_ids(dataset)
    print(f"Found {len(ids)} conversations")

    scp_lines = []
    TARGET_SR = 16000
    for cid in ids:
        conv = conversation(dataset, cid)
        for spk in (1, 2):
            samples, sr = conv.audio(spk)
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            if sr != TARGET_SR:
                from scipy.signal import resample_poly
                samples = resample_poly(samples, TARGET_SR, sr).astype(np.float32)
                sr = TARGET_SR
            wav_path = wav_dir / f"{cid}_spk{spk}.wav"
            sf.write(str(wav_path), samples, sr)
            utt_id = f"{cid}_spk{spk}"
            scp_lines.append(f"{utt_id} {wav_path}\n")

    scp_path = out / "wav.scp"
    with open(scp_path, "w") as f:
        f.writelines(sorted(scp_lines))

    print(f"Wrote {len(scp_lines)} entries to {scp_path}")


if __name__ == "__main__":
    main()
