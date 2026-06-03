#!/usr/bin/env python3
"""Kyutai Semantic VAD — STT-based end-of-turn detection.

Uses Kyutai STT-1B's VAD head (vad_heads[2]) for continuous EOT scores.
Each speaker is processed independently as the "user" channel at 12.5Hz.
Interruption scores reuse the other speaker's VAD probability as a proxy.

Model: kyutai/stt-1b-en_fr (~1B params)
Frame rate: 12.5Hz
"""
from __future__ import annotations

import itertools
import math
import sys
import time
from pathlib import Path

import julius
import numpy as np
import sphn
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runner import run  # noqa: E402

import moshi.models
import moshi.models.loaders as loaders

HF_REPO         = "kyutai/stt-1b-en_fr"
SILENCE_PREFIX_S = 1.0
AUDIO_DELAY_S    = 5.0

_mimi    = None
_lm_gen  = None


def _load_models(device: str):
    global _mimi, _lm_gen
    if _mimi is not None:
        return
    print("Loading Kyutai STT model...", flush=True)
    info    = loaders.CheckpointInfo.from_hf_repo(HF_REPO)
    _mimi   = info.get_mimi(device=device)
    lm      = info.get_moshi(device=device, dtype=torch.bfloat16)
    # temp=0, temp_text=0: greedy — fastest; text tokens discarded, only vad_heads used
    _lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)
    print("Model loaded.", flush=True)


def _vad_scores(wav: np.ndarray, sr: int, device: str) -> np.ndarray:
    """Run STT VAD on a single-channel wav. Returns VAD probs at 12.5Hz."""
    audio = torch.from_numpy(wav).float().to(device)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # (1, T)
    audio = julius.resample_frac(audio, sr, _mimi.sample_rate)

    if audio.shape[-1] % _mimi.frame_size != 0:
        pad = _mimi.frame_size - audio.shape[-1] % _mimi.frame_size
        audio = torch.nn.functional.pad(audio, (0, pad))

    n_prefix = math.ceil(SILENCE_PREFIX_S * _mimi.frame_rate)
    n_suffix = math.ceil(AUDIO_DELAY_S    * _mimi.frame_rate)
    silence  = torch.zeros((1, 1, _mimi.frame_size), dtype=torch.float32, device=device)

    chunks = itertools.chain(
        itertools.repeat(silence, n_prefix),
        torch.split(audio[:, None], _mimi.frame_size, dim=-1),
        itertools.repeat(silence, n_suffix),
    )

    scores = []
    with _mimi.streaming(1), _lm_gen.streaming(1):
        for chunk in chunks:
            audio_tokens = _mimi.encode(chunk)
            _, vad_heads = _lm_gen.step_with_extra_heads(audio_tokens)
            if vad_heads:
                scores.append(vad_heads[2][0, 0, 0].cpu().float().item())

    return np.array(scores, dtype=np.float32)


def predict_scores(sample_dir: Path) -> dict:
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _load_models(device)

    wav1, sr1 = sphn.read(str(sample_dir / "speaker_1_audio.wav"))
    wav2, sr2 = sphn.read(str(sample_dir / "speaker_2_audio.wav"))

    vad1 = _vad_scores(wav1, sr1, device)
    vad2 = _vad_scores(wav2, sr2, device)

    T = min(len(vad1), len(vad2))
    vad1, vad2 = vad1[:T], vad2[:T]

    duration = wav1.shape[-1] / sr1
    elapsed  = time.time() - t0
    print(f"{sample_dir.name}: {duration:.1f}s audio → {elapsed:.1f}s ({duration/elapsed:.1f}x RT)", flush=True)

    return {
        "frame_rate_hz":                _mimi.frame_rate,
        "eot_score_speaker_1":          vad1,
        "eot_score_speaker_2":          vad2,
        "interruption_score_speaker_1": vad1,
        "interruption_score_speaker_2": vad2,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",    default=None)
    parser.add_argument("--run-name", default="kyutai_semantic_vad")
    args = parser.parse_args()

    split_file = Path(args.split) if args.split else None
    sys.exit(run("kyutai_semantic_vad", predict_scores, run_name=args.run_name, split_file=split_file))
