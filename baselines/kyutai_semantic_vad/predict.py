#!/usr/bin/env python3
"""Kyutai Semantic VAD — STT-based end-of-turn detection.

Uses Kyutai STT-1B's VAD head (vad_heads[2]) for continuous EOT scores.
Each speaker is treated as an independent "user" channel — the model
itself is single-channel and only this VAD/EOT head is documented, so
there is no direct cross-speaker overlap signal available. Both channels
are run through one batched stream (batch_size=2) so the ~1B-param LM
does a single forward pass per frame for both speakers, rather than two
sequential passes.

Interruption score uses 1 - vad_K ("speaker K is actively speaking, i.e.
not about to end their turn") as a single-channel proxy for "speaker K is
barging in" — directionally correct (low EOT <=> mid-utterance) but not a
true overlap/barge-in detector.

Model: kyutai/stt-1b-en_fr-candle (~1B params)
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

HF_REPO = "kyutai/stt-1b-en_fr-candle"  # the "-candle" variant carries the VAD extra-head weights

_mimi             = None
_lm_gen           = None
_silence_prefix_s = None
_audio_delay_s    = None


def _load_models(device: str):
    global _mimi, _lm_gen, _silence_prefix_s, _audio_delay_s
    if _mimi is not None:
        return
    print("Loading Kyutai STT model...", flush=True)
    info    = loaders.CheckpointInfo.from_hf_repo(HF_REPO)
    _mimi   = info.get_mimi(device=device)
    lm      = info.get_moshi(device=device, dtype=torch.bfloat16)
    _silence_prefix_s = info.stt_config.get("audio_silence_prefix_seconds", 1.0)
    _audio_delay_s    = info.stt_config.get("audio_delay_seconds", 5.0)
    # temp=0, temp_text=0: greedy — fastest; text tokens discarded, only vad_heads used
    _lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)
    print("Model loaded.", flush=True)


def _prep_wav(wav: np.ndarray, sr: int, device: str) -> torch.Tensor:
    audio = torch.from_numpy(wav).float().to(device)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # (1, T)
    return julius.resample_frac(audio, sr, _mimi.sample_rate)


def _vad_scores_joint(
    wav1: np.ndarray, sr1: int, wav2: np.ndarray, sr2: int, device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run STT VAD on both speaker channels in a single batched stream
    (batch_size=2) so the ~1B-param LM does one forward pass per frame
    for both speakers — not two sequential passes. Returns (vad1, vad2)
    at 12.5Hz."""
    a1 = _prep_wav(wav1, sr1, device)
    a2 = _prep_wav(wav2, sr2, device)

    T = max(a1.shape[-1], a2.shape[-1])
    if T % _mimi.frame_size != 0:
        T += _mimi.frame_size - T % _mimi.frame_size
    a1 = torch.nn.functional.pad(a1, (0, T - a1.shape[-1]))
    a2 = torch.nn.functional.pad(a2, (0, T - a2.shape[-1]))
    audio = torch.cat([a1, a2], dim=0)  # (2, T)

    n_prefix = math.ceil(_silence_prefix_s * _mimi.frame_rate)
    n_suffix = math.ceil(_audio_delay_s    * _mimi.frame_rate)
    silence  = torch.zeros((2, 1, _mimi.frame_size), dtype=torch.float32, device=device)

    chunks = itertools.chain(
        itertools.repeat(silence, n_prefix),
        torch.split(audio[:, None], _mimi.frame_size, dim=-1),
        itertools.repeat(silence, n_suffix),
    )

    scores1, scores2 = [], []
    with _mimi.streaming(2), _lm_gen.streaming(2):
        for chunk in chunks:
            audio_tokens = _mimi.encode(chunk)
            _, vad_heads = _lm_gen.step_with_extra_heads(audio_tokens)
            if vad_heads:
                v = vad_heads[2][:, 0, 0].cpu().float().numpy()  # (2,)
                scores1.append(v[0])
                scores2.append(v[1])

    return np.array(scores1, dtype=np.float32), np.array(scores2, dtype=np.float32)


def predict_scores(sample_dir: Path) -> dict:
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _load_models(device)

    wav1, sr1 = sphn.read(str(sample_dir / "speaker_1_audio.wav"))
    wav2, sr2 = sphn.read(str(sample_dir / "speaker_2_audio.wav"))

    vad1, vad2 = _vad_scores_joint(wav1, sr1, wav2, sr2, device)

    T = min(len(vad1), len(vad2))
    vad1, vad2 = vad1[:T], vad2[:T]

    duration = wav1.shape[-1] / sr1
    elapsed  = time.time() - t0
    print(f"{sample_dir.name}: {duration:.1f}s audio → {elapsed:.1f}s ({duration/elapsed:.1f}x RT)", flush=True)

    return {
        "frame_rate_hz":                _mimi.frame_rate,
        "eot_score_speaker_1":          vad1,
        "eot_score_speaker_2":          vad2,
        "interruption_score_speaker_1": 1 - vad1,
        "interruption_score_speaker_2": 1 - vad2,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",    default=None)
    parser.add_argument("--run-name", default="kyutai_semantic_vad")
    args = parser.parse_args()

    split_file = Path(args.split) if args.split else None
    sys.exit(run("kyutai_semantic_vad", predict_scores, run_name=args.run_name, split_file=split_file))
