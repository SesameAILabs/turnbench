#!/usr/bin/env python3
"""Mimi-based Endpointer (Kyutai) — prediction entry point.

Two-stream LSTM over Mimi embeddings, trained on SpokenWOZ.
5-class output: [silence, system, system_end, user, user_end]

Score mapping to unified benchmark format:
  eot_score_speaker_1          <- P(user_end)
  eot_score_speaker_2          <- P(system_end)
  interruption_score_speaker_1 <- P(user)      [proxy: speaker 1 starts while sys active]
  interruption_score_speaker_2 <- P(system)    [proxy: speaker 2 starts while usr active]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runner import run  # noqa: E402

from model import (  # noqa: E402
    MimiLSTM, AudioFeatureExtractor,
    MODEL_DEFAULTS, AUDIO_DEFAULTS,
    load_model,
)

# Class indices — order from special_tokens in training config:
# {bos: 0, system_end: 1, user_end: 2, system: 3, user: 4}
IDX_BOS         = 0
IDX_SYSTEM_END  = 1
IDX_USER_END    = 2
IDX_SYSTEM      = 3
IDX_USER        = 4

CHECKPOINT = Path(__file__).resolve().parent / "checkpoint.pt"

_model: MimiLSTM | None = None
_extractor: AudioFeatureExtractor | None = None


def _get_model(device: str) -> tuple[MimiLSTM, AudioFeatureExtractor]:
    global _model, _extractor
    if _model is None:
        _model = load_model(str(CHECKPOINT), device=device)
    if _extractor is None:
        _extractor = AudioFeatureExtractor(**AUDIO_DEFAULTS, device=device)
    return _model, _extractor


def predict_scores(sample_dir: Path) -> dict:
    import time
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, extractor = _get_model(device)

    target_sr = AUDIO_DEFAULTS["sr"]

    def load_wav(path):
        wav, sr = sf.read(path, dtype="float32")
        if sr != target_sr:
            wav = torchaudio.functional.resample(
                torch.from_numpy(wav), sr, target_sr
            ).numpy()
        return wav

    wav1 = load_wav(sample_dir / "speaker_1_audio.wav")
    wav2 = load_wav(sample_dir / "speaker_2_audio.wav")

    h1, c1 = model.init_hidden(1, device)
    h2, c2 = model.init_hidden(1, device)
    all_logits = []

    with torch.no_grad():
        for feat1_chunk, feat2_chunk in extractor.stream(wav1, wav2):
            for t in range(feat1_chunk.shape[-1]):
                logits, h1, c1, h2, c2 = model.infer_ar_step(
                    feat1_chunk[:, :, t].to(device),
                    feat2_chunk[:, :, t].to(device),
                    h1, c1, h2, c2,
                )
                all_logits.append(logits)

    out = torch.stack(all_logits, dim=1)                   # (1, T, 5)
    probs = torch.softmax(out[0], dim=-1).cpu().numpy()    # (T, 5)

    duration = len(wav1) / target_sr
    elapsed = time.time() - t0
    print(f"{sample_dir.name}: {duration:.1f}s audio → {elapsed:.1f}s inference ({duration/elapsed:.1f}x RT)", flush=True)

    return {
        "frame_rate_hz":               AUDIO_DEFAULTS["frame_rate_hz"],
        "eot_score_speaker_1":         probs[:, IDX_USER_END],
        "eot_score_speaker_2":         probs[:, IDX_SYSTEM_END],
        "interruption_score_speaker_1": probs[:, IDX_USER],
        "interruption_score_speaker_2": probs[:, IDX_SYSTEM],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=None, help="Path to split file (e.g. eval/splits/dev.txt)")
    parser.add_argument("--run-name", default="mimi_endpointer")
    args = parser.parse_args()

    split_file = Path(args.split) if args.split else None
    sys.exit(run("mimi_endpointer", predict_scores, run_name=args.run_name, split_file=split_file))
