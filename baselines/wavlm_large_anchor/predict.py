#!/usr/bin/env python3
"""WavLM-Large ANCHOR (TT-only) — sliding-window inference on TurnBench.

Downloads the checkpoint from HuggingFace and runs 4 s sliding-window AR
decoding per speaker channel.  Each window is processed independently;
because windows end at the current time, no future audio is observed.

Architecture:
  WavLM-Large (frozen)
    → 4-layer Transformer audio encoder
    → 12-layer AR Transformer decoder (6-token TT-only vocabulary)
    → 5-class turn-taking distribution {C, NA, I, BC, T}

Setup:
    pip install espnet s3prl soundfile numpy huggingface_hub
    # CausalS3prlFrontend is NOT in stock ESPnet — install from HF repo:
    #   wget -P $(python -c "import espnet2; print(espnet2.__path__[0])")/asr/frontend/ \\
    #       https://huggingface.co/ZhuoyanTao/causal-wavlm-turn-taking/resolve/main/espnet2/asr/frontend/causal_s3prl.py

Usage:
    python -m baselines.wavlm_large_anchor.predict                   # score on dev
    python -m baselines.wavlm_large_anchor.predict --out preds.json  # write predictions
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from turnbench.data import DEV_DATASET, Conversation, conversation, conversation_ids, resolve_dataset  # noqa: E402
from turnbench.submission import SCHEMA_VERSION, ConversationPrediction, SpeakerEvents, Submission  # noqa: E402
from turnbench.sweep import ConversationProbs, ProbsFile, REFRACTORY_S, SpeakerProbs, commit_events  # noqa: E402

HF_REPO = "ZhuoyanTao/causal-wavlm-turn-taking"
CKPT_DIR = "universa_turn_taking_only_turn_a40"
SR = 16000
FRAME_HZ = 25.0        # 40 ms stride
FIRST_FRAME_S = 0.20   # 200 ms warm-up skip
CONTEXT_S = 4.0         # sliding window context
STRIDE_S = 0.04         # 40 ms stride per frame
BATCH_SIZE = 64         # windows per forward pass
N_CLASSES = 5
# Token→LabelIndex: C=0, NA=1, I=2, T→4, BC→3
TOKEN_TO_LABELINDEX = [0, 1, 2, 4, 3]
C_IDX, NA_IDX, I_IDX, BC_IDX, T_IDX = 0, 1, 2, 3, 4

# Operating point (rule 2: highest recall at fp_rate ≤ 0.1)
EOT_THETA = 0.90
INT_THETA = 0.20

_model = None
_universa = None
_tt_meta_idx = None
_tt_val0_idx = None


def _load_model(device: str):
    global _model, _universa, _tt_meta_idx, _tt_val0_idx
    if _model is not None:
        return
    from huggingface_hub import hf_hub_download
    from espnet2.tasks.universa import UniversaTask

    if os.environ.get("TT_TF32", "0") == "1":  # ~1.3-2x on H100 tensor cores; gate on a
        torch.backends.cuda.matmul.allow_tf32 = True   # probs-delta check before trusting:
        torch.backends.cudnn.allow_tf32 = True         # the AR decode is discrete.

    # Download checkpoint + config + tokenizer data
    model_dir = Path(hf_hub_download(HF_REPO, f"{CKPT_DIR}/valid.loss.best.pth")).parent
    for f in ["config.yaml", "data/metric2id", "data/metric2type", "data/tokens.json"]:
        hf_hub_download(HF_REPO, f"{CKPT_DIR}/{f}")

    config_file = str(model_dir / "config.yaml")
    model_file = str(model_dir / "valid.loss.best.pth")

    # config.yaml references its tokenizer data by path relative to the exp dir
    # (data/metric2id etc., the espnet convention) — build from inside it.
    cwd = os.getcwd()
    os.chdir(model_dir)
    try:
        _model, _ = UniversaTask.build_model_from_file(config_file, model_file, device)
    finally:
        os.chdir(cwd)
    _model.to(device).eval()
    _universa = _model.universa

    # Get turn-taking token indices
    meta_key = "turn_taking@meta_label"
    val0_key = "turn_taking@0"
    _tt_meta_idx = _universa.metric_tokenizer.vocab_indices[meta_key]
    _tt_val0_idx = _universa.metric_tokenizer.vocab_indices[val0_key]
    print(f"Loaded {CKPT_DIR} on {device} (tt_meta={_tt_meta_idx}, tt_val0={_tt_val0_idx})", flush=True)


def _to_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SR)
    return audio


def _ar_decode_batch(audio_enc, audio_enc_lengths) -> np.ndarray:
    """Batched greedy AR decode → (B, N_CLASSES) probs in LabelIndex order."""
    B = audio_enc.shape[0]
    device = audio_enc.device
    sos_id = _universa.sos

    seq = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
    seq_l = torch.ones(B, dtype=torch.long, device=device)
    found = torch.zeros(B, dtype=torch.bool, device=device)
    all_probs = torch.full((B, N_CLASSES), 1.0 / N_CLASSES, dtype=torch.float32, device=device)

    for _ in range(50):
        dec_out, _ = _universa.decoder(audio_enc, audio_enc_lengths, seq, seq_l)
        next_toks = dec_out[:, -1, :].argmax(dim=-1)
        hit = (~found) & (next_toks == _tt_meta_idx)
        if hit.any():
            hit_idx = hit.nonzero(as_tuple=True)[0]
            seq_hit = torch.cat([
                seq[hit_idx],
                torch.full((hit_idx.shape[0], 1), _tt_meta_idx, dtype=torch.long, device=device),
            ], dim=1)
            seq_l_hit = seq_l[hit_idx] + 1
            dec_out2, _ = _universa.decoder(
                audio_enc[hit_idx], audio_enc_lengths[hit_idx], seq_hit, seq_l_hit)
            val_logits = dec_out2[:, -1, _tt_val0_idx:_tt_val0_idx + N_CLASSES]
            all_probs[hit_idx] = F.softmax(val_logits.float(), dim=-1)
            found[hit_idx] = True
        if found.all():
            break
        active = ~found & (next_toks != _universa.eos)
        if not active.any():
            break
        seq = torch.cat([seq, next_toks.unsqueeze(1)], dim=1)
        seq_l = seq_l + 1

    # Remap token→LabelIndex
    tp = all_probs.cpu().numpy()
    lp = np.zeros_like(tp)
    for ti, li in enumerate(TOKEN_TO_LABELINDEX):
        lp[:, li] = tp[:, ti]
    return lp


def _probs_for_signal(sig: np.ndarray, device: str) -> np.ndarray:
    """Sliding-window inference → (T, 5) probs at 25 Hz, first 200 ms skipped."""
    context_samples = int(CONTEXT_S * SR)
    stride_samples = int(STRIDE_S * SR)
    min_start_sample = int(FIRST_FRAME_S * SR)

    windows: List[np.ndarray] = []
    frame_end = min_start_sample + stride_samples
    while frame_end <= len(sig):
        frame_start = max(0, frame_end - context_samples)
        chunk = sig[frame_start:frame_end]
        if len(chunk) < context_samples:
            chunk = np.concatenate([np.zeros(context_samples - len(chunk), dtype=np.float32), chunk])
        windows.append(chunk)
        frame_end += stride_samples

    if not windows:
        return np.zeros((0, 5), dtype=np.float32)

    lengths = torch.full((BATCH_SIZE,), context_samples, dtype=torch.long, device=device)
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(windows), BATCH_SIZE):
            batch = windows[i:i + BATCH_SIZE]
            B = len(batch)
            audio_batch = torch.tensor(np.stack(batch), dtype=torch.float32).to(device)
            feats, feats_len = _model._extract_feats(audio_batch, lengths[:B])
            audio_enc, enc_len = _universa.encode(feats, feats_len)
            all_probs.extend(_ar_decode_batch(audio_enc, enc_len))

    return np.array(all_probs, dtype=np.float32)


def _frame_count(duration_s: float) -> int:
    return math.floor(duration_s * FRAME_HZ)


def _eot_score(probs: np.ndarray) -> np.ndarray:
    """EOT = P(NA) + P(T)."""
    return (probs[:, NA_IDX] + probs[:, T_IDX]).astype(np.float32)


def _int_score(probs: np.ndarray) -> np.ndarray:
    """INT = 1 - P(C)."""
    return (1.0 - probs[:, C_IDX]).astype(np.float32)


def _snap(score: np.ndarray, expected: int) -> list[float]:
    if len(score) >= expected:
        return score[:expected].tolist()
    return np.pad(score, (0, expected - len(score))).tolist()


def predict(conv: Conversation, device: str):
    _load_model(device)
    p1 = _probs_for_signal(_to_16k(*conv.audio(1)), device)
    p2 = _probs_for_signal(_to_16k(*conv.audio(2)), device)
    return _eot_score(p1), _eot_score(p2), _int_score(p1), _int_score(p2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEV_DATASET)
    parser.add_argument("--out", default=None, help="write predictions JSON")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eot-theta", type=float, default=EOT_THETA)
    parser.add_argument("--int-theta", type=float, default=INT_THETA)
    parser.add_argument("--probs-out-dir", type=Path, default=None,
                        help="write per-frame probs-{eot,int}.json for this dataset (no predictions)")
    parser.add_argument("--shard", type=int, nargs=2, metavar=("K", "N"), default=None,
                        help="process ids[K::N]; probs files get a .shardK-of-N suffix")
    args = parser.parse_args()

    dataset = resolve_dataset(source=args.dataset)
    ids = conversation_ids(dataset)
    if args.shard is not None:
        ids = ids[args.shard[0]::args.shard[1]]

    # Probs mode: emit the continuous per-frame scores on the canonical grid —
    # the sweepable twin of the committed predictions (see baselines/README.md).
    if args.probs_out_dir is not None:
        entries = {"eot": [], "int": []}
        for cid in ids:
            conv = conversation(dataset, cid)
            eot1, eot2, int1, int2 = predict(conv, args.device)
            n = _frame_count(conv.duration_s)
            entries["eot"].append(ConversationProbs(
                conversation_id=cid,
                speaker_1=SpeakerProbs(prob=_snap(eot1, n)),
                speaker_2=SpeakerProbs(prob=_snap(eot2, n))))
            entries["int"].append(ConversationProbs(
                conversation_id=cid,
                speaker_1=SpeakerProbs(prob=_snap(int1, n)),
                speaker_2=SpeakerProbs(prob=_snap(int2, n))))
            print(f"  {cid}: probs done", flush=True)
        args.probs_out_dir.mkdir(parents=True, exist_ok=True)
        sfx = f".shard{args.shard[0]}-of-{args.shard[1]}" if args.shard is not None else ""
        for task in ("eot", "int"):
            pf = ProbsFile(schema_version=1, task=task, frame_rate_hz=FRAME_HZ, probs=entries[task])
            path = args.probs_out_dir / f"probs-{task}{sfx}.json"
            path.write_text(pf.model_dump_json())
            print(f"Wrote {path}", flush=True)
        return 0

    predictions = []
    for cid in ids:
        conv = conversation(dataset, cid)
        eot1, eot2, int1, int2 = predict(conv, args.device)
        n = _frame_count(conv.duration_s)
        predictions.append(ConversationPrediction(
            conversation_id=cid,
            speaker_1=SpeakerEvents(
                eot=commit_events(_snap(eot1, n), FRAME_HZ, args.eot_theta),
                interruption=commit_events(_snap(int1, n), FRAME_HZ, args.int_theta),
            ),
            speaker_2=SpeakerEvents(
                eot=commit_events(_snap(eot2, n), FRAME_HZ, args.eot_theta),
                interruption=commit_events(_snap(int2, n), FRAME_HZ, args.int_theta),
            ),
        ))
        print(f"  {cid}: done", flush=True)

    submission = Submission(schema_version=SCHEMA_VERSION, predictions=predictions)

    if args.out:
        Path(args.out).write_text(submission.model_dump_json(indent=2), encoding="utf-8")
        print(f"Wrote {len(predictions)} predictions to {args.out}")
        return 0

    from turnbench.score import score_submission, task_cells
    scores = score_submission(submission, dataset)
    print(f"wavlm_large_anchor — {len(predictions)} conversations")
    for name, s in (("EOT", scores.task_eot), ("INT", scores.task_int)):
        recall, fp_rate, latency = task_cells(s)
        print(f"  {name}: recall={recall} fp_rate={fp_rate} latency_ms={latency}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
