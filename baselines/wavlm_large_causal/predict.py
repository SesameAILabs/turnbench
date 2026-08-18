#!/usr/bin/env python3
"""WavLM-Large Causal Predictor — per-channel inference on TurnBench.

Downloads the checkpoint from HuggingFace and runs a single causal forward
pass per speaker channel.  No sliding window — fully streaming.

Architecture:
  WavLM-Large (frozen, causal-masked)    1024d @50fps
  Conv1d stride-2 subsampling              1024→256d, 50→25fps (40ms)
  4-layer causal TransformerEncoder        256d, 4 heads, FFN 1024
  Linear head                             256→5  {C, NA, I, BC, T}

Setup:
    pip install espnet s3prl soundfile numpy huggingface_hub
    # CausalS3prlFrontend is NOT in stock ESPnet — install it from the HF repo:
    #   wget -P $(python -c "import espnet2; print(espnet2.__path__[0])")/asr/frontend/ \\
    #       https://huggingface.co/ZhuoyanTao/causal-wavlm-turn-taking/resolve/main/espnet2/asr/frontend/causal_s3prl.py

Usage:
    python -m baselines.wavlm_large_causal.predict                   # score on dev
    python -m baselines.wavlm_large_causal.predict --out preds.json  # write predictions
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from turnbench.data import DEV_DATASET, Conversation, conversation, conversation_ids, resolve_dataset  # noqa: E402
from turnbench.submission import SCHEMA_VERSION, ConversationPrediction, SpeakerEvents, Submission  # noqa: E402
from turnbench.sweep import ConversationProbs, ProbsFile, REFRACTORY_S, SpeakerProbs, commit_events  # noqa: E402

HF_REPO = "ZhuoyanTao/causal-wavlm-turn-taking"
CKPT_DIR = "tt_pred_large_turn_swbd_res"
SR = 16000
FRAME_HZ = 25.0        # 40 ms stride after stride-2 subsampling
FIRST_FRAME_S = 0.20   # skip first 200 ms (5 frames)
C_IDX, NA_IDX, I_IDX, BC_IDX, T_IDX = 0, 1, 2, 3, 4

# Operating point (rule 2: highest recall at fp_rate ≤ 0.1, from turnbench.sweep)
EOT_THETA = 0.85
INT_THETA = 0.20

_model = None


def _load_model(device: str):
    global _model
    if _model is not None:
        return _model
    from huggingface_hub import hf_hub_download

    # Download checkpoint
    ckpt_path = hf_hub_download(HF_REPO, f"{CKPT_DIR}/valid.loss.best.pth")

    # Import model class (requires CausalS3prlFrontend in ESPnet)
    # Try importing from HF repo's pyscripts first, then from PYTHONPATH
    pyscripts = Path(hf_hub_download(HF_REPO, "pyscripts/turn_taking_predictor_model.py")).parent
    sys.path.insert(0, str(pyscripts))
    from turn_taking_predictor_model import CausalTurnTakingPredictor

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config", {})
    model = CausalTurnTakingPredictor(
        upstream_name=cfg.get("upstream_name", "wavlm_large"),
        model_dim=cfg.get("model_dim", 256),
        freeze_upstream=True,
        causal_upstream=cfg.get("causal_upstream", True),
        causal_encoder=cfg.get("causal_encoder", True),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    _model = model
    print(f"Loaded {CKPT_DIR} on {device}", flush=True)
    return model


def _to_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SR)
    return audio


def _probs_for_signal(model, sig: np.ndarray, device: str,
                      chunk_s: float = 30.0, overlap_s: float = 5.0) -> np.ndarray:
    """Per-channel forward pass → (T, 5) probs at 25 Hz, first 200 ms skipped."""
    chunk_samp = int(chunk_s * SR)
    overlap_samp = int(overlap_s * SR)
    min_start_frames = int(FIRST_FRAME_S * FRAME_HZ)  # 5

    if len(sig) <= chunk_samp:
        # Short enough for single forward pass
        with torch.no_grad():
            x = torch.from_numpy(sig).unsqueeze(0).to(device)
            L = torch.tensor([len(sig)], dtype=torch.long, device=device)
            log_probs, feat_len = model(x, L)
            probs = log_probs[0, :int(feat_len[0])].exp().cpu().numpy()
        return probs[min_start_frames:]

    # Chunked inference for long audio
    # Get overlap frame count
    with torch.no_grad():
        dummy = torch.zeros(1, overlap_samp, device=device)
        _, fl = model(dummy, torch.tensor([overlap_samp], device=device))
        overlap_frames = int(fl[0].item())

    all_probs = []
    start = 0
    chunk_idx = 0
    while start < len(sig):
        end = min(start + chunk_samp, len(sig))
        chunk = sig[start:end]
        with torch.no_grad():
            x = torch.from_numpy(chunk).unsqueeze(0).to(device)
            L = torch.tensor([len(chunk)], dtype=torch.long, device=device)
            log_probs, feat_len = model(x, L)
            probs = log_probs[0, :int(feat_len[0])].exp().cpu().numpy()

        if chunk_idx == 0:
            probs = probs[min_start_frames:]
        else:
            probs = probs[overlap_frames:]

        if len(probs) > 0:
            all_probs.append(probs)

        start += chunk_samp - overlap_samp
        chunk_idx += 1

    return np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 5), dtype=np.float32)


def _frame_count(duration_s: float) -> int:
    return math.floor(duration_s * FRAME_HZ)


def _eot_score(probs: np.ndarray) -> np.ndarray:
    """EOT = P(NA) + P(T)."""
    return (probs[:, NA_IDX] + probs[:, T_IDX]).astype(np.float32)


def _int_score(probs: np.ndarray) -> np.ndarray:
    """INT = P(I)."""
    return probs[:, I_IDX].astype(np.float32)


def _snap(score: np.ndarray, expected: int) -> list[float]:
    """Truncate or pad to canonical frame count."""
    if len(score) >= expected:
        return score[:expected].tolist()
    return np.pad(score, (0, expected - len(score))).tolist()


def predict(conv: Conversation, device: str):
    """Run inference on both channels, return (eot1, eot2, int1, int2) score arrays."""
    model = _load_model(device)
    p1 = _probs_for_signal(model, _to_16k(*conv.audio(1)), device)
    p2 = _probs_for_signal(model, _to_16k(*conv.audio(2)), device)
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
    print(f"wavlm_large_causal — {len(predictions)} conversations")
    for name, s in (("EOT", scores.task_eot), ("INT", scores.task_int)):
        recall, fp_rate, latency = task_cells(s)
        print(f"  {name}: recall={recall} fp_rate={fp_rate} latency_ms={latency}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
