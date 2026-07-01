# Author: Sathvik Udupa (2026)
# Email:  udupa@fit.vutbr.cz
# Paper:  Streaming Endpointer for Spoken Dialogue using Neural Audio Codecs and Label-Delayed Training, https://arxiv.org/abs/2506.07081, ASRU 2025

"""Mimi Endpointer — dev scoring or submission JSON for the TURN benchmark.

Runs inference over all conversations (fast full-sequence mode by default),
then either:
  - (default, no --out) sweeps all thresholds on the dev set, prints a
    recall / FP-rate / latency table, saves sweep_results.json.
  - (--out FILE) writes a submission JSON using the checkpoint's default
    thresholds (or --threshold-eot / --threshold-int overrides).
    On dev, also scores and prints metrics. On test, just writes.

Usage:
    # dev sweep (score in-place):
    python -m baselines.mimi_endpointer.predict

    # dev submission JSON:
    python -m baselines.mimi_endpointer.predict --out predictions-dev.json

    # test submission JSON:
    python -m baselines.mimi_endpointer.predict \\
        --dataset mundo-ai/turn-benchmark-test \\
        --out predictions-test.json

    # other checkpoint:
    python -m baselines.mimi_endpointer.predict \\
        --run-name swbd_oto_d1f --out swbd_oto_d1f-dev.json

    # probs only (no threshold):
    python -m baselines.mimi_endpointer.predict --probs-only \\
        [--dataset mundo-ai/turn-benchmark-test] \\
        [--probs-out-dir predictions/mimi_endpointer_probs]
"""
from __future__ import annotations

import sys
import time

from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE))

from eval.data import (                                                      # noqa: E402
    DEV_DATASET,
    Dataset,
    conversation,
    conversation_ids,
    resolve_dataset,
)
from eval.score import score_submission                                       # noqa: E402
from eval.submission import (                                                # noqa: E402
    SCHEMA_VERSION,
    ConversationPrediction,
    SpeakerEvents,
    Submission,
)
from eval.sweep import (                                                     # noqa: E402
    ConversationProbs as _ConversationProbs,
    ProbsFile as _ProbsFile,
    SCHEMA_VERSION as _PROBS_SCHEMA_VERSION,
    SpeakerProbs as _SpeakerProbs,
    commit_events,
    frame_count,
    operating_point as _operating_point,
    sweep as _eval_sweep,
)
from model import AudioFeatureExtractor, AUDIO_DEFAULTS, load_model          # noqa: E402
from huggingface_hub import hf_hub_download                                  # noqa: E402

IDX_USER   = 4  # {bos:0, system_end:1, user_end:2, system:3, user:4}
IDX_SYSTEM = 3

HF_REPO    = "viks66/mimi-endpointer"
FRAME_RATE = AUDIO_DEFAULTS["frame_rate_hz"]   # 25.0 Hz
TARGET_SR  = AUDIO_DEFAULTS["sr"]              # 24000

# (eot_thr, int_thr) in probs space:
#   eot: prob = 1 - p_user,  fire when (1-p_user) > eot_thr
#   int: prob = p_user,       fire when  p_user    > int_thr
# Fallbacks only — on dev, thresholds are auto-selected by eval.sweep.operating_point.
CHECKPOINT_DEFAULTS: dict[str, tuple[float, float]] = {
    "pretrained":   (0.95, 0.20),
    "oto_d1f":      (0.85, 0.10),
    "swbd_d1f":     (0.90, 0.25),
    "swbd_oto_d1f": (0.85, 0.30),
}


def _load_checkpoint(run_name: str, device: str, checkpoint: Path | None = None):
    if checkpoint is not None:
        local = checkpoint
    else:
        print(f"Downloading {run_name}.pt from {HF_REPO}…", flush=True)
        local = Path(hf_hub_download(HF_REPO, f"{run_name}.pt"))
    print(f"Loading checkpoint: {local}", flush=True)
    return load_model(str(local), device=device)


def _load_wav(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav), sr, TARGET_SR
        ).numpy()
    return wav


def _infer_conversation(conv, mimi_model, extractor, device, fast=True):
    """Returns (p_user, p_system) each shape (T,) at FRAME_RATE Hz."""
    wav1 = _load_wav(*conv.audio(1))
    wav2 = _load_wav(*conv.audio(2))
    with torch.no_grad():
        if fast:
            feat1 = extractor(wav1).to(device)
            feat2 = extractor(wav2).to(device)
            T = min(feat1.shape[-1], feat2.shape[-1])
            x = torch.stack([feat1[:, :, :T], feat2[:, :, :T]], dim=1)
            logits = mimi_model.infer(x)
            probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
        else:
            h1, c1 = mimi_model.init_hidden(1, device)
            h2, c2 = mimi_model.init_hidden(1, device)
            all_logits = []
            for feat1, feat2 in extractor.stream(wav1, wav2):
                for t in range(feat1.shape[-1]):
                    logits, h1, c1, h2, c2 = mimi_model.infer_ar_step(
                        feat1[:, :, t].to(device), feat2[:, :, t].to(device),
                        h1, c1, h2, c2,
                    )
                    all_logits.append(logits)
            probs = torch.softmax(
                torch.stack(all_logits, dim=1)[0], dim=-1
            ).cpu().numpy()
    return probs[:, IDX_USER], probs[:, IDX_SYSTEM]


def _build_probs_file(
    conv_probs: list[tuple[str, np.ndarray, np.ndarray, float]],
    task: str,
) -> _ProbsFile:
    """Build a ProbsFile for the given task from inference probs.

    EOT: prob = 1 - p_user / 1 - p_system (high = turn ending), fires when prob > θ.
    INT: prob = p_user / p_system (high = taking floor), fires when prob > θ.
    Each array is trimmed to exactly frame_count(duration_s, FRAME_RATE) frames.
    """
    entries = []
    for conv_id, p_user, p_system, duration_s in conv_probs:
        n = frame_count(duration_s, FRAME_RATE)
        if task == "eot":
            p1, p2 = (1.0 - p_user[:n]).tolist(), (1.0 - p_system[:n]).tolist()
        else:
            p1, p2 = p_user[:n].tolist(), p_system[:n].tolist()
        entries.append(_ConversationProbs(
            conversation_id=conv_id,
            speaker_1=_SpeakerProbs(prob=p1),
            speaker_2=_SpeakerProbs(prob=p2),
        ))
    return _ProbsFile(
        schema_version=_PROBS_SCHEMA_VERSION,
        task=task,
        frame_rate_hz=FRAME_RATE,
        probs=entries,
    )


def _write_probs(conv_probs, path: Path, task: str) -> None:
    pf = _build_probs_file(conv_probs, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pf.model_dump_json(indent=2))
    print(f"Probs ({task}) → {path}", flush=True)


def _make_submission(
    thr_eot: float,
    thr_int: float,
    conv_probs: list[tuple[str, np.ndarray, np.ndarray, float]],
) -> Submission:
    """Build Submission using eval.sweep.commit_events (consistent with probs sweep).

    thr_eot: threshold in probs-eot space (1-p_user, high = EOT)
    thr_int: threshold in probs-int space (p_user, high = taking floor)
    """
    predictions = []
    for conv_id, p_user, p_system, duration_s in conv_probs:
        n = frame_count(duration_s, FRAME_RATE)
        predictions.append(ConversationPrediction(
            conversation_id=conv_id,
            speaker_1=SpeakerEvents(
                eot=         commit_events((1.0 - p_user[:n]).tolist(),   FRAME_RATE, thr_eot),
                interruption=commit_events(p_user[:n].tolist(),           FRAME_RATE, thr_int),
            ),
            speaker_2=SpeakerEvents(
                eot=         commit_events((1.0 - p_system[:n]).tolist(), FRAME_RATE, thr_eot),
                interruption=commit_events(p_system[:n].tolist(),         FRAME_RATE, thr_int),
            ),
        ))
    return Submission(schema_version=SCHEMA_VERSION, predictions=predictions)


def main(
    run_name: str = "pretrained",
    checkpoint: Path | None = None,
    dataset_source: str = DEV_DATASET,
    out: Path | None = None,
    threshold_eot: float | None = None,
    threshold_int: float | None = None,
    fast: bool = True,
    probs_only: bool = False,
) -> None:
    is_dev = (dataset_source == DEV_DATASET)
    split  = "dev" if is_dev else "test"

    if out is None and not probs_only:
        raise ValueError("--out is required (or --probs-only to write probs files only)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  run: {run_name}  split: {split}  mode: {'fast' if fast else 'streaming'}", flush=True)

    mimi_model = _load_checkpoint(run_name, device, checkpoint)
    extractor  = AudioFeatureExtractor(**AUDIO_DEFAULTS, device=device)

    dataset = resolve_dataset(source=dataset_source)
    conv_probs: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    ids = conversation_ids(dataset)
    with tqdm(ids, desc="inference", unit="conv") as bar:
        for conv_id in bar:
            bar.set_postfix(id=conv_id)
            conv = conversation(dataset, conv_id)
            t0 = time.time()
            p_user, p_system = _infer_conversation(conv, mimi_model, extractor, device, fast=fast)
            duration_s = conv.duration_s
            rt = (len(p_user) / FRAME_RATE) / (time.time() - t0)
            bar.set_postfix(id=conv_id, rt=f"{rt:.1f}x")
            conv_probs.append((conv_id, p_user, p_system, duration_s))

    # probs filenames: no prefix for "oto_d1f" (main submission), "{run_name}-" for others
    _pfx = "" if run_name == "oto_d1f" else f"{run_name}-"

    # ── probs-only: write probs files, then exit ─────────────────────────────
    if probs_only:
        _write_probs(conv_probs, _HERE / f"{_pfx}probs-eot.json", "eot")
        _write_probs(conv_probs, _HERE / f"{_pfx}probs-int.json", "int")
        return

    # ── submission mode ──────────────────────────────────────────────────────
    assert out is not None
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if is_dev:
        # Write probs files next to predictions output
        _write_probs(conv_probs, out.parent / f"{_pfx}probs-eot.json", "eot")
        _write_probs(conv_probs, out.parent / f"{_pfx}probs-int.json", "int")

        # Auto-pick operating point for each task via eval.sweep
        if threshold_eot is None:
            rows = _eval_sweep(_build_probs_file(conv_probs, "eot"), dataset)
            op   = _operating_point(rows)
            if op is not None:
                threshold_eot = op.theta
                print(f"Auto EOT threshold (recall={op.recall:.3f}, fp={op.fp_rate:.3f}): {threshold_eot}", flush=True)
            else:
                threshold_eot = CHECKPOINT_DEFAULTS.get(run_name, CHECKPOINT_DEFAULTS["pretrained"])[0]
                print(f"No EOT threshold within fp budget; using default {threshold_eot}", flush=True)

        if threshold_int is None:
            rows = _eval_sweep(_build_probs_file(conv_probs, "int"), dataset)
            op   = _operating_point(rows)
            if op is not None:
                threshold_int = op.theta
                print(f"Auto INT threshold (recall={op.recall:.3f}, fp={op.fp_rate:.3f}): {threshold_int}", flush=True)
            else:
                threshold_int = CHECKPOINT_DEFAULTS.get(run_name, CHECKPOINT_DEFAULTS["pretrained"])[1]
                print(f"No INT threshold within fp budget; using default {threshold_int}", flush=True)

    else:
        # Test: thresholds must be provided (read from dev metrics or passed explicitly)
        if threshold_eot is None or threshold_int is None:
            if run_name not in CHECKPOINT_DEFAULTS:
                raise ValueError(
                    f"No default thresholds for run '{run_name}'. "
                    f"Pass --threshold-eot and --threshold-int explicitly."
                )
            thr_e, thr_i = CHECKPOINT_DEFAULTS[run_name]
            if threshold_eot is None: threshold_eot = thr_e
            if threshold_int is None: threshold_int = thr_i

    print(f"Thresholds: eot={threshold_eot}  int={threshold_int}", flush=True)
    submission = _make_submission(threshold_eot, threshold_int, conv_probs)
    out.write_text(submission.model_dump_json(indent=2))
    print(f"\nWrote {out}", flush=True)

    if is_dev:
        scores = score_submission(submission, dataset)
        eot    = scores.task_eot
        int_   = scores.task_int
        elat   = eot.latency()
        ilat   = int_.latency()
        print(
            f"EOT  recall={eot.recall:.3f}  fp_rate={eot.fp_rate:.3f}  p50={elat.p50:.3f}s\n"
            f"INT  recall={int_.recall:.3f}  fp_rate={int_.fp_rate:.3f}  p50={ilat.p50:.3f}s",
            flush=True,
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name",      default="pretrained",
                        help="checkpoint name; sets fallback thresholds from CHECKPOINT_DEFAULTS")
    parser.add_argument("--checkpoint",    type=Path, default=None,
                        help="explicit local .pt path; overrides HF download")
    parser.add_argument("--dataset",       default=DEV_DATASET,
                        help="HF dataset repo id or local parquet dir (default: dev set)")
    parser.add_argument("--out",           type=Path, default=None,
                        help="write submission JSON here; required unless --probs-only")
    parser.add_argument("--threshold-eot", type=float, default=None,
                        help="EOT threshold in probs-eot space (auto-selected on dev if omitted)")
    parser.add_argument("--threshold-int", type=float, default=None,
                        help="INT threshold in probs-int space (auto-selected on dev if omitted)")
    parser.add_argument("--no-fast",       action="store_true",
                        help="use streaming AR inference instead of fast full-sequence")
    parser.add_argument("--probs-only",    action="store_true",
                        help="write probs-eot.json + probs-int.json only; no predictions")
    args = parser.parse_args()
    main(
        run_name=args.run_name,
        checkpoint=args.checkpoint,
        dataset_source=args.dataset,
        out=args.out,
        threshold_eot=args.threshold_eot,
        threshold_int=args.threshold_int,
        fast=not args.no_fast,
        probs_only=args.probs_only,
    )
