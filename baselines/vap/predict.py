#!/usr/bin/env python3
# Author: Sathvik Udupa (2026)
# Email:  udupa@fit.vutbr.cz
"""Voice Activity Projection (VAP) — prediction entry point for TurnBench.

Two-stream GPT-like transformer over CPC features. Takes stereo audio and
produces continuous floor-state probabilities at 50 Hz.

Score mapping:
  p_now[:, spk] = P(speaker spk holds floor in next 0-0.4 s)
  probs-eot  = 1 − p_now  (high ↔ speaker releasing floor)
  probs-int  =     p_now  (high ↔ speaker taking/holding floor)

Usage:
    bash baselines/vap/run.sh                      # default: dev + test, oto checkpoint
    bash baselines/vap/run.sh --dev                # dev only, oto checkpoint
    bash baselines/vap/run.sh --dev --pretrained   # dev, pretrained checkpoint
    bash baselines/vap/run.sh --test --swbd-oto    # test only, swbd_oto checkpoint
"""
from __future__ import annotations

import json
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
sys.path.insert(0, str(_HERE / "VoiceActivityProjection"))

from turnbench.data import (                                                      # noqa: E402
    DEV_DATASET,
    conversation,
    conversation_ids,
    resolve_dataset,
)
from turnbench.submission import (                                                # noqa: E402
    SCHEMA_VERSION,
    ConversationPrediction,
    SpeakerEvents,
    Submission,
)
from turnbench.sweep import (                                                     # noqa: E402
    ConversationProbs,
    ProbsFile,
    SpeakerProbs,
    commit_events,
    frame_count,
    operating_point,
    sweep,
)
from vap.model import VapGPT, VapConfig                                      # noqa: E402
from vap.utils import batch_to_device                                        # noqa: E402

SAMPLE_RATE = 16_000
FRAME_RATE  = 50.0

HF_REPO = "viks66/VAP_checkpoints"

CHECKPOINT_PATHS: dict[str, Path] = {
    "pretrained": _HERE / "VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt",
}

# Default (eot_thr, int_thr) per checkpoint in probs space.
# eot is (1 − p_now), so old eot_thr=0.10 maps to 1−0.10=0.90 here.
CHECKPOINT_DEFAULTS: dict[str, tuple[float, float]] = {
    "pretrained": (0.90, 0.65),
    "oto":        (0.90, 0.50),
    "swbd":       (0.90, 0.55),
    "swbd_oto":   (0.90, 0.50),
}


def _load_model(run_name: str, device: str, checkpoint: Path | None = None) -> VapGPT:
    if checkpoint is not None:
        ckpt_path = checkpoint
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    elif run_name in CHECKPOINT_PATHS:
        ckpt_path = CHECKPOINT_PATHS[run_name]
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        from huggingface_hub import hf_hub_download
        print(f"Downloading {run_name}.ckpt from {HF_REPO}…", flush=True)
        ckpt_path = Path(hf_hub_download(HF_REPO, f"{run_name}.ckpt"))
    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    conf  = VapConfig()
    model = VapGPT(conf)
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = {k.replace("net.", ""): v for k, v in sd["state_dict"].items() if "VAP.codebook" not in k}
    model.load_state_dict(sd)
    return model.to(device).eval()


def _step_extraction(waveform, model, device, context_time=20, step_time=5):
    """Chunked overlapping-window inference for long audio."""
    step_samples  = int(step_time  * model.sample_rate)
    chunk_samples = int((context_time + step_time) * model.sample_rate)
    step_frames   = int(step_time * model.frame_hz)

    n_samples       = waveform.shape[-1]
    expected_frames = round(n_samples / model.sample_rate * model.frame_hz)

    folds = waveform.unfold(-1, chunk_samples, step_samples).permute(2, 0, 1, 3)
    with torch.no_grad():
        out = model.probs(folds[0].to(device))
        for w in folds[1:]:
            o = model.probs(w.to(device))
            out["p_now"]    = torch.cat([out["p_now"],    o["p_now"][:,    -step_frames:]], dim=1)
            out["p_future"] = torch.cat([out["p_future"], o["p_future"][:, -step_frames:]], dim=1)
        processed = out["p_now"].shape[1]
        if processed < expected_frames:
            omitted = expected_frames - processed
            o = model.probs(waveform[..., -chunk_samples:].to(device))
            out["p_now"]    = torch.cat([out["p_now"],    o["p_now"][:,    -omitted:]], dim=1)
            out["p_future"] = torch.cat([out["p_future"], o["p_future"][:, -omitted:]], dim=1)
    return batch_to_device(out, "cpu")


def _load_wav(wav: np.ndarray, sr: int) -> torch.Tensor:
    t = torch.from_numpy(wav)
    if sr != SAMPLE_RATE:
        t = torchaudio.functional.resample(t, sr, SAMPLE_RATE)
    return t


def _infer_conversation(conv, model: VapGPT, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (p_spk1, p_spk2) each of shape (T,) at FRAME_RATE Hz."""
    wav1 = _load_wav(*conv.audio(1))
    wav2 = _load_wav(*conv.audio(2))
    waveform = torch.stack([wav1, wav2]).unsqueeze(0).to(device)
    duration = waveform.shape[-1] / SAMPLE_RATE
    if duration > 160:
        out = _step_extraction(waveform, model, device)
    else:
        with torch.no_grad():
            out = model.probs(waveform)
    p_now = out["p_now"][0].cpu().numpy()  # (T, 2)
    return p_now[:, 0], p_now[:, 1]


def _build_probs_file(conv_scores: list, task: str) -> ProbsFile:
    entries = []
    for conv_id, p1, p2, duration_s in conv_scores:
        n = frame_count(duration_s, FRAME_RATE)
        if task == "eot":
            prob1 = (1.0 - p1[:n]).tolist()
            prob2 = (1.0 - p2[:n]).tolist()
        else:
            prob1 = p1[:n].tolist()
            prob2 = p2[:n].tolist()
        entries.append(ConversationProbs(
            conversation_id=conv_id,
            speaker_1=SpeakerProbs(prob=prob1),
            speaker_2=SpeakerProbs(prob=prob2),
        ))
    return ProbsFile(schema_version=1, task=task, frame_rate_hz=FRAME_RATE, probs=entries)


def _make_submission(
    thr_eot: float,
    thr_int: float,
    conv_scores: list,
) -> Submission:
    predictions = []
    for conv_id, p1, p2, duration_s in conv_scores:
        n = frame_count(duration_s, FRAME_RATE)
        predictions.append(ConversationPrediction(
            conversation_id=conv_id,
            speaker_1=SpeakerEvents(
                eot=         commit_events((1.0 - p1[:n]).tolist(), FRAME_RATE, thr_eot),
                interruption=commit_events(p1[:n].tolist(),         FRAME_RATE, thr_int),
            ),
            speaker_2=SpeakerEvents(
                eot=         commit_events((1.0 - p2[:n]).tolist(), FRAME_RATE, thr_eot),
                interruption=commit_events(p2[:n].tolist(),         FRAME_RATE, thr_int),
            ),
        ))
    return Submission(schema_version=SCHEMA_VERSION, predictions=predictions)


def _print_sweep_table(rows, op, task: str) -> None:
    print(f"\n{task.upper()} sweep:")
    header = f"  {'theta':>5}  {'recall':>7}  {'fp_rate':>7}  {'lat_p50':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        marker = " ←" if op is not None and r.theta == op.theta else ""
        print(f"  {r.theta:5.2f}  {r.recall:7.3f}  {r.fp_rate:7.3f}  {r.lat_p50:7.0f}ms{marker}")


def main(
    run_name: str = "oto",
    checkpoint: Path | None = None,
    dataset_source: str = DEV_DATASET,
    out: Path | None = None,
    threshold_eot: float | None = None,
    threshold_int: float | None = None,
    probs_only: bool = False,
    probs_out_dir: Path | None = None,
) -> None:
    pfx = "" if run_name == "oto" else f"{run_name}-"
    is_dev = (dataset_source == DEV_DATASET)
    split  = "dev" if is_dev else "test"

    if not is_dev and not probs_only and threshold_eot is None:
        raise ValueError("--threshold-eot and --threshold-int required for non-dev split")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  run: {run_name}  split: {split}", flush=True)

    model = _load_model(run_name, device, checkpoint)

    dataset = resolve_dataset(source=dataset_source)
    conv_scores: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    ids = conversation_ids(dataset)
    with tqdm(ids, desc="inference", unit="conv") as bar:
        for conv_id in bar:
            bar.set_postfix(id=conv_id)
            conv = conversation(dataset, conv_id)
            t0 = time.time()
            p1, p2 = _infer_conversation(conv, model, device)
            duration_s = conv.duration_s
            rt = (len(p1) / FRAME_RATE) / (time.time() - t0)
            bar.set_postfix(id=conv_id, rt=f"{rt:.1f}x")
            conv_scores.append((conv_id, p1, p2, duration_s))

    # Write probs files (dev only, or --probs-only). --probs-out-dir redirects the
    # output (e.g. emitting TEST probs without clobbering the committed dev files).
    if is_dev or probs_only:
        probs_dir = probs_out_dir if probs_out_dir is not None else _HERE
        probs_dir.mkdir(parents=True, exist_ok=True)
        for task in ("eot", "int"):
            pf = _build_probs_file(conv_scores, task)
            path = probs_dir / f"{pfx}probs-{task}.json"
            path.write_text(pf.model_dump_json(indent=2))
            print(f"Wrote {path}", flush=True)

    if probs_only:
        return

    if is_dev:
        # Auto-select thresholds via sweep
        fb_eot, fb_int = CHECKPOINT_DEFAULTS.get(run_name, (0.5, 0.5))

        eot_pf  = _build_probs_file(conv_scores, "eot")
        eot_rows = sweep(eot_pf, dataset)
        op_eot  = operating_point(eot_rows)
        thr_eot = op_eot.theta if op_eot is not None else fb_eot

        int_pf  = _build_probs_file(conv_scores, "int")
        int_rows = sweep(int_pf, dataset)
        op_int  = operating_point(int_rows)
        thr_int = op_int.theta if op_int is not None else fb_int

        _print_sweep_table(eot_rows, op_eot, "eot")
        _print_sweep_table(int_rows, op_int, "int")
        print(f"\nOperating point: eot_thr={thr_eot}  int_thr={thr_int}", flush=True)

    else:
        thr_eot, thr_int = threshold_eot, threshold_int  # type: ignore[assignment]

    submission = _make_submission(thr_eot, thr_int, conv_scores)
    out_path = out if out is not None else _HERE / f"{pfx}predictions-{split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(submission.model_dump_json(indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name",       default="oto")
    parser.add_argument("--checkpoint",     type=Path, default=None)
    parser.add_argument("--dataset",        default=DEV_DATASET)
    parser.add_argument("--out",            type=Path, default=None)
    parser.add_argument("--threshold-eot",  type=float, default=None)
    parser.add_argument("--threshold-int",  type=float, default=None)
    parser.add_argument("--probs-only",     action="store_true")
    parser.add_argument("--probs-out-dir",  type=Path, default=None)
    args = parser.parse_args()
    main(
        run_name=args.run_name,
        checkpoint=args.checkpoint,
        dataset_source=args.dataset,
        out=args.out,
        threshold_eot=args.threshold_eot,
        threshold_int=args.threshold_int,
        probs_only=args.probs_only,
        probs_out_dir=args.probs_out_dir,
    )
