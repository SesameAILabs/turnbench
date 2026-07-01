# Author: Sathvik Udupa (2026)
# Email:  udupa@fit.vutbr.cz

"""Smart Turn v3 — prediction entry point for TurnBench.

VAD+accumulate+settling pipeline over Whisper-Tiny + linear classifier (ONNX).
Scores both channels in parallel threads at 12.5 Hz.

Score mapping:
  score      = P(turn complete)
  probs-eot  = score          (high ↔ speaker releasing floor)
  probs-int  = 1 − score      (high ↔ speaker actively talking / taking floor)
"""
from __future__ import annotations

import concurrent.futures
import math
import os
import sys
import time
import types
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_SMART_TURN = _HERE / "smart_turn"

# Save caller cwd before chdir — used to resolve relative --out paths.
_CALLER_CWD = Path(os.getcwd()).resolve()

# inference.py loads the ONNX model from a hardcoded relative path; chdir first.
os.chdir(_SMART_TURN)
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_SMART_TURN))

# record_and_predict.py imports pyaudio at module level; stub it out.
_pyaudio_stub = types.ModuleType("pyaudio")
_pyaudio_stub.paInt16 = 0
sys.modules.setdefault("pyaudio", _pyaudio_stub)

import inference as _inference_mod                                            # noqa: E402
from record_and_predict import SileroVAD, ensure_model                       # noqa: E402

import onnxruntime as _ort
_cuda      = "CUDAExecutionProvider" in _ort.get_available_providers()
_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if _cuda else ["CPUExecutionProvider"]
_model_file = "smart-turn-v3.1-gpu.onnx" if _cuda else "smart-turn-v3.1.onnx"
print(f"ONNX provider: {_providers[0]}  model: {_model_file}", flush=True, file=sys.stderr)
_inference_mod.session = _inference_mod.build_session(_model_file)

def predict_endpoint(audio_array):                                            # noqa: E402
    return _inference_mod.predict_endpoint(audio_array)

from eval.data import (                                                       # noqa: E402
    DEV_DATASET,
    conversation,
    conversation_ids,
    resolve_dataset,
)
from eval.submission import (                                                 # noqa: E402
    SCHEMA_VERSION,
    ConversationPrediction,
    SpeakerEvents,
    Submission,
)
from eval.sweep import (                                                      # noqa: E402
    ConversationProbs,
    ProbsFile,
    SpeakerProbs,
    commit_events,
    frame_count,
    operating_point,
    sweep,
)

SR            = 16_000
CHUNK         = 512
OUT_HZ        = 12.5
VAD_THRESH    = 0.5

_CHUNK_MS      = CHUNK / SR * 1000.0
_PRE_CHUNKS    = math.ceil(200.0  / _CHUNK_MS)
_STOP_CHUNKS   = math.ceil(1000.0 / _CHUNK_MS)
_SETTLE_CHUNKS = math.ceil(2000.0 / _CHUNK_MS)
_MAX_CHUNKS    = math.ceil(8000.0 / _CHUNK_MS)
_SETTLE_STRIDE = 8

# Default (eot_thr, int_thr) in probs space.
# score = P(turn complete); probs-eot = score, probs-int = 1 - score.
# Old eot_thr=0.05 maps directly; old int_thr=0.05 inverts to 1-0.05=0.95.
CHECKPOINT_DEFAULTS: dict[str, tuple[float, float]] = {
    "pretrained": (0.05, 0.95),
}


def _load_wav(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr != SR:
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
    return wav


def _score_channel(wav: np.ndarray, silero_path: str) -> np.ndarray:
    """VAD+accumulate+settling pipeline for a single channel. Returns scores at OUT_HZ."""
    n = int(len(wav) / SR * OUT_HZ) + 1
    scores = np.zeros(n, dtype=np.float32)
    vad = SileroVAD(silero_path)
    pre = deque(maxlen=_PRE_CHUNKS)
    seg = []
    active = False
    sil = extra = 0
    cur = 0.0

    for i in range(0, len(wav) - CHUNK + 1, CHUNK):
        c = wav[i:i + CHUNK].astype(np.float32)
        sp = vad.prob(c) > VAD_THRESH
        out_idx = int(i / SR * OUT_HZ)

        if not active:
            pre.append(c)
            if sp:
                cur = 0.0; seg = list(pre) + [c]; active = True; sil = 0; extra = 0
        elif extra == 0:
            seg.append(c)
            if len(seg) > _MAX_CHUNKS:
                seg.pop(0)
            sil = 0 if sp else sil + 1
            if sil >= _STOP_CHUNKS:
                extra = 1
                cur = predict_endpoint(np.concatenate(seg))["probability"]
        else:
            if sp:
                cur = 0.0; seg = [c]; active = True; sil = 0; extra = 0
            else:
                seg.append(c); extra += 1
                if extra % _SETTLE_STRIDE == 1:
                    cur = predict_endpoint(np.concatenate(seg))["probability"]
                if extra >= _SETTLE_CHUNKS:
                    active = False; sil = 0; extra = 0; pre.clear()
        scores[min(out_idx, n - 1)] = cur

    return scores


def _score_conversation(wav1: np.ndarray, wav2: np.ndarray, silero_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Run both channels in parallel threads (ONNX Runtime is thread-safe)."""
    T = max(len(wav1), len(wav2))
    wav1 = np.pad(wav1, (0, max(0, T - len(wav1))))
    wav2 = np.pad(wav2, (0, max(0, T - len(wav2))))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(_score_channel, wav1, silero_path)
        f2 = ex.submit(_score_channel, wav2, silero_path)
        return f1.result(), f2.result()


def _build_probs_file(conv_scores: list, task: str) -> ProbsFile:
    entries = []
    for conv_id, s1, s2, duration_s in conv_scores:
        n = frame_count(duration_s, OUT_HZ)
        if task == "eot":
            prob1 = s1[:n].tolist()
            prob2 = s2[:n].tolist()
        else:
            prob1 = (1.0 - s1[:n]).tolist()
            prob2 = (1.0 - s2[:n]).tolist()
        entries.append(ConversationProbs(
            conversation_id=conv_id,
            speaker_1=SpeakerProbs(prob=prob1),
            speaker_2=SpeakerProbs(prob=prob2),
        ))
    return ProbsFile(schema_version=1, task=task, frame_rate_hz=OUT_HZ, probs=entries)


def _make_submission(
    thr_eot: float,
    thr_int: float,
    conv_scores: list,
) -> Submission:
    predictions = []
    for conv_id, s1, s2, duration_s in conv_scores:
        n = frame_count(duration_s, OUT_HZ)
        predictions.append(ConversationPrediction(
            conversation_id=conv_id,
            speaker_1=SpeakerEvents(
                eot=         commit_events(s1[:n].tolist(),         OUT_HZ, thr_eot),
                interruption=commit_events((1.0 - s1[:n]).tolist(), OUT_HZ, thr_int),
            ),
            speaker_2=SpeakerEvents(
                eot=         commit_events(s2[:n].tolist(),         OUT_HZ, thr_eot),
                interruption=commit_events((1.0 - s2[:n]).tolist(), OUT_HZ, thr_int),
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


def _infer_one(
    conv_id: str, dataset, silero_path: str
) -> tuple[str, np.ndarray, np.ndarray, float]:
    conv = conversation(dataset, conv_id)
    wav1 = _load_wav(*conv.audio(1))
    wav2 = _load_wav(*conv.audio(2))
    s1, s2 = _score_conversation(wav1, wav2, silero_path)
    return conv_id, s1, s2, conv.duration_s


def main(
    run_name: str = "pretrained",
    dataset_source: str = DEV_DATASET,
    out: Path | None = None,
    threshold_eot: float | None = None,
    threshold_int: float | None = None,
    probs_only: bool = False,
    workers: int = 8,
    probs_out_dir: Path | None = None,
) -> None:
    is_dev = (dataset_source == DEV_DATASET)
    split  = "dev" if is_dev else "test"

    if not is_dev and not probs_only and threshold_eot is None:
        raise ValueError("--threshold-eot and --threshold-int required for non-dev split")

    silero_path = ensure_model(str(_SMART_TURN / "silero_vad.onnx"))
    print(f"Silero VAD: {silero_path}  run: {run_name}  split: {split}  workers: {workers}", flush=True)

    dataset = resolve_dataset(source=dataset_source)
    ids = conversation_ids(dataset)
    results: dict[str, tuple[str, np.ndarray, np.ndarray, float]] = {}

    with tqdm(total=len(ids), desc="inference", unit="conv") as bar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_infer_one, cid, dataset, silero_path): cid for cid in ids}
            for fut in concurrent.futures.as_completed(futs):
                result = fut.result()
                results[result[0]] = result
                bar.update(1)
                bar.set_postfix(id=result[0])

    conv_scores = [results[cid] for cid in ids]  # restore submission order

    # --probs-out-dir redirects the probs output (e.g. emitting TEST probs without
    # clobbering the committed dev files). Resolved against the CALLER's cwd —
    # this module chdirs into the smart_turn submodule at import.
    if is_dev or probs_only:
        if probs_out_dir is not None:
            probs_dir = probs_out_dir if probs_out_dir.is_absolute() else _CALLER_CWD / probs_out_dir
        else:
            probs_dir = _HERE
        probs_dir.mkdir(parents=True, exist_ok=True)
        for task in ("eot", "int"):
            pf = _build_probs_file(conv_scores, task)
            path = probs_dir / f"probs-{task}.json"
            path.write_text(pf.model_dump_json(indent=2))
            print(f"Wrote {path}", flush=True)

    if probs_only:
        return

    if is_dev:
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
    out_path = out if out is not None else _HERE / f"predictions-{split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(submission.model_dump_json(indent=2))
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name",      default="pretrained")
    parser.add_argument("--dataset",       default=DEV_DATASET)
    parser.add_argument("--out",           type=Path, default=None)
    parser.add_argument("--threshold-eot", type=float, default=None)
    parser.add_argument("--threshold-int", type=float, default=None)
    parser.add_argument("--probs-only",    action="store_true")
    parser.add_argument("--workers",       type=int, default=8)
    parser.add_argument("--probs-out-dir", type=Path, default=None)
    args = parser.parse_args()
    main(
        run_name=args.run_name,
        dataset_source=args.dataset,
        out=args.out,
        threshold_eot=args.threshold_eot,
        threshold_int=args.threshold_int,
        probs_only=args.probs_only,
        workers=args.workers,
        probs_out_dir=args.probs_out_dir,
    )
