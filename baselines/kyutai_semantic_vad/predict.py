# Author: Sathvik Udupa (2026)
# Email:  udupa@fit.vutbr.cz

"""Kyutai Semantic VAD — dev scoring or submission JSON for the TURN benchmark.

Uses Kyutai STT-1B's VAD head (vad_heads[2]) for continuous EOT scores at ~12.5Hz.
Both speakers run in a single batched stream (batch_size=2) — one forward pass per
frame for both channels.

Score direction: P(turn ending); floor held when score < threshold.

probs-eot.json:  prob = score        (P turn ending);   commit when prob > θ_eot
probs-int.json:  prob = 1 - score    (P taking floor);  commit when prob > θ_int

Thresholds in CHECKPOINT_DEFAULTS are in probs space. On dev, thresholds are
auto-selected by eval.sweep.operating_point.

Parallel inference (N shards):
    # Each shard writes probs for 1/N conversations:
    python -m baselines.kyutai_semantic_vad.predict \\
        --shard K N --probs-only --out-dir partial/shard_K

    # After all shards finish, merge into final probs:
    python -m baselines.kyutai_semantic_vad.predict --merge-partials partial/

    # Generate dev predictions from merged probs (no inference):
    python -m baselines.kyutai_semantic_vad.predict --from-probs \\
        --out baselines/kyutai_semantic_vad/predictions-dev.json
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as TAF
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

import moshi.models                     # noqa: E402
import moshi.models.loaders as loaders  # noqa: E402

from eval.data import (                                                       # noqa: E402
    DEV_DATASET,
    Dataset,
    conversation,
    conversation_ids,
    resolve_dataset,
)
from eval.score import score_submission                                        # noqa: E402
from eval.submission import (                                                 # noqa: E402
    SCHEMA_VERSION,
    ConversationPrediction,
    SpeakerEvents,
    Submission,
)
from eval.sweep import (                                                      # noqa: E402
    ConversationProbs as _ConversationProbs,
    ProbsFile as _ProbsFile,
    SCHEMA_VERSION as _PROBS_SCHEMA_VERSION,
    SpeakerProbs as _SpeakerProbs,
    commit_events,
    frame_count,
    load_probs as _load_probs,
    operating_point as _operating_point,
    sweep as _eval_sweep,
)

HF_REPO = "kyutai/stt-1b-en_fr-candle"

# (eot_thr, int_thr) in probs space:
#   eot: prob = score,       fire when score > eot_thr
#   int: prob = 1 - score,   fire when (1-score) > int_thr
CHECKPOINT_DEFAULTS: dict[str, tuple[float, float]] = {
    "pretrained": (0.05, 0.65),
}

_mimi             = None
_lm_gen           = None
_silence_prefix_s = None
_audio_delay_s    = None
_frame_rate       = None


def _load_models(device: str) -> None:
    global _mimi, _lm_gen, _silence_prefix_s, _audio_delay_s, _frame_rate
    if _mimi is not None:
        return
    print("Loading Kyutai STT model...", flush=True)
    info              = loaders.CheckpointInfo.from_hf_repo(HF_REPO)
    _mimi             = info.get_mimi(device=device)
    lm                = info.get_moshi(device=device, dtype=torch.bfloat16)
    _silence_prefix_s = info.stt_config.get("audio_silence_prefix_seconds", 1.0)
    _audio_delay_s    = info.stt_config.get("audio_delay_seconds", 5.0)
    _lm_gen           = moshi.models.LMGen(lm, temp=0, temp_text=0.0)
    _frame_rate       = float(_mimi.frame_rate)
    print(f"Model loaded. frame_rate={_frame_rate} Hz", flush=True)


def _infer_conversation(conv, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (vad1, vad2) at _frame_rate Hz."""
    wav1, sr1 = conv.audio(1)
    wav2, sr2 = conv.audio(2)

    a1 = torch.from_numpy(wav1).float().unsqueeze(0).to(device)
    a2 = torch.from_numpy(wav2).float().unsqueeze(0).to(device)
    a1 = TAF.resample(a1, sr1, _mimi.sample_rate)
    a2 = TAF.resample(a2, sr2, _mimi.sample_rate)

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


def _build_probs_file(
    conv_scores: list[tuple[str, np.ndarray, np.ndarray, float]],
    task: str,
) -> _ProbsFile:
    """Build a ProbsFile for the given task from inference scores.

    EOT: prob = score (P turn ending), fires when score > θ.
    INT: prob = 1 - score (P taking floor), fires when (1-score) > θ.
    Each speaker array is trimmed to exactly frame_count(duration_s, frame_rate) frames.
    """
    entries = []
    for conv_id, s1, s2, duration_s in conv_scores:
        n = frame_count(duration_s, _frame_rate)
        if task == "eot":
            p1, p2 = s1[:n].tolist(), s2[:n].tolist()
        else:
            p1, p2 = (1.0 - s1[:n]).tolist(), (1.0 - s2[:n]).tolist()
        entries.append(_ConversationProbs(
            conversation_id=conv_id,
            speaker_1=_SpeakerProbs(prob=p1),
            speaker_2=_SpeakerProbs(prob=p2),
        ))
    return _ProbsFile(
        schema_version=_PROBS_SCHEMA_VERSION,
        task=task,
        frame_rate_hz=_frame_rate,
        probs=entries,
    )


def _write_probs(conv_scores, path: Path, task: str) -> None:
    pf = _build_probs_file(conv_scores, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pf.model_dump_json(indent=2))
    print(f"Probs ({task}) → {path}", flush=True)


def _make_submission(
    thr_eot: float,
    thr_int: float,
    conv_scores: list[tuple[str, np.ndarray, np.ndarray, float]],
) -> Submission:
    """Build Submission using eval.sweep.commit_events (consistent with probs sweep).

    thr_eot: threshold in probs-eot space (score, P turn ending)
    thr_int: threshold in probs-int space (1 - score, P taking floor)
    """
    predictions = []
    for conv_id, s1, s2, duration_s in conv_scores:
        n = frame_count(duration_s, _frame_rate)
        predictions.append(ConversationPrediction(
            conversation_id=conv_id,
            speaker_1=SpeakerEvents(
                eot=         commit_events(s1[:n].tolist(),           _frame_rate, thr_eot),
                interruption=commit_events((1.0 - s1[:n]).tolist(),   _frame_rate, thr_int),
            ),
            speaker_2=SpeakerEvents(
                eot=         commit_events(s2[:n].tolist(),           _frame_rate, thr_eot),
                interruption=commit_events((1.0 - s2[:n]).tolist(),   _frame_rate, thr_int),
            ),
        ))
    return Submission(schema_version=SCHEMA_VERSION, predictions=predictions)


def merge_partials(partial_dir: Path, out_dir: Path, dataset_source: str = DEV_DATASET) -> None:
    """Merge shard probs from partial_dir/shard_*/probs-{task}.json → out_dir/probs-{task}.json.

    Shards are round-robin (ids[K::N]), so merge restores original dataset order.
    """
    shard_dirs = sorted(
        partial_dir.glob("shard_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not shard_dirs:
        raise FileNotFoundError(f"No shard_* dirs in {partial_dir}")

    dataset = resolve_dataset(source=dataset_source)
    id_order = {cid: i for i, cid in enumerate(conversation_ids(dataset))}

    out_dir.mkdir(parents=True, exist_ok=True)
    for task in ("eot", "int"):
        all_entries: list[_ConversationProbs] = []
        frame_rate_hz: float | None = None
        for sd in shard_dirs:
            pf = _load_probs(sd / f"probs-{task}.json")
            if frame_rate_hz is None:
                frame_rate_hz = pf.frame_rate_hz
            elif pf.frame_rate_hz != frame_rate_hz:
                raise ValueError(f"frame_rate_hz mismatch across shards: {frame_rate_hz} vs {pf.frame_rate_hz}")
            all_entries.extend(pf.probs)

        all_entries.sort(key=lambda e: id_order[e.conversation_id])
        result = _ProbsFile(
            schema_version=_PROBS_SCHEMA_VERSION,
            task=task,
            frame_rate_hz=frame_rate_hz,
            probs=all_entries,
        )
        out_path = out_dir / f"probs-{task}.json"
        out_path.write_text(result.model_dump_json(indent=2))
        print(f"Merged {len(all_entries)} conversations → {out_path}", flush=True)


def _score_and_write(
    submission: Submission,
    out: Path,
    dataset,
    thr_eot: float,
    thr_int: float,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(submission.model_dump_json(indent=2))
    print(f"Wrote {out}", flush=True)

    scores = score_submission(submission, dataset)
    eot    = scores.task_eot
    int_   = scores.task_int
    elat   = eot.latency()
    ilat   = int_.latency()
    metrics = {
        "eot_thr": thr_eot, "int_thr": thr_int,
        "eot_recall":  round(eot.recall, 6),  "eot_fp_rate": round(eot.fp_rate, 6),
        "eot_lat_p10": round(elat.p10, 3), "eot_lat_p50": round(elat.p50, 3), "eot_lat_p90": round(elat.p90, 3),
        "eot_tp": eot.tp, "eot_fn": eot.fn, "eot_fp": eot.fp, "eot_tn": eot.tn,
        "int_recall":  round(int_.recall, 6),  "int_fp_rate": round(int_.fp_rate, 6),
        "int_lat_p50": round(ilat.p50, 3),
        "int_tp": int_.tp, "int_fn": int_.fn, "int_fp": int_.fp, "int_tn": int_.tn,
    }
    mpath = out.parent / (out.stem + "-metrics.json")
    mpath.write_text(json.dumps(metrics, indent=2))
    print(
        f"EOT  recall={eot.recall:.3f}  fp_rate={eot.fp_rate:.3f}  p50={elat.p50:.3f}s\n"
        f"INT  recall={int_.recall:.3f}  fp_rate={int_.fp_rate:.3f}  p50={ilat.p50:.3f}s\n"
        f"Metrics → {mpath}",
        flush=True,
    )


def from_probs(
    run_name: str,
    out: Path,
    probs_dir: Path = _HERE,
    dataset_source: str = DEV_DATASET,
    threshold_eot: float | None = None,
    threshold_int: float | None = None,
) -> None:
    """Generate predictions from already-written probs files — no inference.

    Dev: auto-sweeps thresholds + scores. Test: requires explicit thresholds.
    """
    is_dev = (dataset_source == DEV_DATASET)
    eot_pf = _load_probs(probs_dir / "probs-eot.json")
    int_pf = _load_probs(probs_dir / "probs-int.json")
    dataset = resolve_dataset(source=dataset_source)
    fb_eot, fb_int = CHECKPOINT_DEFAULTS[run_name]

    if threshold_eot is None:
        if not is_dev:
            raise ValueError("--threshold-eot required for non-dev dataset")
        op = _operating_point(_eval_sweep(eot_pf, resolve_dataset(source=DEV_DATASET)))
        threshold_eot = op.theta if op else fb_eot
        print(f"EOT threshold: {threshold_eot}" + ("" if op else " (fallback)"), flush=True)

    if threshold_int is None:
        if not is_dev:
            raise ValueError("--threshold-int required for non-dev dataset")
        op = _operating_point(_eval_sweep(int_pf, resolve_dataset(source=DEV_DATASET)))
        threshold_int = op.theta if op else fb_int
        print(f"INT threshold: {threshold_int}" + ("" if op else " (fallback)"), flush=True)

    int_by_id = {e.conversation_id: e for e in int_pf.probs}
    fps = eot_pf.frame_rate_hz
    predictions = []
    for entry in eot_pf.probs:
        ie = int_by_id[entry.conversation_id]
        predictions.append(ConversationPrediction(
            conversation_id=entry.conversation_id,
            speaker_1=SpeakerEvents(
                eot=         commit_events(entry.speaker_1.prob, fps, threshold_eot),
                interruption=commit_events(ie.speaker_1.prob,    fps, threshold_int),
            ),
            speaker_2=SpeakerEvents(
                eot=         commit_events(entry.speaker_2.prob, fps, threshold_eot),
                interruption=commit_events(ie.speaker_2.prob,    fps, threshold_int),
            ),
        ))
    submission = Submission(schema_version=SCHEMA_VERSION, predictions=predictions)
    if is_dev:
        _score_and_write(submission, out, dataset, threshold_eot, threshold_int)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(submission.model_dump_json(indent=2))
        print(f"Wrote {out}", flush=True)


def main(
    run_name: str = "pretrained",
    dataset_source: str = DEV_DATASET,
    out: Path | None = None,
    threshold_eot: float | None = None,
    threshold_int: float | None = None,
    probs_only: bool = False,
    shard: tuple[int, int] | None = None,
    out_dir: Path = _HERE,
) -> None:
    is_dev = (dataset_source == DEV_DATASET)
    split  = "dev" if is_dev else "test"

    if out is None and not is_dev and not probs_only:
        raise ValueError("--out is required for non-dev datasets (nothing to score against)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    shard_tag = f"  shard={shard[0]}/{shard[1]}" if shard else ""
    print(f"Device: {device}  run: {run_name}  split: {split}{shard_tag}", flush=True)
    _load_models(device)

    dataset = resolve_dataset(source=dataset_source)
    ids = conversation_ids(dataset)
    if shard is not None:
        k, n = shard
        ids = ids[k::n]   # round-robin: shard k gets ids[k], ids[k+n], ids[k+2n], ...
        print(f"Shard {k}/{n}: {len(ids)} conversations", flush=True)

    conv_scores: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    with tqdm(ids, desc="inference", unit="conv") as bar:
        for conv_id in bar:
            bar.set_postfix(id=conv_id)
            conv = conversation(dataset, conv_id)
            t0 = time.time()
            s1, s2 = _infer_conversation(conv, device)
            duration_s = conv.duration_s
            rt = (len(s1) / _frame_rate) / (time.time() - t0)
            bar.set_postfix(id=conv_id, rt=f"{rt:.1f}x")
            conv_scores.append((conv_id, s1, s2, duration_s))

    # ── probs-only: write probs-eot.json + probs-int.json, then exit ────────
    if probs_only:
        _write_probs(conv_scores, out_dir / "probs-eot.json", "eot")
        _write_probs(conv_scores, out_dir / "probs-int.json", "int")
        return

    # ── submission mode ──────────────────────────────────────────────────────
    assert out is not None
    out = Path(out)

    if is_dev:
        _write_probs(conv_scores, out.parent / "probs-eot.json", "eot")
        _write_probs(conv_scores, out.parent / "probs-int.json", "int")

        if threshold_eot is None:
            rows = _eval_sweep(_build_probs_file(conv_scores, "eot"), dataset)
            op   = _operating_point(rows)
            if op is not None:
                threshold_eot = op.theta
                print(f"Auto EOT threshold (recall={op.recall:.3f}, fp={op.fp_rate:.3f}): {threshold_eot}", flush=True)
            else:
                threshold_eot = CHECKPOINT_DEFAULTS[run_name][0]
                print(f"No EOT op in budget; using default {threshold_eot}", flush=True)

        if threshold_int is None:
            rows = _eval_sweep(_build_probs_file(conv_scores, "int"), dataset)
            op   = _operating_point(rows)
            if op is not None:
                threshold_int = op.theta
                print(f"Auto INT threshold (recall={op.recall:.3f}, fp={op.fp_rate:.3f}): {threshold_int}", flush=True)
            else:
                threshold_int = CHECKPOINT_DEFAULTS[run_name][1]
                print(f"No INT op in budget; using default {threshold_int}", flush=True)

    else:
        if threshold_eot is None or threshold_int is None:
            thr_e, thr_i = CHECKPOINT_DEFAULTS[run_name]
            if threshold_eot is None: threshold_eot = thr_e
            if threshold_int is None: threshold_int = thr_i

    print(f"Thresholds: eot={threshold_eot}  int={threshold_int}", flush=True)
    submission = _make_submission(threshold_eot, threshold_int, conv_scores)

    if is_dev:
        _score_and_write(submission, out, dataset, threshold_eot, threshold_int)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(submission.model_dump_json(indent=2))
        print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name",        default="pretrained")
    parser.add_argument("--dataset",         default=DEV_DATASET)
    parser.add_argument("--out",             type=Path, default=None)
    parser.add_argument("--threshold-eot",   type=float, default=None)
    parser.add_argument("--threshold-int",   type=float, default=None)
    parser.add_argument("--probs-only",      action="store_true",
                        help="write probs-eot/int.json only; no predictions")
    parser.add_argument("--shard",           type=int, nargs=2, metavar=("K", "N"), default=None,
                        help="process shard K of N (0-indexed, round-robin)")
    parser.add_argument("--out-dir",         type=Path, default=_HERE,
                        help="directory for probs output (default: baselines/kyutai_semantic_vad)")
    parser.add_argument("--merge-partials",  type=Path, default=None, metavar="PARTIAL_DIR",
                        help="merge shard_*/probs-*.json from PARTIAL_DIR → --out-dir; no inference")
    parser.add_argument("--from-probs",      action="store_true",
                        help="generate dev predictions from existing probs files; no inference")
    args = parser.parse_args()

    if args.merge_partials is not None:
        merge_partials(args.merge_partials, args.out_dir, dataset_source=args.dataset)
    elif args.from_probs:
        if args.out is None:
            is_dev = (args.dataset == DEV_DATASET)
            args.out = args.out_dir / ("predictions-dev.json" if is_dev else "predictions-test.json")
        from_probs(
            args.run_name, args.out,
            probs_dir=args.out_dir,
            dataset_source=args.dataset,
            threshold_eot=args.threshold_eot,
            threshold_int=args.threshold_int,
        )
    else:
        main(
            run_name=args.run_name,
            dataset_source=args.dataset,
            out=args.out,
            threshold_eot=args.threshold_eot,
            threshold_int=args.threshold_int,
            probs_only=args.probs_only,
            shard=tuple(args.shard) if args.shard else None,
            out_dir=args.out_dir,
        )
