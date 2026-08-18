#!/usr/bin/env python3
"""ESPnet Turn-Taking Prediction — INDIVIDUAL-CHANNEL variant.

Same model as the `espnet_turntaking` baseline (the "Talking Turns" frozen
Whisper-medium + 5-class head, Arora et al. ICLR 2025), but a different
inference strategy: instead of running once on the two-speaker mono MIX and
attributing events to speakers with energy VAD, this runs the model **twice —
once on each isolated speaker channel** — and reads each channel's native
turn-change / interruption probability directly as that speaker's events:

    eot_speaker_K          = commit(P_T  on speaker_K_audio)
    interruption_speaker_K = commit(P_I  on speaker_K_audio)

No mixing, no energy attribution: the per-channel run *is* the per-speaker
signal. Two inference passes per conversation.

What this trades (measured on dev, vs. the mix baseline):
  * End-of-turn is largely a SINGLE-speaker event ("this speaker stopped"),
    so per-channel detects it much better (recall 0.67 vs 0.37).
  * Interruption is RELATIONAL (needs the other speaker), and the model was
    trained on the mix, so per-channel is weaker there (recall 0.70 vs 0.76 at
    lower fp). See README.md for the full head-to-head.

Each per-frame score is turned into committed event times with an online
hysteresis + refractory detector (`_commit`); every emitted time is the
acoustic time the model has heard up to at that frame, so each commit depends
only on audio up to that time.

This baseline is self-contained. It needs the ESPnet model dir in
`ESPNET_TT_EXP` (Hugging Face `espnet/Turn_taking_prediction_SWBD`) and
`espnet2` importable. A per-frame cache (`--cache-dir`) lets repeated runs /
threshold sweeps skip the model.

    python -m baselines.espnet_turntaking_perchannel.predict                 # score on dev
    python -m baselines.espnet_turntaking_perchannel.predict --out preds.json
    python -m baselines.espnet_turntaking_perchannel.predict --sweep
    python -m baselines.espnet_turntaking_perchannel.predict --shard i --num-shards n
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# the repo root is two levels up (baselines/espnet_turntaking_perchannel/predict.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from turnbench.data import (  # noqa: E402
    DEV_DATASET,
    Conversation,
    conversation,
    conversation_ids,
    resolve_dataset,
)
from turnbench.submission import (  # noqa: E402
    SCHEMA_VERSION,
    ConversationPrediction,
    SpeakerEvents,
    Submission,
)

SR = 16000
START_CHUNK = 3200      # 0.2 s — first prediction offset
HOP = 640               # 40 ms — prediction stride (25 Hz)
WIN = 480000            # 30 s — Whisper-medium max context
BATCH = 8
I_IDX, T_IDX = 2, 4     # class indices in the model output [C, NA, I, BC, T]

# Commitment thresholds (hysteresis + refractory), tuned on dev with --sweep;
# override via env. EOT and interruption use separate operating points.
EOT_TAU_HIGH = float(os.environ.get("PC_EOT_TAU_HIGH", "0.20"))
EOT_TAU_LOW = float(os.environ.get("PC_EOT_TAU_LOW", "0.08"))
INT_TAU_HIGH = float(os.environ.get("PC_INT_TAU_HIGH", "0.15"))
INT_TAU_LOW = float(os.environ.get("PC_INT_TAU_LOW", "0.06"))
REFRACTORY_S = float(os.environ.get("PC_REFRACTORY_S", "2.0"))

_DEFAULT_CACHE = (
    Path(__file__).resolve().parent.parent.parent
    / "predictions" / "espnet_turntaking_perchannel" / "cache"
)

_model = None
_device = None


def _load_model():
    global _model, _device
    if _model is not None:
        return _model
    import torch
    from espnet2.tasks.slu import SLUTask

    if os.environ.get("TT_TF32", "0") == "1":  # 2.3x on H100 tensor cores, ~5e-3 prob delta
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    exp = os.environ.get("ESPNET_TT_EXP")
    if not exp:
        raise RuntimeError(
            "Set ESPNET_TT_EXP to the espnet experiment directory containing "
            "config.yaml and valid.loss.ave.pth (model: Hugging Face "
            "espnet/Turn_taking_prediction_SWBD). espnet2 must be importable."
        )
    exp = Path(exp)
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    m, _ = SLUTask.build_model_from_file(
        str(exp / "config.yaml"), str(exp / "valid.loss.ave.pth"), _device
    )
    m.eval()
    _model = m
    return m


def _to_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SR)
    return audio.astype("float32")


def _probs_for_signal(sig: np.ndarray) -> np.ndarray:
    """Sliding-window model over one mono signal -> [T,5] probs [C,NA,I,BC,T]."""
    import torch

    m = _load_model()
    with torch.no_grad():
        sp = torch.from_numpy(sig).to(_device)
        n = (len(sig) - START_CHUNK) // HOP
        if n <= 0:
            return np.zeros((0, 5), dtype=np.float32)
        out = np.empty((n, 5), dtype=np.float32)

        def heads(enc, olens):
            feats = m.transform_mean(m.act_fn(enc))
            last = torch.stack(
                [feats[k, olens[k] - 1] for k in range(feats.shape[0])]
            )
            return torch.softmax(m.transform_linear(last), dim=-1)

        i = 0
        while i < n:                               # growing windows (<30 s)
            end = (i + 1) * HOP + START_CHUNK
            if end - max(0, end - WIN) >= WIN:
                break
            w = sp[max(0, end - WIN):end].unsqueeze(0)
            L = torch.tensor([w.shape[1]], device=_device)
            enc, ol = m.encode(w, L)
            out[i] = heads(enc, ol)[0].cpu().numpy()
            i += 1
        while i < n:                               # steady-state windows, batched
            js = list(range(i, min(i + BATCH, n)))
            wins = [
                sp[(j + 1) * HOP + START_CHUNK - WIN:(j + 1) * HOP + START_CHUNK]
                for j in js
            ]
            L = torch.full((len(js),), WIN, dtype=torch.long, device=_device)
            enc, ol = m.encode(torch.stack(wins), L)
            out[js[0]:js[-1] + 1] = heads(enc, ol).cpu().numpy()
            i = js[-1] + 1
    return out


def _frame_time(n: int) -> np.ndarray:
    """Acoustic time (s) the model has heard up to at each prediction frame."""
    return (START_CHUNK + (np.arange(n) + 1) * HOP) / SR


def _commit(score: np.ndarray, times: np.ndarray, tau_high: float,
            tau_low: float, t_max: float | None = None,
            refractory_s: float = 0.0) -> list[float]:
    """Online hysteresis detector -> sorted, strictly-increasing commit times.

    Emit `times[i]` the first frame `score[i] >= tau_high` while armed, then
    stay disarmed until `score[i] <= tau_low`; consecutive commits are
    >= refractory_s apart. Causal: uses only frame i."""
    out: list[float] = []
    armed = True
    last = -1e18
    for i in range(len(score)):
        s = score[i]
        if armed and s >= tau_high:
            t = float(times[i])
            if (t_max is None or t < t_max) and (t - last >= refractory_s):
                out.append(t)
                last = t
            armed = False
        elif not armed and s <= tau_low:
            armed = True
    return out


def channel_probs(conv: Conversation, cache_dir: Path):
    """(probs1, probs2): the model run on each isolated channel -> [T,5] each."""
    cpath = cache_dir / f"{conv.conversation_id}.npz"
    if cpath.exists():
        d = np.load(cpath)
        return d["probs1"], d["probs2"]
    probs1 = _probs_for_signal(_to_16k(*conv.audio(1)))
    probs2 = _probs_for_signal(_to_16k(*conv.audio(2)))
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cpath, probs1=probs1, probs2=probs2)
    return probs1, probs2


def _events_for_channel(probs, duration, eth, etl, ith, itl, refr) -> SpeakerEvents:
    t = _frame_time(len(probs))
    return SpeakerEvents(
        eot=_commit(probs[:, T_IDX], t, eth, etl, duration, refr),
        interruption=_commit(probs[:, I_IDX], t, ith, itl, duration, refr),
    )


def predict(conv: Conversation, cache_dir: Path, *,
            eth=None, etl=None, ith=None, itl=None, refr=None) -> ConversationPrediction:
    eth = EOT_TAU_HIGH if eth is None else eth
    etl = EOT_TAU_LOW if etl is None else etl
    ith = INT_TAU_HIGH if ith is None else ith
    itl = INT_TAU_LOW if itl is None else itl
    refr = REFRACTORY_S if refr is None else refr
    p1, p2 = channel_probs(conv, cache_dir)
    return ConversationPrediction(
        conversation_id=conv.conversation_id,
        speaker_1=_events_for_channel(p1, conv.duration_s, eth, etl, ith, itl, refr),
        speaker_2=_events_for_channel(p2, conv.duration_s, eth, etl, ith, itl, refr),
    )


def _sweep(dataset, cache_dir, ids):
    """Tune per-track thresholds on the official scorer (no model re-run)."""
    from turnbench.score import score_submission
    cached = [(conversation(dataset, i)) for i in ids]
    for c in cached:
        channel_probs(c, cache_dir)  # ensure cached

    def build(track, tau_high):
        tau_low = round(tau_high * 0.4, 4)
        preds = []
        for c in cached:
            kw = dict(eth=EOT_TAU_HIGH, etl=EOT_TAU_LOW, ith=INT_TAU_HIGH, itl=INT_TAU_LOW)
            if track == "eot":
                kw.update(eth=tau_high, etl=tau_low)
            else:
                kw.update(ith=tau_high, itl=tau_low)
            preds.append(predict(c, cache_dir, refr=REFRACTORY_S, **kw))
        return Submission(schema_version=SCHEMA_VERSION, predictions=preds)

    for track in ("eot", "interruption"):
        print(f"== sweep {track} (per-channel) ==")
        print(f"{'tau_hi':>7} {'recall':>7} {'fp_rate':>8} {'lat p50':>8}")
        for th in np.arange(0.05, 0.55, 0.05):
            th = round(float(th), 3)
            agg = score_submission(build(track, th), dataset)
            s = agg.task_eot if track == "eot" else agg.task_int
            print(f"{th:>7.2f} {s.recall:>7.3f} {s.fp_rate:>8.3f} {s.latency().p50:>8.0f}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEV_DATASET,
                        help="HF dataset repo id, or a local directory of parquet shards")
    parser.add_argument("--out", default=None,
                        help="write a predictions JSON here instead of scoring")
    parser.add_argument("--cache-dir", default=str(_DEFAULT_CACHE))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--sweep", action="store_true",
                        help="tune commit thresholds over the cache (no model re-run)")
    args = parser.parse_args()
    cache_dir = Path(args.cache_dir)

    dataset = resolve_dataset(source=args.dataset)
    ids = conversation_ids(dataset)

    if args.num_shards > 1:  # cache-only, parallel
        mine = ids[args.shard::args.num_shards]
        for i in mine:
            channel_probs(conversation(dataset, i), cache_dir)
            print(f"shard {args.shard}/{args.num_shards}: cached {i}", file=sys.stderr)
        print(f"shard {args.shard}/{args.num_shards}: done ({len(mine)} convs)", file=sys.stderr)
        return 0

    if args.sweep:
        _sweep(dataset, cache_dir, ids)
        return 0

    submission = Submission(
        schema_version=SCHEMA_VERSION,
        predictions=[predict(conversation(dataset, i), cache_dir) for i in ids],
    )

    if args.out is not None:
        Path(args.out).write_text(submission.model_dump_json(indent=2), encoding="utf-8")
        print(f"Wrote {len(submission.predictions)} predictions to {args.out}", file=sys.stderr)
        return 0

    from turnbench.score import score_submission, task_cells
    scores = score_submission(submission, dataset)
    print(f"espnet_turntaking_perchannel — {len(submission.predictions)} conversations")
    for name, s in (("EOT", scores.task_eot), ("INT", scores.task_int)):
        recall, fp_rate, latency = task_cells(s)
        print(f"  {name}: recall={recall} fp_rate={fp_rate} latency_ms={latency}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
