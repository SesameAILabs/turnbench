#!/usr/bin/env python3
"""ESPnet Turn-Taking Prediction (Switchboard) — discrete TurnBench baseline.

The judge model from *"Talking Turns"* (Arora et al., ICLR 2025): a frozen
Whisper-medium encoder (~306 M) + small 5-class head, emitting per-40 ms (25 Hz)
probabilities over {Continuation(C), Silence(NA), Interruption(I),
Backchannel(BC), Turn-change(T)}.

The model is SINGLE-STREAM and was trained on Switchboard's two-speaker mono
**mix**; it has no per-speaker output. The dataset ships the two isolated
channels only, so we reconstruct the mix as `speaker_1 + speaker_2` (verified
bit-equivalent to the original combined recording), run the model once on it,
and attribute its native turn-change / interruption signals to a speaker with
cheap energy VAD on the two channels:

    eot for speaker K          <- P_T, attributed to the floor-holder K
    interruption for speaker K  <- P_I, attributed to the barge-in speaker K

Each per-frame score is turned into committed event times by an online
hysteresis detector (`_commit`): a time is emitted when the score rises to
`tau_high`, with no re-fire until it falls to `tau_low` and a `refractory_s`
minimum gap. The emitted time is the acoustic time the model has heard up to at
that frame, so each commit depends only on audio up to that time.

This is the standard baseline shape (cf. `baselines/rms_vad/predict.py`): a
self-contained predictor returning one `ConversationPrediction` per
conversation, scored by `eval.score`.

    python -m baselines.espnet_turntaking.predict                 # score on dev
    python -m baselines.espnet_turntaking.predict --out preds.json # write a JSON
    python -m baselines.espnet_turntaking.predict --dataset <hf repo|local dir>

Needs the ESPnet model dir in `ESPNET_TT_EXP` (Hugging Face
`espnet/Turn_taking_prediction_SWBD`) and `espnet2` importable. A per-frame
cache (`--cache-dir`) lets repeated runs / threshold sweeps skip the model.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# eval/ is two levels up (baselines/espnet_turntaking/predict.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.data import (  # noqa: E402
    DEV_DATASET,
    Conversation,
    conversation,
    conversation_ids,
    resolve_dataset,
)
from eval.submission import (  # noqa: E402
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
W_FRAMES = 12           # ~0.5 s look-back for hold/onset attribution

# Commitment thresholds (hysteresis + refractory), tuned on the dev split with
# `sweep.py` against the official scorer; override via env. EOT and interruption
# use separate operating points because their score scales differ.
EOT_TAU_HIGH = float(os.environ.get("TT_EOT_TAU_HIGH", "0.12"))
EOT_TAU_LOW = float(os.environ.get("TT_EOT_TAU_LOW", "0.048"))
EOT_REFRACTORY_S = float(os.environ.get("TT_EOT_REFRACTORY_S", "2.0"))
INT_TAU_HIGH = float(os.environ.get("TT_INT_TAU_HIGH", "0.14"))
INT_TAU_LOW = float(os.environ.get("TT_INT_TAU_LOW", "0.056"))
INT_REFRACTORY_S = float(os.environ.get("TT_INT_REFRACTORY_S", "2.0"))

_DEFAULT_CACHE = (
    Path(__file__).resolve().parent.parent.parent
    / "predictions" / "espnet_turntaking_dev" / "cache"
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


def _frame_energy(sig: np.ndarray, n_frames: int) -> np.ndarray:
    """Per-frame RMS aligned to the prediction frames: frame i covers samples
    [START_CHUNK + i*HOP, START_CHUNK + (i+1)*HOP)."""
    e = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = START_CHUNK + i * HOP
        seg = sig[s:s + HOP]
        if len(seg):
            e[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12)
    return e


def _smooth_prev(x: np.ndarray, w: int) -> np.ndarray:
    """Mean of x over the previous w frames (causal, excludes current)."""
    out = np.zeros_like(x)
    csum = np.concatenate([[0.0], np.cumsum(x)])
    for i in range(len(x)):
        lo = max(0, i - w)
        out[i] = (csum[i] - csum[lo]) / max(1, i - lo)
    return out


def _attribute(probs_mix: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> dict:
    """Four per-speaker per-frame score channels from mix probs + energy VAD."""
    C, NA, I, BC, T = (probs_mix[:, k] for k in range(5))
    tot = e1 + e2 + 1e-8
    a1, a2 = e1 / tot, e2 / tot                    # energy share, in [0,1]
    voiced = (e1 + e2) > 0.25 * np.median(e1 + e2 + 1e-8)
    a1, a2 = a1 * voiced, a2 * voiced
    hold1, hold2 = _smooth_prev(a1, W_FRAMES), _smooth_prev(a2, W_FRAMES)
    onset1 = np.clip(a1 - _smooth_prev(a1, W_FRAMES), 0, 1) * hold2
    onset2 = np.clip(a2 - _smooth_prev(a2, W_FRAMES), 0, 1) * hold1
    eot = T  # EOT = turn-change probability only (NA omitted)
    return {
        "eot_score_speaker_1": (eot * hold1).astype(np.float32),
        "eot_score_speaker_2": (eot * hold2).astype(np.float32),
        "interruption_score_speaker_1": (I * onset1).astype(np.float32),
        "interruption_score_speaker_2": (I * onset2).astype(np.float32),
    }


def _frame_time(n: int) -> np.ndarray:
    """Acoustic time (s) the model has heard up to at each prediction frame."""
    return (START_CHUNK + (np.arange(n) + 1) * HOP) / SR


def _commit(score: np.ndarray, times: np.ndarray, tau_high: float,
            tau_low: float, t_max: float | None = None,
            refractory_s: float = 0.0) -> list[float]:
    """Online hysteresis detector -> sorted, strictly-increasing commit times.

    Emit `times[i]` the first frame `score[i] >= tau_high` while armed, then stay
    disarmed until `score[i] <= tau_low`; consecutive commits are >= refractory_s
    apart. Causal: uses only frame i."""
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


def speaker_events(scores: dict, n: int, duration_s: float) -> dict:
    """Per-frame score channels -> {speaker: SpeakerEvents}."""
    t = _frame_time(n)
    events = {}
    for spk in (1, 2):
        events[spk] = SpeakerEvents(
            eot=_commit(scores[f"eot_score_speaker_{spk}"], t,
                        EOT_TAU_HIGH, EOT_TAU_LOW, duration_s, EOT_REFRACTORY_S),
            interruption=_commit(scores[f"interruption_score_speaker_{spk}"], t,
                                 INT_TAU_HIGH, INT_TAU_LOW, duration_s,
                                 INT_REFRACTORY_S),
        )
    return events


def conversation_scores(conv: Conversation, cache_dir: Path) -> dict:
    """Per-frame attribution channels for one conversation, via cache or model."""
    cache_path = cache_dir / f"{conv.conversation_id}.npz"
    if cache_path.exists():
        d = np.load(cache_path)
        return _attribute(d["probs_mix"], d["e1"], d["e2"])
    ch1 = _to_16k(*conv.audio(1))
    ch2 = _to_16k(*conv.audio(2))
    n = min(len(ch1), len(ch2))
    ch1, ch2 = ch1[:n], ch2[:n]
    probs_mix = _probs_for_signal(ch1 + ch2)        # mix = speaker_1 + speaker_2
    tn = len(probs_mix)
    e1, e2 = _frame_energy(ch1, tn), _frame_energy(ch2, tn)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, probs_mix=probs_mix, e1=e1, e2=e2)
    return _attribute(probs_mix, e1, e2)


def predict(conv: Conversation, cache_dir: Path = _DEFAULT_CACHE) -> ConversationPrediction:
    scores = conversation_scores(conv, cache_dir)
    n = len(scores["eot_score_speaker_1"])
    events = speaker_events(scores, n, conv.duration_s)
    return ConversationPrediction(
        conversation_id=conv.conversation_id,
        speaker_1=events[1],
        speaker_2=events[2],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default=DEV_DATASET,
        help="HF dataset repo id, or a local directory of parquet shards",
    )
    parser.add_argument(
        "--out", default=None,
        help="write a predictions JSON here instead of scoring",
    )
    parser.add_argument(
        "--cache-dir", default=str(_DEFAULT_CACHE),
        help="per-frame score cache (reused if present, else written)",
    )
    parser.add_argument(
        "--shard", type=int, default=0,
        help="this shard's index in [0, num-shards) — process conversation_ids"
             "[shard::num-shards] and only populate the cache (no JSON/score)",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="total shards; >1 splits the model run across parallel processes "
             "sharing --cache-dir. Run once with --num-shards 1 afterwards to "
             "emit the merged JSON from the now-complete cache.",
    )
    args = parser.parse_args()
    cache_dir = Path(args.cache_dir)

    dataset = resolve_dataset(source=args.dataset)
    task_ids = conversation_ids(dataset)

    # Sharded mode: just compute + cache this shard's conversations, in parallel
    # with sibling processes writing the same cache (distinct ids, no collision).
    if args.num_shards > 1:
        mine = task_ids[args.shard::args.num_shards]
        for task_id in mine:
            conversation_scores(conversation(dataset, task_id), cache_dir)
            print(f"shard {args.shard}/{args.num_shards}: cached {task_id}",
                  file=sys.stderr)
        print(f"shard {args.shard}/{args.num_shards}: done ({len(mine)} convs)",
              file=sys.stderr)
        return 0

    submission = Submission(
        schema_version=SCHEMA_VERSION,
        predictions=[
            predict(conversation(dataset, task_id), cache_dir)
            for task_id in task_ids
        ],
    )

    if args.out is not None:
        Path(args.out).write_text(
            submission.model_dump_json(indent=2), encoding="utf-8"
        )
        print(f"Wrote {len(submission.predictions)} predictions to {args.out}",
              file=sys.stderr)
        return 0

    from eval.score import score_submission, task_cells  # lazy: needs typer/rich

    scores = score_submission(submission, dataset)
    print(f"espnet_turntaking — {len(submission.predictions)} conversations")
    for task_name, score in (("EOT", scores.task_eot), ("INT", scores.task_int)):
        recall, fp_rate, latency = task_cells(score)
        print(f"  {task_name}: recall={recall} fp_rate={fp_rate} latency_ms={latency}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
