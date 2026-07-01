#!/usr/bin/env python3
"""Build the TurnBench submission artifacts for `espnet_turntaking_perchannel`
from the per-frame probability cache — no model re-run.

The new baseline flow (`baselines/#readme`):
  * `probs-{eot,int}.json` — per-frame continuous scores on the **dev** set, on
    the canonical grid (`floor(duration_s * 25)` frames/speaker). `eval.sweep`
    scores these centrally and picks the operating point (highest recall at
    `fp_rate <= 0.1`), giving theta_eot / theta_int.
  * `predictions-{dev,test}.json` — events committed at those thresholds with the
    central rule `eval.sweep.commit_events` (single rising-edge theta + 2 s
    refractory).

Continuous signal, per speaker channel K: `eot = P_T`, `int = P_I`, read straight
off channel K's 5-class softmax (`probs1` / `probs2` in the cache). The model's
first prediction is at `START_CHUNK` = 0.2 s, so its arrays omit the first 5
frames of the canonical grid; we **left-pad 5 zeros** (no EOT/interruption can
occur in `[0, 0.2 s)`). Verified: `floor(dur*25) - len(cache) == 5` for every
conversation, so this lands exactly on the grid and keeps the time axis exact.

    # 1) dev probs for the sweep
    python -m baselines.espnet_turntaking_perchannel.submit probs --task eot \
        --out baselines/espnet_turntaking_perchannel/probs-eot.json
    python -m baselines.espnet_turntaking_perchannel.submit probs --task int \
        --out baselines/espnet_turntaking_perchannel/probs-int.json
    # 2) operating point
    uv run python -m eval.sweep baselines/espnet_turntaking_perchannel/probs-eot.json  # -> theta_eot
    uv run python -m eval.sweep baselines/espnet_turntaking_perchannel/probs-int.json  # -> theta_int
    # 3) committed predictions at (theta_eot, theta_int)
    python -m baselines.espnet_turntaking_perchannel.submit predictions --split dev \
        --theta-eot T1 --theta-int T2 \
        --out baselines/espnet_turntaking_perchannel/predictions-dev.json
    python -m baselines.espnet_turntaking_perchannel.submit predictions --split test \
        --theta-eot T1 --theta-int T2 \
        --cache-dir predictions/espnet_turntaking_perchannel/cache \
        --out baselines/espnet_turntaking_perchannel/predictions-test.json
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from eval.durations import load_durations
from eval.submission import ConversationPrediction, SpeakerEvents, Submission
from eval.sweep import (
    ConversationProbs,
    ProbsFile,
    SpeakerProbs,
    commit_events,
)

from baselines.espnet_turntaking_perchannel.predict import (
    I_IDX,
    T_IDX,
    _DEFAULT_CACHE,
)

FPS = 25.0
PREROLL = 5  # START_CHUNK (0.2 s) * 25 Hz — canonical frames the model never emits


def _canon(arr: np.ndarray, duration_s: float) -> list[float]:
    """Left-pad the model's per-frame scores with PREROLL zeros and land them on
    the canonical grid length `floor(duration_s * 25)`. The 0.2 s pre-roll where
    the model makes no prediction becomes prob 0 (no rising edge, no event).
    Verified `floor(dur*25) - len(arr) == PREROLL` for every conversation, so the
    trailing trim/pad is a defensive no-op."""
    target = math.floor(duration_s * FPS)
    padded = np.concatenate(
        [np.zeros(PREROLL, dtype=np.float64), np.asarray(arr, dtype=np.float64)]
    )
    if len(padded) < target:
        padded = np.concatenate([padded, np.zeros(target - len(padded))])
    return [float(x) for x in padded[:target]]


def _channels(cid: str, cache_dir: Path):
    """(eot1, int1, eot2, int2): raw per-frame P_T / P_I for each speaker channel."""
    d = np.load(cache_dir / f"{cid}.npz")
    p1, p2 = d["probs1"], d["probs2"]
    return p1[:, T_IDX], p1[:, I_IDX], p2[:, T_IDX], p2[:, I_IDX]


def emit_probs(task: str, split: str, cache_dir: Path, out: Path) -> None:
    dur = load_durations(split)
    entries = []
    for cid in sorted(dur, key=int):
        eot1, int1, eot2, int2 = _channels(cid, cache_dir)
        s1, s2 = (eot1, eot2) if task == "eot" else (int1, int2)
        entries.append(
            ConversationProbs(
                conversation_id=cid,
                speaker_1=SpeakerProbs(prob=_canon(s1, dur[cid])),
                speaker_2=SpeakerProbs(prob=_canon(s2, dur[cid])),
            )
        )
    pf = ProbsFile(schema_version=1, task=task, frame_rate_hz=FPS, probs=entries)
    out.write_text(pf.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {len(entries)} {task} prob rows ({split}) -> {out}", file=sys.stderr)


def emit_predictions(
    theta_eot: float, theta_int: float, split: str, cache_dir: Path, out: Path
) -> None:
    dur = load_durations(split)
    preds = []
    for cid in sorted(dur, key=int):
        eot1, int1, eot2, int2 = _channels(cid, cache_dir)
        e1, i1 = _canon(eot1, dur[cid]), _canon(int1, dur[cid])
        e2, i2 = _canon(eot2, dur[cid]), _canon(int2, dur[cid])
        preds.append(
            ConversationPrediction(
                conversation_id=cid,
                speaker_1=SpeakerEvents(
                    eot=commit_events(e1, FPS, theta_eot),
                    interruption=commit_events(i1, FPS, theta_int),
                ),
                speaker_2=SpeakerEvents(
                    eot=commit_events(e2, FPS, theta_eot),
                    interruption=commit_events(i2, FPS, theta_int),
                ),
            )
        )
    sub = Submission(schema_version=1, predictions=preds)
    out.write_text(sub.model_dump_json(indent=2), encoding="utf-8")
    print(
        f"wrote {len(preds)} predictions ({split}) @ theta_eot={theta_eot} "
        f"theta_int={theta_int} -> {out}",
        file=sys.stderr,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probs", help="emit dev per-frame probs for one task")
    p.add_argument("--task", choices=["eot", "int"], required=True)
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--cache-dir", default=str(_DEFAULT_CACHE))
    p.add_argument("--out", required=True)

    q = sub.add_parser("predictions", help="commit events at the operating point")
    q.add_argument("--theta-eot", type=float, required=True)
    q.add_argument("--theta-int", type=float, required=True)
    q.add_argument("--split", default="dev", choices=["dev", "test"])
    q.add_argument("--cache-dir", default=str(_DEFAULT_CACHE))
    q.add_argument("--out", required=True)

    a = ap.parse_args()
    if a.cmd == "probs":
        emit_probs(a.task, a.split, Path(a.cache_dir), Path(a.out))
    else:
        emit_predictions(
            a.theta_eot, a.theta_int, a.split, Path(a.cache_dir), Path(a.out)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
