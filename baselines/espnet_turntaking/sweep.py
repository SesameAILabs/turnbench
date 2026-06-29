#!/usr/bin/env python3
"""Pick the espnet_turntaking operating point on the official scorer.

`docs/SUBMISSION_FORMAT.md` blesses in-memory sweeping: build a `Submission` of
committed event times for a candidate threshold and call
`eval.score.score_submission` against the real (2-of-3) gold. This sweeps each
track's hysteresis `tau_high` (with `tau_low = low_ratio * tau_high` and a fixed
`refractory`), holding the other track at the module defaults in `predict.py`,
and prints recall / fp_rate / latency p50 so the operating point is reproducible.

Per-frame scores come from the cache (`--cache-dir`), so no model run is needed.

    python -m baselines.espnet_turntaking.sweep
    python -m baselines.espnet_turntaking.sweep --track eot --grid 0.04:0.20:0.02
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from baselines.espnet_turntaking import predict as P  # noqa: E402
from eval.data import (  # noqa: E402
    DEV_DATASET,
    conversation,
    conversation_ids,
    resolve_dataset,
)
from eval.score import score_submission  # noqa: E402
from eval.submission import (  # noqa: E402
    SCHEMA_VERSION,
    ConversationPrediction,
    SpeakerEvents,
    Submission,
)

TRACKS = ("eot", "interruption")
DEFAULT_GRID = {"eot": "0.02:0.40:0.02", "interruption": "0.01:0.25:0.01"}


def _parse_grid(spec: str) -> list[float]:
    lo, hi, step = (float(x) for x in spec.split(":"))
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 4) for i in range(n)]


def _events(score, times, dur, th, tl, refr):
    return P._commit(score, times, th, tl, dur, refr)


def build_submission(convs, track, tau_high, tau_low, refractory):
    """A Submission with `track` committed at the swept thresholds and the other
    track at the predict.py defaults."""
    predictions = []
    for cid, dur, scores in convs:
        n = len(scores["eot_score_speaker_1"])
        times = P._frame_time(n)
        speakers = {}
        for spk in (1, 2):
            eot_score = scores[f"eot_score_speaker_{spk}"]
            int_score = scores[f"interruption_score_speaker_{spk}"]
            if track == "eot":
                eot = _events(eot_score, times, dur, tau_high, tau_low, refractory)
                interruption = _events(int_score, times, dur, P.INT_TAU_HIGH,
                                       P.INT_TAU_LOW, P.INT_REFRACTORY_S)
            else:
                eot = _events(eot_score, times, dur, P.EOT_TAU_HIGH,
                              P.EOT_TAU_LOW, P.EOT_REFRACTORY_S)
                interruption = _events(int_score, times, dur, tau_high, tau_low,
                                       refractory)
            speakers[spk] = SpeakerEvents(eot=eot, interruption=interruption)
        predictions.append(ConversationPrediction(
            conversation_id=cid, speaker_1=speakers[1], speaker_2=speakers[2]))
    return Submission(schema_version=SCHEMA_VERSION, predictions=predictions)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEV_DATASET)
    ap.add_argument("--cache-dir", default=str(P._DEFAULT_CACHE))
    ap.add_argument("--track", choices=TRACKS, default=None)
    ap.add_argument("--grid", default=None, help="tau_high grid lo:hi:step")
    ap.add_argument("--low-ratio", type=float, default=0.4)
    ap.add_argument("--refractory", type=float, default=2.0)
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir)

    dataset = resolve_dataset(source=args.dataset)
    convs = []
    for cid in conversation_ids(dataset):
        conv = conversation(dataset, cid)
        convs.append((cid, conv.duration_s,
                      P.conversation_scores(conv, cache_dir)))

    for track in ([args.track] if args.track else list(TRACKS)):
        grid = _parse_grid(args.grid or DEFAULT_GRID[track])
        print(f"== sweep {track}: tau_low={args.low_ratio}*tau_high, "
              f"refractory={args.refractory}s ==")
        print(f"{'tau_hi':>7} {'tau_lo':>7}  {'recall':>7} {'fp_rate':>8} "
              f"{'lat p10/50/90':>16}")
        for tau_high in grid:
            tau_low = round(tau_high * args.low_ratio, 4)
            sub = build_submission(convs, track, tau_high, tau_low, args.refractory)
            agg = score_submission(sub, dataset)
            score = agg.task_eot if track == "eot" else agg.task_int
            lat = score.latency()
            print(f"{tau_high:>7.3f} {tau_low:>7.3f}  {score.recall:>7.3f} "
                  f"{score.fp_rate:>8.3f}  "
                  f"{lat.p10:>5.0f}/{lat.p50:>5.0f}/{lat.p90:>5.0f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
