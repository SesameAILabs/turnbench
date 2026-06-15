#!/usr/bin/env python3
"""Oracle baseline + reference for the in-memory scoring interface.

Emits the gold events themselves as the prediction, then scores them with
`eval.score.score_submission`. By construction this is a perfect submission
(recall 1.0, fp_rate 0.0) — so it doubles as an end-to-end sanity check on the
scorer, and as the worked example of how a model author scores discrete events
without writing a predictions JSON to disk.

The pattern a real submitter follows is the same, except they build the
`SpeakerEvents` lists from their own model's committed event times (thresholding
their continuous scores however they like — that step is theirs, not ours):

    for threshold in my_thresholds:
        submission = Submission(
            schema_version=SCHEMA_VERSION,
            predictions=[my_discrete_events(conversation, threshold) for ...],
        )
        scores = score_submission(submission, data_dir)   # discrete events -> scores

Usage:
    python -m baselines.oracle_annotator.predict [--dataset <hf repo|local dir>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# eval/ is two levels up (baselines/oracle_annotator/predict.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.data import (  # noqa: E402
    DEV_DATASET,
    conversation_duration_s,
    conversation_ids,
    resolve_dataset,
)
from eval.gold import events_for_conversation  # noqa: E402
from eval.score import score_submission, task_cells  # noqa: E402
from eval.submission import (  # noqa: E402
    SCHEMA_VERSION,
    ConversationPrediction,
    SpeakerEvents,
    Submission,
)


def gold_submission(data_dir: Path) -> Submission:
    """Build the perfect submission: every gold positive, as committed events."""
    predictions = []
    for task_id in conversation_ids(data_dir):
        events = events_for_conversation(data_dir / task_id)
        # A gold median can round a few ms past the audio end; a real model
        # could never emit past the audio it heard, so clamp into bounds.
        duration = conversation_duration_s(data_dir / task_id)
        eot = {1: [], 2: []}
        for anchor in events.eot_positive_events:
            eot[anchor.speaker].append(min(anchor.time_s, duration))
        interruption = {1: [], 2: []}
        for anchor in events.int_positive_events:
            interruption[anchor.speaker].append(min(anchor.time_s, duration))
        predictions.append(
            ConversationPrediction(
                conversation_id=task_id,
                speaker_1=SpeakerEvents(eot=sorted(eot[1]), interruption=sorted(interruption[1])),
                speaker_2=SpeakerEvents(eot=sorted(eot[2]), interruption=sorted(interruption[2])),
            )
        )
    return Submission(schema_version=SCHEMA_VERSION, predictions=predictions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=DEV_DATASET,
        help="HF dataset repo id, or a local directory of conversation dirs",
    )
    args = parser.parse_args()

    data_dir = resolve_dataset(source=args.dataset)
    scores = score_submission(gold_submission(data_dir), data_dir)
    for task_name, score in (("EOT", scores.task_eot), ("INT", scores.task_int)):
        recall, fp_rate, latency = task_cells(score)
        print(f"{task_name}: recall={recall} fp_rate={fp_rate} latency={latency}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
