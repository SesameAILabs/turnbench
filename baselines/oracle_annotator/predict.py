#!/usr/bin/env python3
"""Oracle baseline + reference for the in-memory scoring interface.

Emits the gold events themselves as the prediction, then scores them with
`turnbench.score.score_submission`. By construction this is a perfect submission
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

from turnbench.data import (  # noqa: E402
    DEV_DATASET,
    Dataset,
    conversation,
    conversation_ids,
    resolve_dataset,
)
from turnbench.gold import events_for_conversation  # noqa: E402
from turnbench.score import score_submission, task_cells  # noqa: E402
from turnbench.submission import (  # noqa: E402
    SCHEMA_VERSION,
    ConversationPrediction,
    SpeakerEvents,
    Submission,
)


def gold_submission(dataset: Dataset) -> Submission:
    """Build the perfect submission: every gold positive, as committed events."""

    def times(anchors, speaker: int, duration: float) -> list[float]:
        # A gold median can round a few ms past the audio end; a real model
        # could never emit past the audio it heard, so clamp into bounds.
        return sorted(min(a.time_s, duration) for a in anchors if a.speaker == speaker)

    predictions = []
    for task_id in conversation_ids(dataset):
        conv = conversation(dataset, task_id)
        events = events_for_conversation(conv)
        duration = conv.duration_s
        predictions.append(
            ConversationPrediction(
                conversation_id=task_id,
                speaker_1=SpeakerEvents(
                    eot=times(events.eot_positive_events, 1, duration),
                    interruption=times(events.int_positive_events, 1, duration),
                ),
                speaker_2=SpeakerEvents(
                    eot=times(events.eot_positive_events, 2, duration),
                    interruption=times(events.int_positive_events, 2, duration),
                ),
            )
        )
    return Submission(schema_version=SCHEMA_VERSION, predictions=predictions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=DEV_DATASET,
        help="HF dataset repo id, or a local directory of parquet shards",
    )
    parser.add_argument(
        "--out", default=None, help="write a predictions JSON here instead of scoring"
    )
    args = parser.parse_args()

    dataset = resolve_dataset(source=args.dataset, skip_audio=True)
    submission = gold_submission(dataset)

    if args.out is not None:
        Path(args.out).write_text(submission.model_dump_json(indent=2), encoding="utf-8")
        print(f"Wrote {len(submission.predictions)} predictions to {args.out}", file=sys.stderr)
        return 0

    scores = score_submission(submission, dataset)
    for task_name, score in (("EOT", scores.task_eot), ("INT", scores.task_int)):
        recall, fp_rate, latency = task_cells(score)
        print(f"{task_name}: recall={recall} fp_rate={fp_rate} latency={latency}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
