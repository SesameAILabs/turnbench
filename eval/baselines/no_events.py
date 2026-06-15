"""Degenerate baseline: predicts no events at all — recall 0, fp_rate 0.
The smallest valid submission, and one end of the latency-vs-FP tradeoff.

Usage: python -m eval.baselines.no_events > predictions.json
"""

from eval.data import conversation_ids, dataset_dir
from eval.submission import (
    SCHEMA_VERSION,
    ConversationPrediction,
    SpeakerEvents,
    Submission,
)


def main() -> None:
    no_events = SpeakerEvents(eot=[], interruption=[])
    submission = Submission(
        schema_version=SCHEMA_VERSION,
        predictions=[
            ConversationPrediction(
                conversation_id=task_id, speaker_1=no_events, speaker_2=no_events
            )
            for task_id in conversation_ids(dataset_dir())
        ],
    )
    print(submission.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
