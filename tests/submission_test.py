"""Submission format tests. Run: uv run pytest tests/submission_test.py"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from turnbench.submission import (
    Submission,
    load_submission,
    validate_coverage,
    validate_event_times,
)


def payload(**overrides) -> dict:
    """A minimal valid submission payload; keyword overrides patch the top level."""
    base = {
        "schema_version": 1,
        "predictions": [
            {
                "conversation_id": "233",
                "speaker_1": {"eot": [12.84, 58.43], "interruption": [31.2]},
                "speaker_2": {"eot": [27.1], "interruption": []},
            },
        ],
    }
    base.update(overrides)
    return base


def predictions(*speaker_1_eot_by_id: tuple[str, list[float]]) -> list[dict]:
    """Conversation entries with the given (conversation_id, speaker_1 eot) pairs."""
    return [
        {
            "conversation_id": conversation_id,
            "speaker_1": {"eot": eot, "interruption": []},
            "speaker_2": {"eot": [], "interruption": []},
        }
        for conversation_id, eot in speaker_1_eot_by_id
    ]


def test_valid_submission_round_trips(tmp_path: Path):
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(payload()))

    submission = load_submission(path)

    assert submission.schema_version == 1
    assert submission.predictions[0].speaker_1.eot == [12.84, 58.43]
    assert submission.by_conversation()["233"].speaker_2.interruption == []


def test_unknown_schema_version_rejected():
    with pytest.raises(ValidationError):
        Submission.model_validate(payload(schema_version=2))


def test_extra_keys_rejected():
    with pytest.raises(ValidationError):
        Submission.model_validate(payload(model_name="sneaky"))


def test_missing_speaker_rejected():
    entry = payload()["predictions"][0]
    del entry["speaker_2"]
    with pytest.raises(ValidationError):
        Submission.model_validate(payload(predictions=[entry]))


def test_unsorted_times_rejected():
    with pytest.raises(ValidationError, match="strictly increasing"):
        Submission.model_validate(
            payload(predictions=predictions(("233", [58.43, 12.84])))
        )


def test_duplicate_times_rejected():
    with pytest.raises(ValidationError, match="strictly increasing"):
        Submission.model_validate(
            payload(predictions=predictions(("233", [12.84, 12.84])))
        )


def test_negative_and_non_finite_times_rejected():
    for bad_times in ([-1.0], [float("nan")], [float("inf")]):
        with pytest.raises(ValidationError, match="finite"):
            Submission.model_validate(
                payload(predictions=predictions(("233", bad_times)))
            )


def test_coverage_accepts_exact_match():
    submission = Submission.model_validate(
        payload(predictions=predictions(("233", []), ("417", [])))
    )
    validate_coverage(submission, ["233", "417"])


def test_coverage_rejects_missing_and_unknown_and_duplicates():
    submission = Submission.model_validate(
        payload(predictions=predictions(("233", []), ("999", [])))
    )
    with pytest.raises(ValueError, match="missing"):
        validate_coverage(submission, ["233", "417", "999"])
    with pytest.raises(ValueError, match="unknown"):
        validate_coverage(submission, ["233"])

    duplicated = Submission.model_validate(
        payload(predictions=predictions(("233", []), ("233", [])))
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_coverage(duplicated, ["233"])


def test_event_times_must_fit_the_audio():
    submission = Submission.model_validate(
        payload(predictions=predictions(("233", [12840.0])))
    )
    with pytest.raises(ValueError, match="past the end"):
        validate_event_times(submission.predictions[0], duration_s=300.0)
