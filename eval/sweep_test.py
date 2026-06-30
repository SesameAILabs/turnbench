"""Probabilities-file schema + validation tests. Run: uv run pytest eval/sweep_test.py"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval.sweep import (
    ProbsFile,
    load_probs,
    validate_coverage,
    validate_frame_counts,
    validate_probs,
)


def payload(**overrides) -> dict:
    """A minimal valid probs payload; keyword overrides patch the top level.

    "233" has 3 frames per speaker at 25 Hz, so its canonical duration is in
    [3/25, 4/25) s — the tests below use 0.14 s (mid-bucket, float-safe).
    """
    base = {
        "schema_version": 1,
        "task": "eot",
        "frame_rate_hz": 25.0,
        "probs": [
            {
                "conversation_id": "233",
                "speaker_1": {"prob": [0.1, 0.2, 0.9]},
                "speaker_2": {"prob": [0.0, 0.5, 0.3]},
            },
        ],
    }
    base.update(overrides)
    return base


def entry(conversation_id: str, speaker_1_prob: list[float]) -> dict:
    return {
        "conversation_id": conversation_id,
        "speaker_1": {"prob": speaker_1_prob},
        "speaker_2": {"prob": [0.0, 0.0, 0.0]},
    }


def test_valid_probs_round_trips(tmp_path: Path):
    path = tmp_path / "probs-dev.json"
    path.write_text(json.dumps(payload()))

    probs = load_probs(path)

    assert probs.schema_version == 1
    assert probs.task == "eot"
    assert probs.frame_rate_hz == 25.0
    assert probs.by_conversation()["233"].speaker_1.prob == [0.1, 0.2, 0.9]


def test_extra_keys_rejected():
    with pytest.raises(ValidationError):
        ProbsFile.model_validate(payload(model_name="sneaky"))


def test_unknown_schema_version_rejected():
    with pytest.raises(ValidationError):
        ProbsFile.model_validate(payload(schema_version=2))


def test_probabilities_outside_unit_interval_rejected():
    for bad in (1.5, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValidationError, match=r"\[0, 1\]"):
            ProbsFile.model_validate(payload(probs=[entry("233", [0.1, bad, 0.9])]))


def test_frame_rate_must_be_positive_and_finite():
    for bad in (0.0, -25.0, float("inf")):
        with pytest.raises(ValidationError, match="frame_rate_hz"):
            ProbsFile.model_validate(payload(frame_rate_hz=bad))


def test_coverage_rejects_missing_unknown_and_duplicate():
    probs = ProbsFile.model_validate(payload(probs=[entry("233", [0.1, 0.2, 0.9])]))
    with pytest.raises(ValueError, match="missing"):
        validate_coverage(probs, ["233", "417"])

    extra = ProbsFile.model_validate(
        payload(probs=[entry("233", [0.1, 0.2, 0.9]), entry("999", [0.1, 0.2, 0.9])])
    )
    with pytest.raises(ValueError, match="unknown"):
        validate_coverage(extra, ["233"])

    duplicated = ProbsFile.model_validate(
        payload(probs=[entry("233", [0.1, 0.2, 0.9]), entry("233", [0.1, 0.2, 0.9])])
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_coverage(duplicated, ["233"])


def test_frame_count_mismatch_rejected():
    probs = ProbsFile.model_validate(payload())  # 3 frames/speaker at 25 Hz
    validate_frame_counts(probs, {"233": 0.14})  # floor(0.14 * 25) == 3 — ok

    with pytest.raises(ValueError, match="frames, expected"):
        validate_frame_counts(probs, {"233": 0.50})  # expects floor(12.5) == 12


def test_validate_probs_accepts_a_consistent_file():
    probs = ProbsFile.model_validate(payload())
    validate_probs(probs, {"233": 0.14})  # coverage + frame counts both pass
