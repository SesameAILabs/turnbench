"""Probabilities-file schema + validation tests. Run: uv run pytest tests/sweep_test.py"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from turnbench.sweep import (
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

# ---- commit rule + candidate thresholds -------------------------------------


def _commit_events_reference(probs, frame_rate_hz, theta, refractory_s):
    """The original pure-Python commit rule, kept verbatim as the behavioural
    reference for the vectorized implementation."""
    refractory_frames = max(1, round(refractory_s * frame_rate_hz))
    times = []
    above_prev = False
    last = None
    for i, prob in enumerate(probs):
        above = prob > theta
        if above and not above_prev and (last is None or i - last >= refractory_frames):
            times.append((i + 1) / frame_rate_hz)
            last = i
        above_prev = above
    return times


def test_commit_events_matches_reference_implementation():
    """The vectorized commit rule must reproduce the reference loop exactly —
    including the subtlety that a refractory-suppressed rising edge does NOT
    extend the refractory window (only committed events do)."""
    import random

    from turnbench.sweep import commit_events

    rng = random.Random(7)
    cases = [
        [],                                    # empty
        [0.0] * 50,                            # never fires
        [1.0] * 50,                            # one rising edge at frame 0
        [0.0, 1.0] * 200,                      # chattering: refractory suppression
        [rng.random() ** 4 for _ in range(2000)],   # compressed-scale scores
        [1 - rng.random() ** 4 for _ in range(2000)],  # mass near 1
    ]
    for probs in cases:
        for theta in (0.0, 0.001, 0.03, 0.5, 0.97, 1.0):
            for refractory_s in (0.04, 2.0):
                assert commit_events(probs, 25.0, theta, refractory_s=refractory_s) == \
                    _commit_events_reference(probs, 25.0, theta, refractory_s=refractory_s), \
                    f"mismatch at theta={theta} refractory={refractory_s}"


def test_candidate_thetas_are_quantiles_plus_uniform_grid():
    """Candidates = quantiles of the model's own pooled nonzero scores (resolution
    where the mass is — a compressed score gets sub-0.01 candidates a uniform grid
    can never contain) UNIONED with the uniform 0.01 grid (safety net for operating
    points in thin tails, where quantile candidates go sparse)."""
    from turnbench.sweep import candidate_thetas

    compressed = ProbsFile.model_validate(payload(probs=[
        entry("233", [0.001 * i for i in (1, 1, 2)]),
    ]))
    thetas = candidate_thetas(compressed, n=16)
    assert thetas == sorted(set(thetas))
    assert min(thetas) == 0.001  # quantile part: resolution inside the attained mass
    assert any(t < 0.01 for t in thetas)  # ...below the uniform grid's floor
    assert all(round(0.01 * i, 2) in thetas for i in range(1, 100))  # uniform part present

    all_zero = ProbsFile.model_validate(payload(probs=[entry("233", [0.0, 0.0, 0.0])]))
    assert candidate_thetas(all_zero) == [round(0.01 * i, 2) for i in range(1, 100)]
