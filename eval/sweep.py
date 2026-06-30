"""TurnBench dev threshold-sweep: the probabilities files and their validation.

The sweep figure (paper Fig. 1) sweeps a single decision threshold over a model's
*continuous* per-frame probabilities on the dev set and scores each operating
point with eval.score. To make that reproducible without re-running anyone's
model, a submitter commits the raw per-frame probabilities — `probs-eot.json` for
the EOT sweep, `probs-int.json` for the interruption sweep. Each is the continuous
twin of a predictions JSON: instead of committed event times, a per-frame
probability per speaker, on a fixed frame grid. A model with only one head
commits only that task's file. One schema serves both, discriminated by `task`.

This module owns that schema and a strict, loud validator (mirroring
eval/submission.py): every dev conversation present exactly once, both speakers,
every probability finite and in [0, 1], and exactly one probability per frame on
the benchmark's canonical grid (`floor(duration * frame_rate_hz)`). The grid is
the benchmark's, not the submitter's — so when the sweep commits events it reads
their times from the grid, never from the file, and they are exact by
construction. Scoring (threshold -> Submission -> eval.score) lands on top of
this.

Dev only: there is no test probabilities file — test is scored from each model's
single committed `predictions-test.json` at its declared operating point.

    python -m eval.sweep validate baselines/<name>/probs-eot.json
"""

import json
import math
from pathlib import Path
from typing import Annotated, Literal

import typer
from pydantic import BaseModel, ConfigDict, field_validator

from eval.durations import load_durations

SCHEMA_VERSION = 1


class SpeakerProbs(BaseModel):
    """One speaker's per-frame probability for the file's task — one value per
    frame on the canonical grid, where frame i covers [i / fps, (i + 1) / fps)."""

    model_config = ConfigDict(extra="forbid")
    prob: list[float]

    @field_validator("prob")
    @classmethod
    def in_unit_interval(cls, probs: list[float]) -> list[float]:
        for prob in probs:
            # NaN / inf compare False against both bounds, so this also rejects them.
            if not (0.0 <= prob <= 1.0):
                raise ValueError(
                    f"probabilities must be finite and in [0, 1], got {prob}"
                )
        return probs


class ConversationProbs(BaseModel):
    """Both channels' per-frame EOT probabilities for one conversation.
    `speaker_1` / `speaker_2` mirror the dataset's audio channels."""

    model_config = ConfigDict(extra="forbid")
    conversation_id: str
    speaker_1: SpeakerProbs
    speaker_2: SpeakerProbs


class ProbsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    task: Literal["eot", "int"]
    frame_rate_hz: float
    probs: list[ConversationProbs]

    @field_validator("frame_rate_hz")
    @classmethod
    def positive_finite(cls, frame_rate_hz: float) -> float:
        if not (frame_rate_hz > 0.0 and frame_rate_hz != float("inf")):
            raise ValueError(f"frame_rate_hz must be finite and > 0, got {frame_rate_hz}")
        return frame_rate_hz

    def by_conversation(self) -> dict[str, ConversationProbs]:
        return {entry.conversation_id: entry for entry in self.probs}


def load_probs(path: Path) -> ProbsFile:
    """Parse and schema-validate a probabilities JSON file."""
    return ProbsFile.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def frame_count(duration_s: float, frame_rate_hz: float) -> int:
    """The canonical number of frames for a conversation at this frame rate:
    `floor(duration * fps)`. Frame i covers [i / fps, (i + 1) / fps); this is the
    single framing convention every probs file is held to."""
    return math.floor(duration_s * frame_rate_hz)


def validate_coverage(probs: ProbsFile, conversation_ids: list[str]) -> None:
    """Every dev conversation must appear exactly once — no missing, unknown, or
    duplicate. Same rule as a predictions submission: absence is an error, not
    "found nothing"."""
    submitted = [entry.conversation_id for entry in probs.probs]
    duplicates = {cid for cid in submitted if submitted.count(cid) > 1}
    if duplicates:
        raise ValueError(f"duplicate conversation_ids: {sorted(duplicates)}")
    missing = set(conversation_ids) - set(submitted)
    if missing:
        raise ValueError(f"missing conversation_ids: {sorted(missing, key=int)}")
    unknown = set(submitted) - set(conversation_ids)
    if unknown:
        raise ValueError(f"unknown conversation_ids: {sorted(unknown)}")


def validate_frame_counts(
    probs: ProbsFile, duration_s_by_id: dict[str, float]
) -> None:
    """Each speaker's probability array must hold exactly one value per frame on
    the canonical grid (`floor(duration * frame_rate_hz)`). An off-by-N array
    means its implied time axis has drifted from the audio, so it is rejected,
    not silently snapped."""
    fps = probs.frame_rate_hz
    for entry in probs.probs:
        duration_s = duration_s_by_id[entry.conversation_id]
        expected = frame_count(duration_s, fps)
        for speaker, speaker_probs in ((1, entry.speaker_1), (2, entry.speaker_2)):
            actual = len(speaker_probs.prob)
            if actual != expected:
                raise ValueError(
                    f"conversation {entry.conversation_id} speaker_{speaker}: "
                    f"{actual} frames, expected {expected} "
                    f"(= floor({duration_s:.2f}s * {fps} Hz))"
                )


def validate_probs(probs: ProbsFile, duration_s_by_id: dict[str, float]) -> None:
    """Full strict validation against the dataset: coverage + exact frame counts.
    Probability range and finiteness are enforced at parse time by the schema.
    `duration_s_by_id` maps every dataset conversation_id to its audio duration."""
    validate_coverage(probs, list(duration_s_by_id))
    validate_frame_counts(probs, duration_s_by_id)


app = typer.Typer(add_completion=False, help="TurnBench dev threshold-sweep tooling.")


@app.callback()
def main() -> None:
    """TurnBench dev threshold-sweep tooling (validate now; score to follow)."""


@app.command()
def validate(
    probs_path: Annotated[
        Path, typer.Argument(help="probabilities JSON (probs-eot.json / probs-int.json)")
    ],
) -> None:
    """Validate a probs file against the dev grid (strict and loud — no dataset)."""
    probs = load_probs(probs_path)
    validate_probs(probs, load_durations("dev"))
    typer.echo(
        f"OK: {probs_path} — task={probs.task}, {len(probs.probs)} conversations, "
        f"frame_rate_hz={probs.frame_rate_hz}"
    )


if __name__ == "__main__":
    app()
