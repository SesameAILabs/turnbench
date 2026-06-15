"""The submission format: a single predictions JSON. See README.md.

A submission declares, per conversation and per speaker, the times (s) at
which the model detects EOTs and interruptions. The structure is fixed-key
throughout; unknown keys are rejected. Schema problems and coverage problems
(missing / duplicate / unknown conversations) fail loudly — a submission is
data, and irregular data must crash the run, not be skipped into a slightly
different score.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

SCHEMA_VERSION = 1


class SpeakerEvents(BaseModel):
    """One speaker's predicted event times (s), each list sorted ascending.

    `eot`: this speaker's turn ends. `interruption`: this speaker takes the
    floor from the other. Timestamps are commit times — the time by which all
    audio the decision depended on has been heard.
    """

    model_config = ConfigDict(extra="forbid")
    eot: list[float]
    interruption: list[float]

    @field_validator("eot", "interruption")
    @classmethod
    def sorted_non_negative_finite(cls, times: list[float]) -> list[float]:
        for time_s in times:
            if not (time_s >= 0.0 and time_s != float("inf")):
                raise ValueError(f"event times must be finite and >= 0, got {time_s}")
        if any(
            earlier >= later for earlier, later in zip(times, times[1:])
        ):
            raise ValueError(f"event times must be strictly increasing, got {times}")
        return times


class ConversationPrediction(BaseModel):
    """All predicted events for one conversation. `speaker_1` / `speaker_2`
    mirror the dataset's `speaker_{1,2}_audio.flac` channels — never remapped
    by who the "agent" is. Empty lists mean "ran it, found nothing"."""

    model_config = ConfigDict(extra="forbid")
    conversation_id: str
    speaker_1: SpeakerEvents
    speaker_2: SpeakerEvents


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    predictions: list[ConversationPrediction]

    def by_conversation(self) -> dict[str, ConversationPrediction]:
        return {
            prediction.conversation_id: prediction for prediction in self.predictions
        }


def load_submission(path: Path) -> Submission:
    """Parse and schema-validate a predictions JSON file."""
    return Submission.model_validate(json.loads(path.read_text(encoding="utf-8")))


def validate_coverage(
    submission: Submission, conversation_ids: list[str]
) -> None:
    """Every released conversation must appear exactly once — no extras.

    A conversation absent from the file is ambiguous (found nothing, or forgot
    to run it?), so absence is an error; "found nothing" is spelled with empty
    event lists.
    """
    submitted = [prediction.conversation_id for prediction in submission.predictions]
    duplicates = {
        conversation_id
        for conversation_id in submitted
        if submitted.count(conversation_id) > 1
    }
    if duplicates:
        raise ValueError(f"duplicate conversation_ids: {sorted(duplicates)}")
    missing = set(conversation_ids) - set(submitted)
    if missing:
        raise ValueError(f"missing conversation_ids: {sorted(missing, key=int)}")
    unknown = set(submitted) - set(conversation_ids)
    if unknown:
        raise ValueError(f"unknown conversation_ids: {sorted(unknown)}")


def validate_event_times(
    prediction: ConversationPrediction, duration_s: float
) -> None:
    """Every predicted time must lie within the conversation's audio."""
    for speaker, events in ((1, prediction.speaker_1), (2, prediction.speaker_2)):
        for task_name, times in (("eot", events.eot), ("interruption", events.interruption)):
            out_of_range = [time_s for time_s in times if time_s > duration_s]
            if out_of_range:
                raise ValueError(
                    f"conversation {prediction.conversation_id} speaker_{speaker} "
                    f"{task_name}: events past the end of the audio "
                    f"({duration_s:.2f}s): {out_of_range} — are these seconds?"
                )
