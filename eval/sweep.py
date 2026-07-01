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

    python -m eval.sweep baselines/<name>/probs-eot.json
"""

import json
import math
from dataclasses import dataclass
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


# ---- sweep + scoring (the dev threshold-sweep behind the paper figure) -------

DEFAULT_GRID = [round(0.05 + 0.05 * i, 2) for i in range(19)]  # 0.05 .. 0.95
REFRACTORY_S = 2.0  # matches the baselines' own commit refractory (e.g. espnet)


def commit_events(
    probs: list[float], frame_rate_hz: float, theta: float, *, refractory_s: float = REFRACTORY_S
) -> list[float]:
    """The standard, central commit rule applied to every model's probs: emit one
    event at each rising edge where the probability crosses above `theta` (deduped
    by a refractory), timestamped at the frame's end. As theta rises the crossing
    moves later, so latency rises and spurious fires drop."""
    refractory_frames = max(1, round(refractory_s * frame_rate_hz))
    times: list[float] = []
    above_prev = False
    last = None
    for i, prob in enumerate(probs):
        above = prob > theta
        if above and not above_prev and (last is None or i - last >= refractory_frames):
            times.append((i + 1) / frame_rate_hz)
            last = i
        above_prev = above
    return times


@dataclass(frozen=True)
class SweepRow:
    theta: float
    recall: float
    fp_rate: float          # the task's own fp_rate (mid-turn pauses for eot; backchannels for int) — drives the operating point
    false_int_rate: float   # events fired on backchannel spans / all backchannels — the figure's red axis
    lat_p10: float
    lat_p50: float
    lat_p90: float
    tp: int
    fp: int
    fn: int


def sweep(probs: ProbsFile, dataset, thetas: list[float] = DEFAULT_GRID, *, refractory_s: float = REFRACTORY_S):
    """Score `probs` at each threshold against dev gold → one SweepRow per theta.
    Gold is built once per conversation and reused across thresholds."""
    from eval.data import conversation
    from eval.gold import events_for_conversation
    from eval.score import TaskScore, merge, score_task

    fps = probs.frame_rate_hz
    cached = [
        (entry, events_for_conversation(conversation(dataset, entry.conversation_id)))
        for entry in probs.probs
    ]
    rows: list[SweepRow] = []
    for theta in thetas:
        task, backchannel = TaskScore(), TaskScore()
        for entry, gold in cached:
            events = {
                1: commit_events(entry.speaker_1.prob, fps, theta, refractory_s=refractory_s),
                2: commit_events(entry.speaker_2.prob, fps, theta, refractory_s=refractory_s),
            }
            if probs.task == "eot":
                positives, negatives, excluded = (
                    gold.eot_positive_events, gold.eot_negative_spans, gold.eot_excluded,
                )
            else:
                positives, negatives, excluded = (
                    gold.int_positive_events, gold.int_negative_spans, gold.int_excluded,
                )
            merge(task, score_task(positives, negatives, events, excluded))
            # false-interruption rate = events firing on backchannel spans (the INT
            # task's negatives). For task="int" this coincides with fp_rate above.
            merge(backchannel, score_task([], gold.int_negative_spans, events, gold.int_excluded))
        latency = task.latency()
        rows.append(SweepRow(
            theta, task.recall, task.fp_rate, backchannel.fp_rate,
            latency.p10, latency.p50, latency.p90, task.tp, task.fp, task.fn,
        ))
    return rows


def operating_point(rows: list[SweepRow], *, fp_budget: float = 0.1) -> SweepRow | None:
    """Highest-recall theta with fp_rate ≤ budget — operate as aggressively as the
    false-positive budget allows. None if no threshold reaches the budget.

    Maximizing recall (not minimizing latency) avoids the degenerate near-silent
    point: a threshold that almost never fires has fp_rate ≈ 0 and an arbitrary
    latency, so a latency objective would reward recall ≈ 0. A boundary pick (edge
    of the theta grid) signals the probs are poorly calibrated — recall need not be
    monotone in theta under the rising-edge commit rule."""
    qualifying = [r for r in rows if r.fp_rate == r.fp_rate and r.fp_rate <= fp_budget]
    return max(qualifying, key=lambda r: r.recall) if qualifying else None


def main(
    probs_path: Annotated[
        Path, typer.Argument(help="probabilities JSON (probs-eot.json / probs-int.json)")
    ],
    fp_budget: Annotated[float, typer.Option(help="operating-point false-positive budget")] = 0.1,
) -> None:
    """Validate a probs file, sweep it over the dev gold, and print per-theta
    metrics + the operating point (highest recall at fp_rate ≤ budget). In-memory
    only — the figure is rendered straight from the probs files by
    data_analysis/plot_sweep.py."""
    from eval.data import DEV_DATASET, resolve_dataset

    probs = load_probs(probs_path)
    validate_probs(probs, load_durations("dev"))
    rows = sweep(probs, resolve_dataset(source=DEV_DATASET))
    for r in rows:
        typer.echo(
            f"θ={r.theta:.2f}  recall={r.recall:.3f}  fp={r.fp_rate:.3f}  "
            f"false_int={r.false_int_rate:.3f}  lat_p50={r.lat_p50:.0f}ms"
        )
    op = operating_point(rows, fp_budget=fp_budget)
    if op is not None:
        typer.secho(
            f"operating point: θ={op.theta:.2f}  recall={op.recall:.3f}  "
            f"fp_rate={op.fp_rate:.3f} ≤ {fp_budget}  lat_p50={op.lat_p50:.0f}ms",
            fg="green",
        )
    else:
        typer.secho(f"no threshold reaches fp_rate ≤ {fp_budget}", fg="yellow")


if __name__ == "__main__":
    typer.run(main)
