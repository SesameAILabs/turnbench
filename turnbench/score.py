"""Binary, event-anchored scoring of a predictions JSON against the gold.

A submission declares, per conversation and per speaker, the times (s) at which
it detects EOTs (that speaker's turn ending) and interruptions (that speaker
taking the floor from the other). Timestamps are commit times — the time by
which all audio the decision depended on has been heard. Predictions are read
only inside the scored regions from turnbench/gold.py; everywhere else they are
invisible (neither rewarded nor penalised).

Positive events at time t are searched in [t - TAU_PRE_S, t + TAU_MAX_S], with
the upper end clamped to this speaker's next event of the same task (a fire
past that point belongs to the next event, not this one, so the window must not
reach in and claim it). TAU_PRE_S is a matching tolerance (the gold boundary is
only annotation-exact), TAU_MAX_S the latency deadline. Each speaker is scored
on their own channel, so only same-speaker anchors bound the window. Matching is
one-to-one: positives are processed in time order and each claims the earliest
unclaimed prediction in its window, so one prediction can never satisfy two
events (TP, latency = t_pred - t) and a burst in one window is one detection.

Negative spans (mid-turn pauses, backchannels) are charged the same tolerance:
a prediction in [start - TAU_PRE_S, end] is a false positive — at most one per
span. Predictions claimed by a positive, or inside any positive window, are
exempt: a correct detection is never also an FP. Speculation is therefore
symmetric — firing just before a segment end is rewarded iff the turn really
ends there, and penalised iff the speaker was only pausing.

Metrics per task:

    recall  = TP / (TP + FN)
    fp_rate = FP / (FP + TN)      # negative spans that fired / all negatives
    latency p10/p50/p90 over TPs

Ranking: keep models with fp_rate <= budget AND recall >= floor, rank by latency.

Usage:
    python -m turnbench.score predictions.json

Gold is built on the fly from the raw annotator tracks alongside the audio
(turnbench/gold.py) — that code is the source of truth for the gold. Data comes
straight from HuggingFace (cached locally; see turnbench/data.py); the eval server
points the same scorer at the private test set with `--dataset <private repo>`.
"""

import bisect
import math
from dataclasses import dataclass, field
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from turnbench.data import (
    DEV_DATASET,
    Conversation,
    Dataset,
    conversation,
    conversation_ids,
    resolve_dataset,
)
from turnbench.gold import (
    TAU_MAX_S,
    TAU_PRE_S,
    AnchorEvent,
    Interval,
    events_for_conversation,
)
from turnbench.submission import (
    ConversationPrediction,
    Submission,
    load_submission,
    validate_coverage,
    validate_event_times,
)


@dataclass(frozen=True)
class Latency:
    p10: float
    p50: float
    p90: float


@dataclass
class TaskScore:
    tp: int = 0
    fn: int = 0
    fp: int = 0
    tn: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else float("nan")

    @property
    def fp_rate(self) -> float:
        denominator = self.fp + self.tn
        return self.fp / denominator if denominator else float("nan")

    def latency(self) -> Latency:
        if not self.latencies_ms:
            return Latency(float("nan"), float("nan"), float("nan"))
        latencies = np.asarray(self.latencies_ms, dtype=np.float64)
        return Latency(*(float(np.percentile(latencies, p)) for p in (10, 50, 90)))


@dataclass(frozen=True)
class ConversationScores:
    task_eot: TaskScore
    task_int: TaskScore


def scoreable_times(
    predicted_times: list[float], excluded: list[Interval]
) -> list[float]:
    """Drop predicted event times inside excluded intervals (no reliable gold
    there — neither rewarded nor penalised). `predicted_times` must be sorted
    ascending; returns a sorted list."""
    assert all(
        earlier <= later
        for earlier, later in zip(predicted_times, predicted_times[1:])
    ), f"predicted event times must be sorted ascending, got {predicted_times!r}"
    return [
        time_s
        for time_s in predicted_times
        if not any(
            interval.start <= time_s <= interval.end for interval in excluded
        )
    ]


class PredictionLedger:
    """One speaker's scoreable predicted times with one-to-one claiming."""

    def __init__(self, times: list[float]) -> None:
        self.times = times
        self.claimed: set[int] = set()
        self.positive_windows: list[tuple[float, float]] = []

    def unclaimed_in(self, window_start_s: float, window_end_s: float):
        index = bisect.bisect_left(self.times, window_start_s)
        while index < len(self.times) and self.times[index] <= window_end_s:
            if index not in self.claimed:
                yield index
            index += 1

    def claim_first_in(self, window_start_s: float, window_end_s: float) -> float | None:
        """Claim the earliest unclaimed prediction in the window, if any."""
        for index in self.unclaimed_in(window_start_s, window_end_s):
            self.claimed.add(index)
            return self.times[index]
        return None

    def fires_in(self, window_start_s: float, window_end_s: float) -> bool:
        """Any unclaimed prediction in the window that is outside every
        positive window? (Correct detections are never also FPs.)"""
        return any(
            not any(
                start <= self.times[index] <= end
                for start, end in self.positive_windows
            )
            for index in self.unclaimed_in(window_start_s, window_end_s)
        )


def score_task(
    positive_events: list[AnchorEvent],
    negative_spans: list[Interval],
    predicted_times_by_speaker: dict[int, list[float]],
    excluded: list[Interval],
    *,
    tau_pre_s: float = TAU_PRE_S,
    tau_max_s: float = TAU_MAX_S,
) -> TaskScore:
    """Score one task's predicted event times against its gold event sets.

    `predicted_times_by_speaker` maps speaker (1 | 2) to that speaker's sorted
    predicted times (s) for this task: EOT times on the speaker whose turn ends,
    interruption times on the interrupter.
    """
    predictions_by_speaker: dict[int, PredictionLedger] = {}

    def predictions_for(speaker: int) -> PredictionLedger:
        if speaker not in predictions_by_speaker:
            excluded_for_speaker = [
                interval for interval in excluded if interval.speaker == speaker
            ]
            predictions_by_speaker[speaker] = PredictionLedger(
                scoreable_times(
                    predicted_times_by_speaker[speaker], excluded_for_speaker
                )
            )
        return predictions_by_speaker[speaker]

    # The latency deadline is capped at this speaker's *next* event of the same
    # task: past it, a prediction belongs to that event, not this one, so the
    # window must not reach into it and claim it. Each speaker's channel is
    # scored independently, so only same-speaker anchors bound the window — the
    # other speaker's turns never shrink it. (Capping the window this way leaves
    # the FP-exemption union unchanged, so FP/TN are identical to an uncapped
    # window; only TP/FN attribution between adjacent same-speaker events moves.)
    next_event_by_speaker: dict[int, list[float]] = {}
    for event in positive_events:
        next_event_by_speaker.setdefault(event.speaker, []).append(event.time_s)
    for times in next_event_by_speaker.values():
        times.sort()

    def next_event_after(speaker: int, time_s: float) -> float:
        anchors = next_event_by_speaker[speaker]
        index = bisect.bisect_right(anchors, time_s)
        return anchors[index] if index < len(anchors) else float("inf")

    scores = TaskScore()
    # Positives first, in time order: each claims the earliest unclaimed
    # prediction in its window, and its window exempts predictions from FP.
    for event in sorted(positive_events, key=lambda event: event.time_s):
        window_start_s = event.time_s - tau_pre_s
        window_end_s = min(
            event.time_s + tau_max_s, next_event_after(event.speaker, event.time_s)
        )
        predictions = predictions_for(event.speaker)
        predictions.positive_windows.append((window_start_s, window_end_s))
        matching_time = predictions.claim_first_in(window_start_s, window_end_s)
        if matching_time is None:
            scores.fn += 1
        else:
            scores.tp += 1
            scores.latencies_ms.append((matching_time - event.time_s) * 1000.0)
    for span in negative_spans:
        if predictions_for(span.speaker).fires_in(span.start - tau_pre_s, span.end):
            scores.fp += 1
        else:
            scores.tn += 1
    return scores


def merge(accumulator: TaskScore, addend: TaskScore) -> None:
    accumulator.tp += addend.tp
    accumulator.fn += addend.fn
    accumulator.fp += addend.fp
    accumulator.tn += addend.tn
    accumulator.latencies_ms.extend(addend.latencies_ms)


def qualifies(score: TaskScore, *, fp_budget: float, recall_floor: float) -> bool:
    return score.fp_rate <= fp_budget and score.recall >= recall_floor


def rank_key(score: TaskScore) -> float:
    """Sort key for ranking qualifiers: lower median latency is better."""
    return score.latency().p50


# ---- CLI: score a predictions file over the dataset ---------------------------


def score_conversation(
    prediction: ConversationPrediction, conv: Conversation
) -> ConversationScores:
    validate_event_times(prediction, conv.duration_s)
    conversation_events = events_for_conversation(conv)
    return ConversationScores(
        task_eot=score_task(
            conversation_events.eot_positive_events,
            conversation_events.eot_negative_spans,
            {1: prediction.speaker_1.eot, 2: prediction.speaker_2.eot},
            conversation_events.eot_excluded,
        ),
        task_int=score_task(
            conversation_events.int_positive_events,
            conversation_events.int_negative_spans,
            {1: prediction.speaker_1.interruption, 2: prediction.speaker_2.interruption},
            conversation_events.int_excluded,
        ),
    )


def iter_conversation_scores(
    submission: Submission, dataset: Dataset
) -> Iterator[tuple[str, ConversationScores]]:
    """Score each conversation in `dataset`, yielding (task_id, scores) in order.

    Validates coverage first (every conversation present exactly once). The
    shared scoring loop behind both the CLI and `score_submission`."""
    task_ids = conversation_ids(dataset)
    validate_coverage(submission, task_ids)
    by_conversation = submission.by_conversation()
    for task_id in task_ids:
        yield task_id, score_conversation(
            by_conversation[task_id], conversation(dataset, task_id)
        )


def score_submission(submission: Submission, dataset: Dataset) -> ConversationScores:
    """Aggregate EOT/INT scores for an in-memory submission of discrete events.

    The programmatic entry point: build a Submission of committed event times and
    get back the aggregate scores, no JSON file involved. Submitters call this in
    their own threshold loop to score each operating point they want to try."""
    totals = ConversationScores(TaskScore(), TaskScore())
    for _task_id, scores in iter_conversation_scores(submission, dataset):
        merge(totals.task_eot, scores.task_eot)
        merge(totals.task_int, scores.task_int)
    return totals


def rate(value: float) -> str:
    return "—" if math.isnan(value) else f"{value:.3f}"


def latencies(score: TaskScore) -> str:
    latency = score.latency()
    if math.isnan(latency.p50):
        return "—"
    return f"{latency.p10:.0f}/{latency.p50:.0f}/{latency.p90:.0f}"


def task_cells(score: TaskScore) -> tuple[str, str, str]:
    return rate(score.recall), rate(score.fp_rate), latencies(score)


def main(
    predictions: Annotated[
        Path, typer.Argument(help="predictions JSON (see README.md)")
    ],
    source: Annotated[
        str,
        typer.Option("--dataset", help="HF dataset repo id, or a local directory of parquet shards"),
    ] = DEV_DATASET,
    revision: Annotated[
        str | None,
        typer.Option(help="HF dataset revision (defaults to the pinned dev revision)"),
    ] = None,
) -> None:
    submission = load_submission(predictions)
    dataset = resolve_dataset(source=source, revision=revision, skip_audio=True)
    task_ids = conversation_ids(dataset)

    console = Console()
    progress = Table(title=f"{predictions} — per conversation", title_justify="left")
    progress.add_column("convo", justify="right")
    for task_name in ("EOT", "INT"):
        progress.add_column(f"{task_name} recall", justify="right")
        progress.add_column(f"{task_name} fp", justify="right")
        progress.add_column(f"{task_name} lat p10/50/90", justify="right")

    spinner = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )
    job = spinner.add_task("", total=len(task_ids))

    totals = ConversationScores(TaskScore(), TaskScore())
    with Live(Group(progress, spinner), console=console) as live:
        for task_id, conversation_scores in iter_conversation_scores(submission, dataset):
            spinner.update(job, description=f"scoring convo {task_id}")
            merge(totals.task_eot, conversation_scores.task_eot)
            merge(totals.task_int, conversation_scores.task_int)
            progress.add_row(
                task_id,
                *task_cells(conversation_scores.task_eot),
                *task_cells(conversation_scores.task_int),
            )
            spinner.advance(job)
        live.update(progress)  # final render: table only, spinner gone

    aggregate = Table(
        title=f"{predictions} — aggregate over {len(task_ids)} conversations",
        title_justify="left",
    )
    for column in ("task", "recall", "fp_rate", "lat ms p10/50/90", "tp", "fn", "fp", "tn"):
        aggregate.add_column(column, justify="right")
    for task_name, score in (("EOT", totals.task_eot), ("INT", totals.task_int)):
        aggregate.add_row(
            f"[bold]{task_name}[/]",
            *task_cells(score),
            str(score.tp), str(score.fn), str(score.fp), str(score.tn),
        )
    console.print(aggregate)


if __name__ == "__main__":
    typer.run(main)
