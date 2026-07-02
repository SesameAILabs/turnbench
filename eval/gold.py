"""Gold construction: raw 3-annotator SRTs -> the EOT / Interruption event
sets the scorer anchors its eval windows on (eval/score.py).

This code is the source of truth for the gold — no consensus artifact is
published. The dataset ships only raw annotations (the
`speaker_{1,2}_annotation_{a,b,c}` event-list columns alongside the audio) and
the scorer builds the gold from them at scoring time, so the gold at a given
commit is fully determined by this file.

Stage 1 — consensus. An event is consensus iff at least MIN_AGREEMENT of the
three annotators (a, b, c) emit the same canonical label with (start, end)
intervals agreeing within TIME_TOLERANCE_S on both endpoints; the gold boundary
is the median across the agreeing annotators (MIN_AGREEMENT = 2, a majority).
LABEL_MAP below is the authoritative fine→canonical mapping. Agreement is judged at
the granularity each task needs — the taxonomy is hierarchical, "Turn" being
the coarser level above the interruption labels:

    turn view  — agreement on "does this span CLAIM the floor?" (Turn and
                 floor-taking-Interruption fine labels are one class). Drives
                 turns, and therefore EOT positives and pauses.
    label view — agreement on the canonical label. Drives the INT task
                 (Interruption onsets; Backchannel/NonContent extents).

Spans where no majority forms in a view become that view's *excluded*
intervals — a dissenting annotator whom the majority outvoted is settled, not
excluded, so only genuine no-majority regions are masked: the scorer ignores
predictions inside them, and pauses stop at them (disputed is unknown, not
silent).

Stage 2 — floor construction. Positives are anchor *times* (a window around
them is searched for a predicted event). Negatives are *spans* — stretches
where firing is a false positive:

    EOT negative  — a mid-turn pause, scored only while it is believably a
                    quiet within-turn hold: from the segment end until the
                    speaker resumes, truncated at the first excluded interval
                    (either speaker), the speaker's own non-floor vocalisation
                    (a backchannel mid-"pause" means they are listening), or
                    an other-speaker Interruption. Dropped if contaminated
                    from the start. Other-speaker backchannels do NOT truncate
                    — they are evidence the speaker holds the floor.
    INT negative  — a Backchannel / NonContent event's own extent: the other
                    speaker made a sound but did not take the floor.

The annotation is VAD-segmented, so one speaker's turn is many short segments
separated by gaps. A segment-end is a *real EOT* only when the floor passes to
the other speaker — they speak next, or are already speaking as the segment
ends; otherwise it is a mid-turn pause.

`Laughter` events are currently ignored for the floor — the Turn ∪ Laughter
floor rule (trailing laughter holds the floor) is deferred; see eval/README.md.

CLI:
    python -m eval.gold stats    # per-conversation event-set counts (sanity check)
    python -m eval.gold export   # the gold as one JSON artifact, to stdout
"""

import bisect
import json
import subprocess
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from statistics import median

import typer

from eval.data import (
    ANNOTATORS,
    DEV_REVISION,
    SPEAKERS,
    Conversation,
    conversation,
    conversation_ids,
    resolve_dataset,
)

# Maps the fine-grained annotator labels to the canonical event taxonomy used
# for evaluation (canonical -> fine labels that collapse into it). Labels not
# listed are ignored when building consensus. Laughter and Awkward Silence are
# their own buckets: tt-benchmark's turn-region builder treats them specially
# (leading laughter doesn't claim the floor; awkward silence is a break, not a
# turn). EOT derivation here currently uses `Turn` only — the Turn ∪ Laughter
# floor rule is deferred (see eval/README.md).
LABEL_MAP: dict[str, tuple[str, ...]] = {
    "Turn": (
        "Normal Turn",
        "Regular Turn",  # defensive alias; occurs in neither released split
        "Strong Floor Hold",
        "Bounded Response",
        "Filler",
        "Overlap",
    ),
    "Laughter": ("Laughter",),
    "AwkwardSilence": ("Awkward Silence",),
    # An interruption means the other speaker was actually interrupted: the
    # floor changes hands. Only the floor-taking fine labels are scored as
    # Interruption; non-floor-taking attempts are their own class, which the
    # INT task excludes rather than scores (high-precision gold — at onset
    # the two are indistinguishable, so firing on an attempt is neither
    # rewarded nor penalised). This deliberately narrows tt-benchmark's
    # original any-vocalization definition.
    "Interruption": (
        "Floor-taking Competitive Interruption",
        "Floor-taking Cooperative Interruption",
    ),
    "NonFloorTakingInterruption": (
        "Non-floor Taking Competitive Interruption",
        "Non-floor Taking Cooperative Interruption",
    ),
    "Backchannel": (
        "Acknowledgement Backchannel",
        "Continuer Backchannel",
        "Reaction Backchannel",
    ),
    "NonContent": (
        "Non-Speech Noise",
        "Channel Bleed",
        "Speech, Non-Linguistic",
    ),
}
CANONICAL = {
    fine: canonical for canonical, fines in LABEL_MAP.items() for fine in fines
}

# The taxonomy is hierarchical for floor purposes: "Turn" is the coarser
# level, and floor-taking interruptions are turns at that level. The *turn
# view* asks the coarser question — "does this span CLAIM the floor?" — so a
# span labelled Normal Turn / Floor-taking Interruption / Overlap reaches a
# floor-claiming majority there even though the label view sees disagreement. The label view
# (CANONICAL) still serves the INT task, where that distinction is the point.
TURN_LABELS = LABEL_MAP["Turn"] + LABEL_MAP["Interruption"]
TURN_CANONICAL = {fine: "Turn" for fine in TURN_LABELS}

# Other-speaker sounds that are NOT a floor grab -> Interruption-task negatives.
INT_NEGATIVE_LABELS = ("Backchannel", "NonContent")
# Both interruption classes mark a barge-in when truncating pauses.
INTERRUPTION_LABELS = ("Interruption", "NonFloorTakingInterruption")

MIN_AGREEMENT = 2  # of 3 annotators an event needs (3 = unanimous, 2 = majority)
TIME_TOLERANCE_S = 0.2  # max disagreement on start OR end across annotators
OVERLAP_WINDOW_S = 0.5  # max start-time gap to match events across annotators

# The scoring windows around gold anchors (used by eval/score.py; carried in
# the exported artifact so external scorers don't duplicate them). TAU_PRE_S
# is a matching tolerance — the gold boundary is only annotation-exact — and
# TAU_MAX_S the latency deadline.
TAU_PRE_S = 0.25
TAU_MAX_S = 3.00

REPO_DIR = Path(__file__).parent.parent


@dataclass(frozen=True)
class ConsensusEvent:
    """One majority-agreed annotated segment."""

    speaker: int
    start: float
    end: float
    label: str


@dataclass(frozen=True)
class AnchorEvent:
    """A scoring anchor: a window centred at `time_s` on `speaker`'s channel."""

    speaker: int
    time_s: float


@dataclass(frozen=True)
class Interval:
    speaker: int
    start: float
    end: float


@dataclass(frozen=True)
class ConsensusViews:
    """The two consensus granularities."""

    turn_events: list[ConsensusEvent]  # majority floor-claiming spans (label "Turn")
    turn_excluded: list[Interval]  # no floor-level agreement
    events: list[ConsensusEvent]  # majority canonical-label events (label view)
    excluded: list[Interval]  # no fine-level agreement


@dataclass(frozen=True)
class ConversationEvents:
    eot_positive_events: list[AnchorEvent]  # real turn ends (floor handed over)
    eot_negative_spans: list[Interval]  # believable mid-turn pauses
    int_positive_events: list[AnchorEvent]  # interruptions (interrupter takes floor)
    int_negative_spans: list[Interval]  # backchannel / bleed extents
    eot_excluded: list[Interval]  # turn-view disputes; EOT predictions masked
    int_excluded: list[Interval]  # label-view disputes; INT predictions masked


# ---- stage 1: annotation tracks -> consensus events -------------------------


def map_events(
    events: list[tuple[float, float, str]],
    canonical: dict[str, str],
) -> list[tuple[float, float, str]]:
    """Drop unmapped fine labels; rewrite the rest to canonical labels."""
    return [
        (start, end, canonical[label])
        for start, end, label in events
        if label in canonical
    ]


def merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Merge overlapping (start, end) intervals; returns them sorted."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def find_consensus(
    per_annotator: dict[str, list[tuple[float, float, str]]], speaker: int
) -> tuple[list[ConsensusEvent], list[Interval]]:
    """MIN_AGREEMENT-of-3 consensus over one speaker's canonical events.

    Anchor on each annotator's events in turn: gather the nearest unused
    same-label event from each *later* annotator (within OVERLAP_WINDOW_S of the
    anchor's start), then accept the largest group including the anchor whose
    starts and ends each span at most TIME_TOLERANCE_S, provided it reaches
    MIN_AGREEMENT members. The gold boundary is the median across the group.

    A dissenting annotator's event is settled by the majority, so an unmatched
    event becomes an excluded interval ONLY where it overlaps no accepted
    consensus event — a genuine no-majority region. Disagreement the majority
    already resolved is not re-introduced as uncertainty; otherwise it would
    mask the very event it dissents from.
    """
    events = {annotator: per_annotator[annotator] for annotator in ANNOTATORS}
    used: dict[str, set[int]] = {annotator: set() for annotator in ANNOTATORS}
    consensus: list[ConsensusEvent] = []

    def nearest(target_start: float, label: str, annotator: str) -> int | None:
        best_index, best_distance = None, float("inf")
        for index, (start, _, candidate_label) in enumerate(events[annotator]):
            if index in used[annotator] or candidate_label != label:
                continue
            distance = abs(start - target_start)
            if distance < best_distance and distance <= OVERLAP_WINDOW_S:
                best_distance, best_index = distance, index
        return best_index

    def largest_agreeing(
        group: dict[str, tuple[int, float, float]], anchor: str
    ) -> tuple[str, ...]:
        """The biggest subset of `group` that includes `anchor` and whose
        starts and ends each span at most TIME_TOLERANCE_S."""
        members = list(group)
        for size in range(len(members), 0, -1):
            for subset in combinations(members, size):
                if anchor not in subset:
                    continue
                starts = [group[annotator][1] for annotator in subset]
                ends = [group[annotator][2] for annotator in subset]
                if (
                    max(starts) - min(starts) <= TIME_TOLERANCE_S
                    and max(ends) - min(ends) <= TIME_TOLERANCE_S
                ):
                    return subset
        return ()

    for anchor_position, anchor in enumerate(ANNOTATORS):
        for anchor_index, (start, end, label) in enumerate(events[anchor]):
            if anchor_index in used[anchor]:
                continue
            group = {anchor: (anchor_index, start, end)}
            for other in ANNOTATORS[anchor_position + 1 :]:
                match_index = nearest(start, label, other)
                if match_index is not None:
                    match_start, match_end, _ = events[other][match_index]
                    group[other] = (match_index, match_start, match_end)
            agreeing = largest_agreeing(group, anchor)
            if len(agreeing) < MIN_AGREEMENT:
                continue
            starts = [group[annotator][1] for annotator in agreeing]
            ends = [group[annotator][2] for annotator in agreeing]
            consensus.append(
                ConsensusEvent(
                    speaker, round(median(starts), 4), round(median(ends), 4), label
                )
            )
            for annotator in agreeing:
                used[annotator].add(group[annotator][0])

    consensus_spans = [(event.start, event.end) for event in consensus]
    unmatched = [
        (start, end)
        for annotator in ANNOTATORS
        for index, (start, end, _) in enumerate(events[annotator])
        if index not in used[annotator]
        and not any(
            span_start < end and start < span_end
            for span_start, span_end in consensus_spans
        )
    ]
    excluded = [
        Interval(speaker, round(start, 4), round(end, 4))
        for start, end in merge_intervals(unmatched)
    ]
    return consensus, excluded


def consensus_for_conversation(
    conv: Conversation,
    canonical: dict[str, str] = CANONICAL,
) -> tuple[list[ConsensusEvent], list[Interval]]:
    """Build the consensus events + excluded intervals for one conversation,
    judged at the granularity of `canonical` (the label view by default, the
    coarser turn view with TURN_CANONICAL).

    Reads the three annotator tracks per speaker from `conv.annotations`.
    Returns (consensus events across both speakers, excluded intervals).
    """
    events: list[ConsensusEvent] = []
    excluded: list[Interval] = []
    for speaker in SPEAKERS:
        per_annotator = {
            annotator: map_events(conv.annotations[(speaker, annotator)], canonical)
            for annotator in ANNOTATORS
        }
        speaker_events, speaker_excluded = find_consensus(per_annotator, speaker)
        events.extend(speaker_events)
        excluded.extend(speaker_excluded)
    return events, excluded


# ---- stage 2: consensus views -> scored event sets ---------------------------


def first_start_after(start_times: list[float], time_s: float) -> float:
    """Smallest start strictly greater than time_s, or +inf. `start_times` must be sorted."""
    index = bisect.bisect_right(start_times, time_s)
    return start_times[index] if index < len(start_times) else float("inf")


@dataclass(frozen=True)
class SpeakerTurns:
    """One speaker's turn-view `Turn` segments, sorted by start time."""

    speaker: int
    segments: list[ConsensusEvent]

    @property
    def start_times(self) -> list[float]:
        return [segment.start for segment in self.segments]


def collect_turns(events: list[ConsensusEvent], speaker: int) -> SpeakerTurns:
    segments = sorted(
        (
            event
            for event in events
            if event.label == "Turn" and event.speaker == speaker
        ),
        key=lambda event: event.start,
    )
    return SpeakerTurns(speaker, segments)


def truncated_pause(
    span: Interval, views: ConsensusViews, other_speaker: int
) -> Interval | None:
    """Cut the pause at the first evidence against "a quiet within-turn hold":
    an excluded interval in either view (disputed is unknown, not silent), the
    speaker's own non-floor vocalisation, or an other-speaker Interruption.
    Returns None when nothing believable remains."""
    cut_times = [span.end]
    for interval in views.turn_excluded + views.excluded:
        if interval.start < span.end and interval.end > span.start:
            cut_times.append(interval.start)
    for event in views.events:
        own_vocalisation = event.speaker == span.speaker and event.label != "Turn"
        other_barge_in = (
            event.speaker == other_speaker and event.label in INTERRUPTION_LABELS
        )
        if (own_vocalisation or other_barge_in) and span.start < event.start < span.end:
            cut_times.append(event.start)
    end = min(cut_times)
    if end <= span.start:
        return None
    return Interval(span.speaker, span.start, end)


def build_conversation_events(views: ConsensusViews) -> ConversationEvents:
    """Derive the EOT and Interruption event sets from the consensus views."""
    speaker1_turns = collect_turns(views.turn_events, 1)
    speaker2_turns = collect_turns(views.turn_events, 2)

    eot_positive_events: list[AnchorEvent] = []
    eot_negative_spans: list[Interval] = []
    for speaker_turns in [speaker1_turns, speaker2_turns]:
        other_speaker_turns = (
            speaker2_turns if speaker_turns.speaker == 1 else speaker1_turns
        )
        self_start_times = speaker_turns.start_times
        other_start_times = other_speaker_turns.start_times
        for segment in speaker_turns.segments:
            # An end strictly inside another of the speaker's own segments
            # (e.g. a floor-taking interruption within their longer turn) is
            # not a turn boundary at all.
            inside_own_segment = any(
                own is not segment and own.start < segment.end < own.end
                for own in speaker_turns.segments
            )
            if inside_own_segment:
                continue
            # A real EOT: the floor leaves this speaker — the other speaker
            # takes over before this speaker resumes, is already mid-turn as
            # this segment ends (an overlapping take-over), or nobody speaks
            # again. Otherwise it is a mid-turn pause.
            next_self = first_start_after(self_start_times, segment.end)
            next_other = first_start_after(other_start_times, segment.end)
            other_is_speaking = any(
                other.start < segment.end < other.end
                for other in other_speaker_turns.segments
            )
            if next_self == float("inf") or next_other < next_self or other_is_speaking:
                anchor = AnchorEvent(speaker_turns.speaker, segment.end)
                if anchor not in eot_positive_events:
                    eot_positive_events.append(anchor)
            else:
                pause = truncated_pause(
                    Interval(speaker_turns.speaker, segment.end, next_self),
                    views,
                    other_speaker_turns.speaker,
                )
                if pause is not None and pause not in eot_negative_spans:
                    eot_negative_spans.append(pause)

    int_positive_events = [
        AnchorEvent(event.speaker, event.start)
        for event in views.events
        if event.label == "Interruption"
    ]
    int_negative_spans = [
        Interval(event.speaker, event.start, event.end)
        for event in views.events
        if event.label in INT_NEGATIVE_LABELS
    ]
    # Consensus non-floor-taking attempts are neither positives nor
    # negatives: at onset they are indistinguishable from real interruptions,
    # so firing on one is neither rewarded nor penalised.
    int_excluded = views.excluded + [
        Interval(event.speaker, event.start, event.end)
        for event in views.events
        if event.label == "NonFloorTakingInterruption"
    ]

    return ConversationEvents(
        eot_positive_events,
        eot_negative_spans,
        int_positive_events,
        int_negative_spans,
        eot_excluded=views.turn_excluded,
        int_excluded=int_excluded,
    )


def events_for_conversation(conv: Conversation) -> ConversationEvents:
    """One conversation's annotation tracks -> the scored EOT / INT event sets.

    Consensus is built twice, at the granularity each task needs: the turn
    view (floor-claiming vs not) drives turns and EOT; the label view drives
    the INT task.
    """
    events, excluded = consensus_for_conversation(conv)
    turn_events, turn_excluded = consensus_for_conversation(
        conv, canonical=TURN_CANONICAL
    )
    return build_conversation_events(
        ConsensusViews(turn_events, turn_excluded, events, excluded)
    )


# ---- CLI ---------------------------------------------------------------------

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def stats() -> None:
    """Sanity-check the gold: per-conversation and aggregate event-set counts.
    Mid-turn-pause negatives should vastly outnumber real EOTs; both INT sets
    should be populated."""
    dataset = resolve_dataset(skip_audio=True)
    task_ids = conversation_ids(dataset)

    print(
        f"{'task':>6s} {'eot+':>5s} {'eot-':>5s} {'int+':>5s} {'int-':>5s} {'excl':>5s}"
    )
    totals = [0, 0, 0, 0, 0]
    for task_id in task_ids:
        conversation_events = events_for_conversation(conversation(dataset, task_id))
        counts = [
            len(conversation_events.eot_positive_events),
            len(conversation_events.eot_negative_spans),
            len(conversation_events.int_positive_events),
            len(conversation_events.int_negative_spans),
            len(conversation_events.eot_excluded)
            + len(conversation_events.int_excluded),
        ]
        totals = [sum_pair[0] + sum_pair[1] for sum_pair in zip(totals, counts)]
        print(f"{task_id:>6s} " + " ".join(f"{count:>5d}" for count in counts))
    print("-" * 38)
    print(f"{'TOTAL':>6s} " + " ".join(f"{count:>5d}" for count in totals))
    print(f"\nEOT negatives / positives ratio: {totals[1] / max(totals[0], 1):.1f}x")


def scorer_sha() -> str:
    """Short commit SHA of this scorer, with -dirty when the tree has edits."""
    return subprocess.run(
        ["git", "describe", "--always", "--dirty"],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@app.command()
def export() -> None:
    """Dump the gold event sets as one JSON artifact, for scorers that cannot
    run this module (e.g. the in-browser dev scorer on the leaderboard site).

    The artifact is a derived cache — code stays the source of truth;
    regenerate after any gold change. Each conversation is ConversationEvents
    serialised verbatim plus the audio duration; the artifact is stamped with
    the scorer commit and dataset revision that fully determine it, so
    consumers never read audio, SRTs, or this repo's constants.
    """
    dataset = resolve_dataset(skip_audio=True)
    conversations = {}
    for task_id in conversation_ids(dataset):
        conv = conversation(dataset, task_id)
        conversations[task_id] = {
            "duration_s": conv.duration_s,
            **asdict(events_for_conversation(conv)),
        }
    artifact = {
        "scorer_sha": scorer_sha(),
        "dataset_revision": DEV_REVISION,
        "tau_pre_s": TAU_PRE_S,
        "tau_max_s": TAU_MAX_S,
        "conversations": conversations,
    }
    print(json.dumps(artifact))


if __name__ == "__main__":
    app()
