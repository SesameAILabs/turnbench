"""Gold construction tests: consensus (SRTs -> 3/3-agreed events) and floor
construction (consensus views -> scored event sets).
Run: uv run pytest eval/gold_test.py"""

from pathlib import Path

from eval.gold import (
    TURN_CANONICAL,
    AnchorEvent,
    ConsensusEvent,
    ConsensusViews,
    Interval,
    build_conversation_events,
    consensus_for_conversation,
    events_for_conversation,
    parse_srt,
)


def srt(*entries: tuple[float, float, str]) -> str:
    """Render (start_s, end_s, fine_label) entries as SRT text."""

    def timestamp(seconds: float) -> str:
        millis = round(seconds * 1000)
        return (
            f"{millis // 3_600_000:02d}:{millis // 60_000 % 60:02d}:"
            f"{millis // 1000 % 60:02d},{millis % 1000:03d}"
        )

    return "\n\n".join(
        f"{i}\n{timestamp(start)} --> {timestamp(end)}\n[{label}] ..."
        for i, (start, end, label) in enumerate(entries, 1)
    )


def write_conversation(
    directory: Path, speaker1: dict[str, str], speaker2: dict[str, str] | None = None
) -> Path:
    """Write per-annotator SRT texts for both speakers; missing ones are empty."""
    for speaker, annotations in ((1, speaker1), (2, speaker2 or {})):
        for annotator in ("a", "b", "c"):
            path = directory / f"speaker_{speaker}_annotation_{annotator}.srt"
            path.write_text(annotations.get(annotator, ""))
    return directory


# ---- consensus: SRTs -> 3/3-agreed events ------------------------------------


def test_parse_srt_reads_times_and_fine_label(tmp_path):
    path = tmp_path / "x.srt"
    path.write_text("1\n00:01:02,500 --> 00:01:03,250\n[Normal Turn] hello\n")

    assert parse_srt(path) == [(62.5, 63.25, "Normal Turn")]


def test_unanimous_event_becomes_consensus_with_median_endpoints(tmp_path):
    # Same canonical label, endpoints within tolerance -> median gold boundary.
    conversation = write_conversation(
        tmp_path,
        {
            "a": srt((1.00, 2.00, "Normal Turn")),
            "b": srt((1.10, 2.10, "Regular Turn")),
            "c": srt((1.05, 2.05, "Strong Floor Hold")),
        },
    )

    events, excluded = consensus_for_conversation(conversation)

    assert events == [ConsensusEvent(1, 1.05, 2.05, "Turn")]
    assert excluded == []


def test_endpoint_spread_beyond_tolerance_is_excluded(tmp_path):
    # Starts agree but ends spread > 0.2 s -> no consensus; spans are excluded.
    conversation = write_conversation(
        tmp_path,
        {
            "a": srt((1.0, 2.0, "Normal Turn")),
            "b": srt((1.0, 2.5, "Normal Turn")),
            "c": srt((1.0, 2.0, "Normal Turn")),
        },
    )

    events, excluded = consensus_for_conversation(conversation)

    assert events == []
    assert len(excluded) == 1  # the three spans merge into one interval
    assert (excluded[0].start, excluded[0].end) == (1.0, 2.5)


def test_label_disagreement_is_excluded(tmp_path):
    # Two annotators say Backchannel, one says Interruption -> not 3/3.
    conversation = write_conversation(
        tmp_path,
        {
            "a": srt((1.0, 1.5, "Acknowledgement Backchannel")),
            "b": srt((1.0, 1.5, "Continuer Backchannel")),
            "c": srt((1.0, 1.5, "Floor-taking Competitive Interruption")),
        },
    )

    events, excluded = consensus_for_conversation(conversation)

    assert events == []
    assert len(excluded) == 1


def test_turn_view_unifies_turn_and_floor_taking_labels(tmp_path):
    # Identical extents, labels split across Normal Turn / Floor-taking
    # Interruption / Overlap: the label view disagrees (Turn vs Interruption),
    # but all three labels claim the floor — the turn view is unanimous.
    conversation = write_conversation(
        tmp_path,
        {
            "a": srt((1.0, 5.0, "Normal Turn")),
            "b": srt((1.0, 5.0, "Floor-taking Cooperative Interruption")),
            "c": srt((1.0, 5.0, "Overlap")),
        },
    )

    label_events, label_excluded = consensus_for_conversation(conversation)
    turn_events, turn_excluded = consensus_for_conversation(
        conversation, canonical=TURN_CANONICAL
    )

    assert label_events == [] and len(label_excluded) == 1
    assert turn_events == [ConsensusEvent(1, 1.0, 5.0, "Turn")]
    assert turn_excluded == []


def test_turn_view_rejects_non_floor_taking_dispute(tmp_path):
    # "Non-floor Taking" is a genuine disagreement about the floor itself:
    # no consensus in either view.
    conversation = write_conversation(
        tmp_path,
        {
            "a": srt((1.0, 5.0, "Normal Turn")),
            "b": srt((1.0, 5.0, "Normal Turn")),
            "c": srt((1.0, 5.0, "Non-floor Taking Cooperative Interruption")),
        },
    )

    turn_events, turn_excluded = consensus_for_conversation(
        conversation, canonical=TURN_CANONICAL
    )

    assert turn_events == []
    assert len(turn_excluded) == 1


def test_interruption_consensus_requires_floor_taking_agreement(tmp_path):
    # Interruption is floor-taking by definition ("interrupted" means the
    # other speaker stops). 3/3 floor-taking -> a consensus Interruption;
    # 3/3 non-floor-taking -> its own class; subtype dispute -> excluded.
    for name in ("ft", "nft", "mixed"):
        (tmp_path / name).mkdir()
    floor_taking = write_conversation(
        tmp_path / "ft",
        {
            "a": srt((1.0, 2.0, "Floor-taking Competitive Interruption")),
            "b": srt((1.0, 2.0, "Floor-taking Cooperative Interruption")),
            "c": srt((1.0, 2.0, "Floor-taking Competitive Interruption")),
        },
    )
    non_floor_taking = write_conversation(
        tmp_path / "nft",
        {
            "a": srt((1.0, 2.0, "Non-floor Taking Competitive Interruption")),
            "b": srt((1.0, 2.0, "Non-floor Taking Cooperative Interruption")),
            "c": srt((1.0, 2.0, "Non-floor Taking Competitive Interruption")),
        },
    )
    disputed = write_conversation(
        tmp_path / "mixed",
        {
            "a": srt((1.0, 2.0, "Floor-taking Competitive Interruption")),
            "b": srt((1.0, 2.0, "Non-floor Taking Competitive Interruption")),
            "c": srt((1.0, 2.0, "Floor-taking Competitive Interruption")),
        },
    )

    ft_events, ft_excluded = consensus_for_conversation(floor_taking)
    nft_events, nft_excluded = consensus_for_conversation(non_floor_taking)
    mixed_events, mixed_excluded = consensus_for_conversation(disputed)

    assert ft_events == [ConsensusEvent(1, 1.0, 2.0, "Interruption")]
    assert nft_events == [ConsensusEvent(1, 1.0, 2.0, "NonFloorTakingInterruption")]
    assert mixed_events == [] and len(mixed_excluded) == 1


def test_unmapped_labels_are_ignored(tmp_path):
    # A fine label outside LABEL_MAP contributes nothing — not even exclusion.
    conversation = write_conversation(
        tmp_path,
        {
            "a": srt((1.0, 2.0, "Some Unknown Label")),
            "b": srt((1.0, 2.0, "Some Unknown Label")),
            "c": srt((1.0, 2.0, "Some Unknown Label")),
        },
    )

    events, excluded = consensus_for_conversation(conversation)

    assert events == []
    assert excluded == []


# ---- floor construction: consensus views -> scored event sets ----------------


def turn(speaker: int, start: float, end: float) -> ConsensusEvent:
    """One VAD-segmented Turn annotation for speaker at [start, end]."""
    return ConsensusEvent(speaker, start, end, "Turn")


def views(
    events: list[ConsensusEvent] | None = None,
    excluded: list[Interval] | None = None,
    turn_events: list[ConsensusEvent] | None = None,
    turn_excluded: list[Interval] | None = None,
) -> ConsensusViews:
    """Consensus views for tests. Unless given, the turn view mirrors the
    label view's Turn events (true whenever no interruption labels are in play)."""
    events = events or []
    return ConsensusViews(
        turn_events=(
            turn_events
            if turn_events is not None
            else [event for event in events if event.label == "Turn"]
        ),
        turn_excluded=turn_excluded or [],
        events=events,
        excluded=excluded or [],
    )


def test_mid_turn_pause_is_a_negative_span_until_resumption():
    # Speaker 1: [0–2] …gap… [2.5–5] — same speaker resumes after the gap
    conversation_events = build_conversation_events(
        views(events=[turn(1, 0.0, 2.0), turn(1, 2.5, 5.0)])
    )

    assert conversation_events.eot_negative_spans == [Interval(1, 2.0, 2.5)]
    assert AnchorEvent(1, 2.0) not in conversation_events.eot_positive_events


def test_handover_is_eot_positive():
    # Speaker 1 ends at 2.0; speaker 2 takes over at 2.2 before speaker 1 resumes
    conversation_events = build_conversation_events(
        views(events=[turn(1, 0.0, 2.0), turn(2, 2.2, 5.0)])
    )

    assert AnchorEvent(1, 2.0) in conversation_events.eot_positive_events
    assert conversation_events.eot_negative_spans == []


def test_last_segment_is_eot_positive():
    # Speaker 1's final Turn segment — nobody speaks again
    conversation_events = build_conversation_events(views(events=[turn(1, 0.0, 2.0)]))

    assert AnchorEvent(1, 2.0) in conversation_events.eot_positive_events
    assert conversation_events.eot_negative_spans == []


def test_overlapping_takeover_is_eot_positive():
    # Speaker 2 starts at 1.7 — BEFORE speaker 1's segment ends at 2.0 — and is
    # still talking at the boundary: the floor passed, even though speaker 1
    # resumes later (an overlapping take-over, not a pause).
    conversation_events = build_conversation_events(
        views(events=[turn(1, 0.0, 2.0), turn(2, 1.7, 5.0), turn(1, 6.0, 8.0)])
    )

    assert AnchorEvent(1, 2.0) in conversation_events.eot_positive_events
    assert all(span.speaker != 1 or span.start != 2.0
               for span in conversation_events.eot_negative_spans)


def test_simultaneous_resume_is_pause_not_handover():
    # At speaker 1's segment end (2.0), both speakers next start at 2.5 — not a handover
    conversation_events = build_conversation_events(
        views(events=[turn(1, 0.0, 2.0), turn(1, 2.5, 5.0), turn(2, 2.5, 4.0)])
    )

    assert Interval(1, 2.0, 2.5) in conversation_events.eot_negative_spans
    assert AnchorEvent(1, 2.0) not in conversation_events.eot_positive_events


def test_classifies_both_speakers():
    # Speaker 1 handover at 2.0; speaker 2 mid-turn pause at 4.0
    conversation_events = build_conversation_events(
        views(events=[turn(1, 0.0, 2.0), turn(2, 2.2, 4.0), turn(2, 4.5, 7.0)])
    )

    assert AnchorEvent(1, 2.0) in conversation_events.eot_positive_events
    assert Interval(2, 4.0, 4.5) in conversation_events.eot_negative_spans
    assert AnchorEvent(2, 7.0) in conversation_events.eot_positive_events


def test_int_negative_is_the_event_extent():
    # A backchannel's whole span is the INT negative, not just its onset
    conversation_events = build_conversation_events(
        views(events=[turn(1, 0.0, 5.0), ConsensusEvent(2, 1.5, 1.9, "Backchannel")])
    )

    assert conversation_events.int_negative_spans == [Interval(2, 1.5, 1.9)]


# ---- the problems found by inspection (conversation 41 and friends) --------


def test_label_disputed_handover_is_an_eot_positive(tmp_path):
    # The conversation-41 10:03 case: speaker 2 takes over while speaker 1
    # finishes; all three annotators agree on the extent but split the label
    # across Normal Turn / Floor-taking Interruption / Overlap. At the floor
    # level that is unanimous "speaker 2 claims the floor" — speaker 1's
    # segment end must be an EOT positive, not the start of a phantom pause.
    write_conversation(
        tmp_path,
        {annotator: srt((0.0, 2.0, "Normal Turn"), (8.0, 9.0, "Normal Turn"))
         for annotator in ("a", "b", "c")},
        {
            "a": srt((1.7, 6.0, "Normal Turn")),
            "b": srt((1.7, 6.0, "Floor-taking Cooperative Interruption")),
            "c": srt((1.7, 6.0, "Overlap")),
        },
    )

    conversation_events = events_for_conversation(tmp_path)

    assert AnchorEvent(1, 2.0) in conversation_events.eot_positive_events
    assert not any(span.speaker == 1 and span.start == 2.0
                   for span in conversation_events.eot_negative_spans)


def test_non_floor_taking_dispute_stays_unscored(tmp_path):
    # The conversation-41 4:35 case: one annotator says NON-floor-taking — a
    # genuine disagreement about the floor itself. No turn is minted, and no
    # pause is scored over the disputed stretch either.
    write_conversation(
        tmp_path,
        {annotator: srt((0.0, 2.0, "Normal Turn"), (8.0, 9.0, "Normal Turn"))
         for annotator in ("a", "b", "c")},
        {
            "a": srt((1.7, 6.0, "Normal Turn")),
            "b": srt((1.7, 6.0, "Normal Turn")),
            "c": srt((1.7, 6.0, "Non-floor Taking Cooperative Interruption")),
        },
    )

    conversation_events = events_for_conversation(tmp_path)

    assert AnchorEvent(1, 2.0) not in conversation_events.eot_positive_events
    assert not any(span.speaker == 1 and span.start == 2.0
                   for span in conversation_events.eot_negative_spans)


def test_pause_truncated_at_excluded_interval():
    # Pause [2.0–6.0], but consensus failed somewhere inside (either speaker):
    # from there on we know nothing — stop scoring the pause at 3.0.
    conversation_events = build_conversation_events(
        views(
            events=[turn(1, 0.0, 2.0), turn(1, 6.0, 8.0)],
            excluded=[Interval(2, 3.0, 4.0)],
        )
    )

    assert conversation_events.eot_negative_spans == [Interval(1, 2.0, 3.0)]


def test_pause_truncated_at_own_vocalisation():
    # The "pausing" speaker backchannels at 3.5 — they are listening, not
    # holding the floor. The pause is only believable up to that point.
    conversation_events = build_conversation_events(
        views(
            events=[
                turn(1, 0.0, 2.0),
                ConsensusEvent(1, 3.5, 3.8, "Backchannel"),
                turn(1, 6.0, 8.0),
            ],
        )
    )

    assert conversation_events.eot_negative_spans == [Interval(1, 2.0, 3.5)]


def test_pause_truncated_at_other_speaker_interruption():
    # The other speaker barges in mid-pause: the floor is contested from there.
    conversation_events = build_conversation_events(
        views(
            events=[
                turn(1, 0.0, 2.0),
                ConsensusEvent(2, 3.2, 3.6, "Interruption"),
                turn(1, 6.0, 8.0),
            ],
        )
    )

    assert conversation_events.eot_negative_spans == [Interval(1, 2.0, 3.2)]


def test_pause_dropped_when_contaminated_from_the_start():
    # Consensus failure already in progress as the segment ends: there is no
    # clean stretch at all, so no negative is scored.
    conversation_events = build_conversation_events(
        views(
            events=[turn(1, 0.0, 2.0), turn(1, 6.0, 8.0)],
            excluded=[Interval(2, 1.5, 4.0)],
        )
    )

    assert conversation_events.eot_negative_spans == []
    assert AnchorEvent(1, 2.0) not in conversation_events.eot_positive_events


def test_other_speaker_backchannel_does_not_truncate():
    # Someone going "mm-hm" during your pause is evidence you DO hold the
    # floor — the pause stays intact.
    conversation_events = build_conversation_events(
        views(
            events=[
                turn(1, 0.0, 2.0),
                ConsensusEvent(2, 3.0, 3.3, "Backchannel"),
                turn(1, 6.0, 8.0),
            ],
        )
    )

    assert conversation_events.eot_negative_spans == [Interval(1, 2.0, 6.0)]


def test_segment_end_inside_own_floor_segment_is_not_classified():
    # The turn view can contain overlapping own segments (a floor-taking
    # interruption inside one's own longer turn): an end strictly inside
    # another own segment is not a turn boundary at all.
    conversation_events = build_conversation_events(
        views(events=[turn(2, 3.5, 5.0), turn(2, 4.5, 4.8)])
    )

    assert AnchorEvent(2, 4.8) not in conversation_events.eot_positive_events
    assert AnchorEvent(2, 5.0) in conversation_events.eot_positive_events
    assert conversation_events.eot_negative_spans == []


def test_int_positive_requires_floor_taking_consensus():
    # A 3/3 floor-taking Interruption is an INT positive; a 3/3
    # NON-floor-taking one is neither a positive nor a negative — its extent
    # is excluded for the INT task (high-precision gold: only events where
    # the other speaker was actually interrupted are scored).
    conversation_events = build_conversation_events(
        views(
            events=[
                turn(1, 0.0, 5.0),
                ConsensusEvent(2, 1.5, 1.9, "Interruption"),
                ConsensusEvent(2, 3.0, 3.4, "NonFloorTakingInterruption"),
            ],
        )
    )

    assert conversation_events.int_positive_events == [AnchorEvent(2, 1.5)]
    assert conversation_events.int_negative_spans == []
    assert Interval(2, 3.0, 3.4) in conversation_events.int_excluded
