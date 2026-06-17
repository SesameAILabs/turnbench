"""Scorer tests: the event-matching rules for EOT and INT, plus end-to-end —
a synthetic conversation (3/3-unanimous annotator tracks) scored against
predictions derived from the gold itself, which must be perfect (recall 1,
fp_rate 0, latency 0) even with a pause directly adjacent to a real EOT.
Run: uv run pytest eval/score_test.py
"""

import pytest
from pytest import approx

from eval.data import ANNOTATORS, Annotation, Conversation
from eval.gold import AnchorEvent, Interval
from eval.gold_test import conv
from eval.score import (
    TaskScore,
    qualifies,
    rank_key,
    score_conversation,
    score_task,
)
from eval.submission import ConversationPrediction, SpeakerEvents


def anchor_event(speaker: int, time_s: float) -> AnchorEvent:
    """Scoring anchor at time_s on a speaker's channel."""
    return AnchorEvent(speaker, time_s)


def pause(speaker: int, start: float, end: float) -> Interval:
    """A negative span (mid-turn pause / backchannel extent) on a speaker."""
    return Interval(speaker, start, end)


def score(
    *,
    positive_events: list[AnchorEvent] | None = None,
    negative_spans: list[Interval] | None = None,
    speaker_1_times: list[float] | None = None,
    speaker_2_times: list[float] | None = None,
    excluded: list[Interval] | None = None,
) -> TaskScore:
    """Score one task's predicted event times against its gold event sets."""
    return score_task(
        positive_events or [],
        negative_spans or [],
        {1: speaker_1_times or [], 2: speaker_2_times or []},
        excluded or [],
    )


def test_true_positive_and_latency():
    # Gold: speaker 1's turn ends at 2.0s. Model: predicts an event at 2.2s.
    result = score(positive_events=[anchor_event(1, 2.0)], speaker_1_times=[2.2])

    assert result.tp == 1
    assert result.fn == 0
    assert result.latencies_ms[0] == approx(200.0)


def test_miss_when_prediction_arrives_too_late():
    # Gold: turn ends at 2.0s (window closes at 2.0 + τ_max = 4.0s)
    # Model: predicts at 5.0s — outside the window
    result = score(positive_events=[anchor_event(1, 2.0)], speaker_1_times=[5.0])

    assert result.tp == 0
    assert result.fn == 1


def test_early_prediction_counts_within_tolerance():
    # Gold: turn ends at 2.0s; prediction at 1.8s is within τ_pre = 0.25s
    result = score(positive_events=[anchor_event(1, 2.0)], speaker_1_times=[1.8])

    assert result.tp == 1
    assert result.latencies_ms[0] == approx(-200.0)


def test_fp_when_firing_inside_a_pause():
    # Gold: pause from 5.0s to 6.0s; model fires mid-pause
    result = score(negative_spans=[pause(1, 5.0, 6.0)], speaker_1_times=[5.5])

    assert result.fp == 1
    assert result.tn == 0


def test_fp_when_firing_just_before_a_pause():
    # The τ_pre tolerance is charged symmetrically: a speculative fire in the
    # last 0.25s of speech before a pause is an FP, just as one before a real
    # EOT would be a TP.
    result = score(negative_spans=[pause(1, 5.0, 6.0)], speaker_1_times=[4.8])

    assert result.fp == 1


def test_no_fp_after_the_speaker_resumes():
    # The span ends at resumption (6.0s); firing during resumed speech is
    # outside every scored region — invisible, not an FP.
    result = score(negative_spans=[pause(1, 5.0, 6.0)], speaker_1_times=[6.5])

    assert result.fp == 0
    assert result.tn == 1


def test_correct_detection_is_never_also_an_fp():
    # The original double-counting bug: a real EOT at 3.0s right after a pause
    # [2.0, 2.3]. The correct prediction at 3.1s must be a TP and nothing else.
    result = score(
        positive_events=[anchor_event(1, 3.0)],
        negative_spans=[pause(1, 2.0, 2.3)],
        speaker_1_times=[3.1],
    )

    assert result.tp == 1
    assert result.fp == 0
    assert result.tn == 1


def test_one_prediction_cannot_satisfy_two_positives():
    # Two real EOTs 1s apart; a single prediction between them claims only the
    # first (earliest-qualifying) — the second is a miss.
    result = score(
        positive_events=[anchor_event(1, 2.0), anchor_event(1, 3.0)],
        speaker_1_times=[2.9],
    )

    assert result.tp == 1
    assert result.fn == 1


def test_burst_in_one_positive_window_is_one_tp_and_no_fp():
    # Three predictions in one positive window: the first is claimed, the rest
    # are inside the positive window and therefore exempt from an overlapping
    # pause — score-neutral, like everything else in a positive window.
    result = score(
        positive_events=[anchor_event(1, 2.0)],
        negative_spans=[pause(1, 2.5, 3.5)],
        speaker_1_times=[2.1, 2.2, 2.6],
    )

    assert result.tp == 1
    assert result.latencies_ms == approx([100.0])
    assert result.fp == 0
    assert result.tn == 1


def test_burst_in_one_negative_span_is_one_fp():
    # Three predictions inside one pause count as a single FP.
    result = score(
        negative_spans=[pause(1, 5.0, 6.0)],
        speaker_1_times=[5.1, 5.2, 5.3],
    )

    assert result.fp == 1
    assert result.tn == 0


def test_firing_only_during_speech_does_not_game_the_metric():
    # A degenerate model fires every 100ms while its speaker talks and never
    # during silence. The τ_pre back-extension of the pause span catches the
    # fire at 4.9s (speech, just before the 5.0s pause) — so the strategy pays
    # fp_rate, it doesn't harvest free speculative TPs.
    speech_fires = [4.7, 4.8, 4.9]  # speech ends at 5.0s, pause until 6.0s
    result = score(
        negative_spans=[pause(1, 5.0, 6.0)],
        speaker_1_times=speech_fires,
    )

    assert result.fp == 1


def test_speakers_are_scored_on_their_own_channel():
    # Gold: speaker 2 event at 2.0s; the model only predicts on speaker 1.
    result = score(positive_events=[anchor_event(2, 2.0)], speaker_1_times=[2.2])

    assert result.tp == 0
    assert result.fn == 1


def test_excluded_interval_hides_prediction():
    # Gold: turn ends at 2.0s. Model: predicts at 2.2s — normally a TP.
    # Excluded: 2.1–2.3s on speaker 1 masks that prediction.
    masked = score(
        positive_events=[anchor_event(1, 2.0)],
        speaker_1_times=[2.2],
        excluded=[Interval(speaker=1, start=2.1, end=2.3)],
    )

    assert masked.tp == 0
    assert masked.fn == 1


def test_excluded_interval_only_masks_its_own_speaker():
    # The same span excluded on speaker 2 does not mask speaker 1's prediction.
    result = score(
        positive_events=[anchor_event(1, 2.0)],
        speaker_1_times=[2.2],
        excluded=[Interval(speaker=2, start=2.1, end=2.3)],
    )

    assert result.tp == 1


def test_unsorted_predictions_fail_loudly():
    try:
        score(positive_events=[anchor_event(1, 2.0)], speaker_1_times=[3.0, 2.2])
    except AssertionError:
        return
    raise AssertionError("unsorted predicted times must be rejected")


def test_qualifies():
    # Clean run: detects the turn-end, stays quiet on the pause
    passing = score(
        positive_events=[anchor_event(1, 2.0)],
        negative_spans=[pause(1, 8.0, 9.0)],
        speaker_1_times=[2.2],
    )
    # Same detection, plus a spurious fire inside the pause
    failing = score(
        positive_events=[anchor_event(1, 2.0)],
        negative_spans=[pause(1, 8.0, 9.0)],
        speaker_1_times=[2.2, 8.5],
    )

    assert qualifies(passing, fp_budget=0.5, recall_floor=0.5)
    assert not qualifies(failing, fp_budget=0.0, recall_floor=0.5)


def test_rank_key_prefers_lower_median_latency():
    fast = TaskScore(tp=1, latencies_ms=[100.0])
    slow = TaskScore(tp=1, latencies_ms=[500.0])
    assert rank_key(fast) < rank_key(slow)


# ---- end-to-end: synthetic conversation -> scores ---------------------------

DURATION_S = 10.0


def unanimous(*entries: Annotation) -> dict[str, list[Annotation]]:
    """All three annotators agree exactly on the given events."""
    return {annotator: list(entries) for annotator in ANNOTATORS}


@pytest.fixture
def conversation() -> Conversation:
    """A synthetic conversation with the scoring regions deliberately adjacent.

    Speaker 1: Turn [1.0–2.0] + [2.3–3.0] — a mid-turn pause spanning
    [2.0, 2.3], then a real EOT at 3.0 only 0.7s later (speaker 2 takes over).
    Speaker 2: Backchannel [1.5–1.7] (INT negative), Interruption [4.5–4.8]
    (INT positive), Turn [3.5–5.0] whose end is a real EOT (nobody speaks again).
    """
    return conv(
        unanimous(
            (1.0, 2.0, "Normal Turn"),
            (2.3, 3.0, "Normal Turn"),
        ),
        unanimous(
            (1.5, 1.7, "Acknowledgement Backchannel"),
            (4.5, 4.8, "Floor-taking Competitive Interruption"),
            (3.5, 5.0, "Normal Turn"),
        ),
        duration_s=DURATION_S,
    )


def prediction(
    *,
    speaker_1: SpeakerEvents | None = None,
    speaker_2: SpeakerEvents | None = None,
) -> ConversationPrediction:
    no_events = SpeakerEvents(eot=[], interruption=[])
    return ConversationPrediction(
        conversation_id="1",
        speaker_1=speaker_1 or no_events,
        speaker_2=speaker_2 or no_events,
    )


def test_gold_derived_predictions_score_perfectly(conversation: Conversation):
    scores = score_conversation(
        prediction(
            speaker_1=SpeakerEvents(eot=[3.0], interruption=[]),
            speaker_2=SpeakerEvents(eot=[5.0], interruption=[4.5]),
        ),
        conversation,
    )

    assert (scores.task_eot.tp, scores.task_eot.fn) == (2, 0)
    assert (scores.task_eot.fp, scores.task_eot.tn) == (0, 1)  # tn = the pause at 2.0s
    assert scores.task_eot.latencies_ms == [0.0, 0.0]
    assert (scores.task_int.tp, scores.task_int.fn) == (1, 0)
    assert (scores.task_int.fp, scores.task_int.tn) == (0, 1)  # tn = the backchannel
    assert scores.task_int.latencies_ms == [0.0]


def test_firing_in_the_pause_is_an_fp(conversation: Conversation):
    scores = score_conversation(
        prediction(speaker_1=SpeakerEvents(eot=[2.1, 3.0], interruption=[])),
        conversation,
    )

    assert scores.task_eot.tp == 1  # the real EOT at 3.0 still counts
    assert scores.task_eot.fp == 1  # 2.1 is inside the pause [2.0, 2.3]


def test_no_events_misses_everything_cleanly(conversation: Conversation):
    scores = score_conversation(prediction(), conversation)

    assert (scores.task_eot.tp, scores.task_eot.fn, scores.task_eot.fp) == (0, 2, 0)
    assert (scores.task_int.tp, scores.task_int.fn, scores.task_int.fp) == (0, 1, 0)


def test_event_past_the_audio_is_rejected(conversation: Conversation):
    with pytest.raises(ValueError, match="past the end"):
        score_conversation(
            prediction(speaker_1=SpeakerEvents(eot=[100.0], interruption=[])),
            conversation,
        )
