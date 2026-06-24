"""Shared adapter: ASR word timestamps from generative full-duplex runs
-> unified submission-format traces.

Generative baselines (moshi, gemini) are evaluated offline: each speaker's
channel is streamed into the model as the "user", the model's output audio
is saved, and both the output and the input are transcribed with word-level
timestamps (NVIDIA parakeet-tdt-0.6b-v2). This module converts those
word timestamps into the four per-frame score arrays.

Expected ASR layout, one directory per task, one subdirectory per direction
(speaker K = the user whose audio was streamed in):

    <asr_root>/<task_id>/speaker_K/
        output.json             words of the model's output audio
        speaker_K_audio.json    words of the user's input audio

Each JSON: {"text": ..., "chunks": [{"text": w, "timestamp": [s, e]}, ...]}

Floor-state derivation (one direction, user = speaker K, the agent stands
in for the other seat J):

  1. Word intervals separated by < `merge_gap_s` are merged into
     continuous speech regions (ASR words have small gaps mid-utterance).
  2. The agent's silent->speaking transitions ("onsets") are the model's
     floor-taking decisions. At each onset:
       - user K inactive  -> EOT pulse: the agent judged K's turn over
                             -> eot_score_speaker_K
       - user K active    -> the agent barges in on K
                             -> interruption_score_speaker_J
  3. Pulses are single frames of 1.0 at `frame_rate_hz`; everything else
     is 0.0. Arrays are trimmed to the conversation duration (drops the
     response tail some pipelines append after the input ends).

Combining the two directions fills the four canonical arrays exactly once:
direction speaker_1 -> eot_score_speaker_1 + interruption_score_speaker_2;
direction speaker_2 -> eot_score_speaker_2 + interruption_score_speaker_1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

FRAME_RATE_HZ = 12.5
MERGE_GAP_S = 0.5


def load_word_intervals(json_path: Path) -> list[tuple[float, float]]:
    """Read [(start_s, end_s)] from one ASR transcript JSON."""
    data = json.loads(json_path.read_text())
    return [(c["timestamp"][0], c["timestamp"][1]) for c in data["chunks"]]


def merge_regions(
    intervals: list[tuple[float, float]],
    max_gap_s: float = MERGE_GAP_S,
) -> list[tuple[float, float]]:
    """Merge word intervals separated by < max_gap_s into speech regions."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    regions = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start - regions[-1][1] < max_gap_s:
            regions[-1][1] = max(regions[-1][1], end)
        else:
            regions.append([start, end])
    return [(s, e) for s, e in regions]


def active_at(regions: list[tuple[float, float]], t: float) -> bool:
    return any(s <= t < e for s, e in regions)


def conversation_duration_s(sample_dir: Path) -> float:
    """Duration of the conversation, from the dataset's speaker audio
    (wav in the official delivery, flac in some local mirrors)."""
    for pattern in ("speaker_1_audio.*", "speaker_2_audio.*"):
        for audio in sorted(sample_dir.glob(pattern)):
            return sf.info(audio).duration
    raise FileNotFoundError(f"No speaker audio found in {sample_dir}")


def direction_pulses(
    direction_dir: Path,
    speaker: int,
    n_frames: int,
    frame_rate_hz: float = FRAME_RATE_HZ,
    merge_gap_s: float = MERGE_GAP_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Score arrays for one direction (user = `speaker`).

    Returns (eot_score_user, interruption_score_agent): pulses at the
    agent's speech onsets, routed by whether the user was speaking.
    """
    agent_regions = merge_regions(
        load_word_intervals(direction_dir / "output.json"), merge_gap_s)
    user_regions = merge_regions(
        load_word_intervals(direction_dir / f"speaker_{speaker}_audio.json"), merge_gap_s)

    eot = np.zeros(n_frames, dtype=np.float32)
    interruption = np.zeros(n_frames, dtype=np.float32)
    for onset_s, _ in agent_regions:
        frame = int(onset_s * frame_rate_hz)
        if not 0 <= frame < n_frames:
            continue  # response tail beyond the conversation, or bad timestamp
        if active_at(user_regions, onset_s):
            interruption[frame] = 1.0
        else:
            eot[frame] = 1.0
    return eot, interruption


def predict_scores_from_asr(
    sample_dir: Path,
    asr_root: Path,
    frame_rate_hz: float = FRAME_RATE_HZ,
    merge_gap_s: float = MERGE_GAP_S,
) -> dict:
    """Build the unified-format result dict for one task from cached ASR."""
    task_dir = asr_root / sample_dir.name
    n_frames = int(conversation_duration_s(sample_dir) * frame_rate_hz)

    eot_1, int_2 = direction_pulses(
        task_dir / "speaker_1", 1, n_frames, frame_rate_hz, merge_gap_s)
    eot_2, int_1 = direction_pulses(
        task_dir / "speaker_2", 2, n_frames, frame_rate_hz, merge_gap_s)

    return {
        "frame_rate_hz": frame_rate_hz,
        "eot_score_speaker_1": eot_1,
        "eot_score_speaker_2": eot_2,
        "interruption_score_speaker_1": int_1,
        "interruption_score_speaker_2": int_2,
    }
