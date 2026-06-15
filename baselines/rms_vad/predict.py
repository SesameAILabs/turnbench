#!/usr/bin/env python3
"""RMS energy VAD baseline.

A speaker is "speaking" wherever their 20 ms RMS exceeds a threshold; a speech
onset is predicted as an interruption and a speech offset as an EOT, each
committed at the end of the window that revealed the flip. The RMS threshold is
this baseline's operating point. It fires on every silence: low latency, high
false-positive rate, the opposite corner of the tradeoff from never firing.

This is the reference shape for a baseline in the discrete benchmark: a
self-contained, standalone predictor that returns one `ConversationPrediction`
per conversation and is scored by `eval.score`. A real model baseline does the
same, with its own model in place of the RMS rule and its own continuous->
discrete step in place of the threshold + edge detection here.

    python -m baselines.rms_vad.predict                  # score on the dev set
    python -m baselines.rms_vad.predict --out preds.json # write a predictions JSON
    python -m baselines.rms_vad.predict --dataset <hf repo|local dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# eval/ is two levels up (baselines/rms_vad/predict.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.data import DEV_DATASET, conversation_ids, resolve_dataset  # noqa: E402
from eval.score import score_submission, task_cells  # noqa: E402
from eval.submission import (  # noqa: E402
    SCHEMA_VERSION,
    ConversationPrediction,
    SpeakerEvents,
    Submission,
)

WINDOW_S = 0.02
RMS_THRESHOLD = 0.01


def is_speaking_per_window(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """One bool per 20 ms window: is this window's RMS above the threshold?"""
    samples_per_window = int(WINDOW_S * sample_rate)
    window_count = len(audio) // samples_per_window
    windows = audio[: window_count * samples_per_window].reshape(
        window_count, samples_per_window
    )
    rms_per_window = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1))
    return rms_per_window > RMS_THRESHOLD


def predict_speaker(audio_path: Path) -> SpeakerEvents:
    """Energy-VAD events for one channel: speech offsets -> EOT, onsets -> interruption."""
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    speaking = is_speaking_per_window(audio, sample_rate)

    eot: list[float] = []
    interruption: list[float] = []
    for window in range(1, len(speaking)):
        if speaking[window] == speaking[window - 1]:
            continue
        # The flip is known once this window has been heard, so the commit
        # time is the window's END.
        commit_time_s = (window + 1) * WINDOW_S
        if speaking[window]:
            interruption.append(commit_time_s)  # speech started
        else:
            eot.append(commit_time_s)  # speech stopped
    return SpeakerEvents(eot=eot, interruption=interruption)


def predict(sample_dir: Path) -> ConversationPrediction:
    """Energy-VAD events for both channels of one conversation."""
    return ConversationPrediction(
        conversation_id=sample_dir.name,
        speaker_1=predict_speaker(sample_dir / "speaker_1_audio.flac"),
        speaker_2=predict_speaker(sample_dir / "speaker_2_audio.flac"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=DEV_DATASET,
        help="HF dataset repo id, or a local directory of conversation dirs",
    )
    parser.add_argument(
        "--out", default=None, help="write a predictions JSON here instead of scoring"
    )
    args = parser.parse_args()

    data_dir = resolve_dataset(source=args.dataset)
    submission = Submission(
        schema_version=SCHEMA_VERSION,
        predictions=[predict(data_dir / task_id) for task_id in conversation_ids(data_dir)],
    )

    if args.out is not None:
        Path(args.out).write_text(submission.model_dump_json(indent=2), encoding="utf-8")
        print(f"Wrote {len(submission.predictions)} predictions to {args.out}", file=sys.stderr)
        return 0

    scores = score_submission(submission, data_dir)
    print(f"rms_vad — {len(submission.predictions)} conversations")
    for task_name, score in (("EOT", scores.task_eot), ("INT", scores.task_int)):
        recall, fp_rate, latency = task_cells(score)
        print(f"  {task_name}: recall={recall} fp_rate={fp_rate} latency_ms={latency}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
