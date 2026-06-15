"""Degenerate baseline: energy VAD. A speaker is "speaking" wherever their
20 ms RMS exceeds a threshold; every time their speech stops is predicted as
an EOT, every time it starts as an interruption. Fires on every silence — low
latency, high FP — the other end of the tradeoff from no_events.

Usage: python -m eval.baselines.vad > predictions.json
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from eval.data import conversation_ids, dataset_dir
from eval.submission import (
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


def main() -> None:
    data_dir = dataset_dir()
    submission = Submission(
        schema_version=SCHEMA_VERSION,
        predictions=[
            ConversationPrediction(
                conversation_id=task_id,
                speaker_1=predict_speaker(data_dir / task_id / "speaker_1_audio.flac"),
                speaker_2=predict_speaker(data_dir / task_id / "speaker_2_audio.flac"),
            )
            for task_id in conversation_ids(data_dir)
        ],
    )
    print(submission.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
