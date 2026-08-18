"""Shared engine for the OpenAI Realtime turn-detection baselines.

The Realtime API's server-side turn detection *is* an end-of-turn detector: we
stream a speaker's channel in over a WebSocket and collect the
`input_audio_buffer.speech_stopped` events the server emits when it decides that
speaker's turn ended. Two turn-detection modes give two baselines:

  - openai_server_vad   — acoustic silence VAD. Commit time folds in the trailing
                          silence the decision needed: audio_end_ms + silence_duration_ms.
                          Audio is burst-fed (faster than real time); the timestamp
                          is pacing-independent.
  - openai_semantic_vad — a classifier that decides end-of-turn from content. Its
                          internal decision delay isn't exposed, so audio is fed at
                          ~real time and each event is timestamped by the audio
                          position when it arrives (folding in that delay).

Each speaker channel is run independently and scored on its own channel — exactly
how turnbench.score works. Causal by construction: the model only ever sees audio up to
the current append, and the commit time folds in the detection wait. Both tasks
come from the same VAD stream, per channel: `speech_stopped` -> EOT (turn end),
`speech_started` -> interruption (speech onset / floor-taking attempt). The scorer
decides which onsets are real interruptions vs. backchannels from the gold.

Uses the official `openai` SDK's async Realtime client (it manages the WebSocket,
auth, and event typing). NOTE: the session-config schema evolves across API
versions — it is isolated in `_session_config()` (GA `gpt-realtime` nests audio
config under `session.audio.input`) so a fresh clone can adjust it to the installed
SDK / API version without touching the detection logic. Verify with your key first.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
from pathlib import Path

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import snapshot_download
from openai import AsyncOpenAI

MODEL = "gpt-realtime"
TARGET_SR = 24_000  # the Realtime API expects 24 kHz mono PCM16
CHUNK_MS = 20  # append granularity
TRAILING_GRACE_S = 5.0  # wait this long after the audio ends for final events


def load_api_key(baseline_dir: Path) -> str:
    """OPENAI_API_KEY, preferring the environment, then the repo-root .env (shared
    by both openai_* baselines), then the baseline folder's .env as a fallback."""
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    repo_root = baseline_dir.parent.parent
    for env in (repo_root / ".env", baseline_dir / ".env"):
        if not env.exists():
            continue
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(
        "OPENAI_API_KEY not found in the environment or a .env (repo root or baseline folder)"
    )


def _pcm16_24k(samples: np.ndarray, sample_rate: int) -> bytes:
    """One mono channel -> 24 kHz PCM16 bytes."""
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sample_rate != TARGET_SR:
        samples = librosa.resample(
            samples.astype(np.float32), orig_sr=sample_rate, target_sr=TARGET_SR
        )
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _session_config(turn_detection: dict) -> dict:
    """The one schema-version-sensitive payload (GA gpt-realtime nesting). Disables
    spoken responses — we only want the turn-detection events."""
    return {
        "type": "realtime",
        "output_modalities": ["text"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": TARGET_SR},
                "turn_detection": {**turn_detection, "create_response": False},
            },
        },
    }


def _clean(times: list[float], duration_s: float) -> list[float]:
    """Sort, de-dupe, round to ms, and keep strictly inside the audio (<= duration_s)
    and strictly increasing (required by SpeakerEvents). Clamp *after* rounding —
    rounding can otherwise nudge a near-end time past the duration."""
    kept: list[float] = []
    for t in sorted(min(t, duration_s) for t in times):
        if not kept or t - kept[-1] > 1e-3:
            kept.append(t)
    return [min(round(t, 3), duration_s) for t in kept]


async def _detect_events(
    client: AsyncOpenAI, pcm: bytes, *, turn_detection: dict, duration_s: float
) -> tuple[list[float], list[float]]:
    """Stream one channel in real time -> (eot_times, interruption_times). The server
    VAD emits `speech_stopped` (turn end -> EOT) and `speech_started` (speech onset ->
    interruption / floor-taking attempt). Each is committed at the audio position the
    model had heard when it fired, folding in the detection wait. Real-time pacing is
    required so the server keeps up and emits every boundary (bursting a whole
    conversation only yields the first few before the buffer is processed)."""
    bytes_per_chunk = int(TARGET_SR * CHUNK_MS / 1000) * 2
    eots: list[float] = []
    ints: list[float] = []
    appended_ms = 0.0

    async with client.realtime.connect(model=MODEL) as conn:
        await conn.session.update(session=_session_config(turn_detection))

        async def feed() -> None:
            nonlocal appended_ms
            for start in range(0, len(pcm), bytes_per_chunk):
                chunk = pcm[start : start + bytes_per_chunk]
                await conn.input_audio_buffer.append(
                    audio=base64.b64encode(chunk).decode("ascii")
                )
                appended_ms += len(chunk) / 2 / TARGET_SR * 1000  # exact (last chunk is partial)
                await asyncio.sleep(CHUNK_MS / 1000)  # real-time pace

        async def receive() -> None:
            async for event in conn:
                if event.type == "input_audio_buffer.speech_stopped":
                    eots.append(appended_ms / 1000.0)  # turn end
                elif event.type == "input_audio_buffer.speech_started":
                    ints.append(appended_ms / 1000.0)  # onset (interruption candidate)
                elif event.type == "error":
                    raise RuntimeError(f"realtime error: {event.error}")

        receiver = asyncio.create_task(receive())
        await feed()
        await asyncio.sleep(TRAILING_GRACE_S)  # let trailing events arrive
        receiver.cancel()

    return _clean(eots, duration_s), _clean(ints, duration_s)


# ---- lazy dataset access -------------------------------------------------------
#
# The dataset ships one conversation per parquet row group, so we read a single
# conversation on demand (its row group only) rather than materialising the whole
# split in RAM (~24 GB for test). Peak memory is then ~concurrency conversations,
# not the entire dataset. The baseline needs only the audio; duration comes from it.


def _shard_files(source: str, revision: str | None) -> list[str]:
    """Parquet shards for a split — a local directory, or an HF dataset snapshot."""
    if Path(source).is_dir():
        return sorted(str(p) for p in Path(source).glob("*.parquet"))
    from turnbench.data import PINNED_REVISIONS

    snapshot = snapshot_download(
        source, repo_type="dataset", revision=revision or PINNED_REVISIONS.get(source),
        allow_patterns="*.parquet",
    )
    return sorted(str(p) for p in Path(snapshot).rglob("*.parquet"))


def dataset_index(source: str, revision: str | None) -> dict[str, tuple[str, int]]:
    """{conversation_id: (parquet_path, row_group)} — reads only the id column, no audio."""
    index: dict[str, tuple[str, int]] = {}
    for shard in _shard_files(source, revision):
        parquet = pq.ParquetFile(shard)
        assert parquet.metadata.num_rows == parquet.metadata.num_row_groups, (
            f"{shard}: expected one conversation per row group"
        )
        ids = parquet.read(columns=["conversation_id"])["conversation_id"].to_pylist()
        for row_group, cid in enumerate(ids):
            index[cid] = (shard, row_group)
    return index


def _conversation_pcms(shard: str, row_group: int) -> tuple[bytes, bytes, float]:
    """Read one conversation's row group -> (speaker_1 pcm, speaker_2 pcm, duration_s)."""
    table = pq.ParquetFile(shard).read_row_group(
        row_group, columns=["speaker_1_audio", "speaker_2_audio"]
    )

    def decode(column: str):
        cell = table[column][0].as_py()
        data = cell["bytes"] if isinstance(cell, dict) else cell
        samples, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        return _pcm16_24k(samples, sample_rate), len(samples) / sample_rate

    pcm1, duration = decode("speaker_1_audio")
    pcm2, _ = decode("speaker_2_audio")
    return pcm1, pcm2, duration


async def _conversation_events(
    client: AsyncOpenAI, shard: str, row_group: int, *, turn_detection: dict
) -> tuple[tuple[list[float], list[float]], tuple[list[float], list[float]]]:
    """Both channels concurrently (independent sessions) -> per speaker (eot, interruption)."""
    # read + decode + resample off the event loop — it's CPU/IO-heavy and would
    # otherwise block every other session's WebSocket keepalive (1011 ping timeout).
    pcm1, pcm2, duration_s = await asyncio.to_thread(_conversation_pcms, shard, row_group)
    spk1, spk2 = await asyncio.gather(
        *(
            _detect_events(client, pcm, turn_detection=turn_detection, duration_s=duration_s)
            for pcm in (pcm1, pcm2)
        )
    )
    return spk1, spk2


# ---- per-baseline driver (mirrors baselines/rms_vad/predict.py) ----------------


def run_openai_baseline(vad_mode: str, baseline_dir: Path) -> int:
    """CLI entrypoint shared by the two openai_* baselines."""
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from turnbench.data import DEV_DATASET
    from turnbench.submission import (
        SCHEMA_VERSION,
        ConversationPrediction,
        SpeakerEvents,
        Submission,
        load_submission,
    )

    parser = argparse.ArgumentParser(
        description=f"OpenAI Realtime {vad_mode} EOT baseline"
    )
    parser.add_argument(
        "--dataset", default=DEV_DATASET, help="HF dataset repo id or local parquet dir"
    )
    parser.add_argument(
        "--out", default=None, help="write predictions JSON here instead of scoring"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only the first N conversations (debugging)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1000,
        help="conversations run in parallel (independent sessions); default fans out "
        "all of them — lower it if the API rate-limits",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip conversations already in --out (re-run to finish a partial run)",
    )
    args = parser.parse_args()

    # stock API turn-detection defaults — the mode only, no parameters
    turn_detection = {"type": vad_mode}
    api_key = load_api_key(baseline_dir)

    index = dataset_index(args.dataset, None)  # {conversation_id: (shard, row_group)}, no audio loaded
    ids = sorted(index, key=int)
    if args.limit is not None:
        ids = ids[: args.limit]

    out_path = Path(args.out) if args.out else None

    # completed conversations, persisted after each one finishes so a dropped run
    # (VPN, server blip) loses nothing — --resume picks up where it left off.
    done: dict[str, ConversationPrediction] = {}
    if args.resume and out_path is not None and out_path.exists():
        keep = set(ids)
        done = {p.conversation_id: p for p in load_submission(out_path).predictions
                if p.conversation_id in keep}
        print(f"resume: {len(done)}/{len(ids)} already done", file=sys.stderr)

    def assembled() -> Submission:  # completed predictions, in dataset order
        return Submission(schema_version=SCHEMA_VERSION,
                          predictions=[done[t] for t in ids if t in done])

    async def run_all() -> None:
        semaphore = asyncio.Semaphore(args.concurrency)
        write_lock = asyncio.Lock()
        async with AsyncOpenAI(api_key=api_key) as client:
            async def one(task_id: str) -> None:
                if task_id in done:
                    return
                async with semaphore:
                    try:
                        shard, row_group = index[task_id]
                        (eot1, int1), (eot2, int2) = await _conversation_events(
                            client, shard, row_group, turn_detection=turn_detection,
                        )
                    except Exception as error:  # a single dropped session must not abort the run
                        print(f"  {task_id}: FAILED ({type(error).__name__}) — re-run with --resume",
                              file=sys.stderr)
                        return
                pred = ConversationPrediction(
                    conversation_id=task_id,
                    speaker_1=SpeakerEvents(eot=eot1, interruption=int1),
                    speaker_2=SpeakerEvents(eot=eot2, interruption=int2),
                )
                async with write_lock:
                    done[task_id] = pred
                    if out_path is not None:  # persist progress incrementally
                        out_path.write_text(assembled().model_dump_json(indent=2), encoding="utf-8")
                print(f"  {task_id}: EOT {len(eot1)}/{len(eot2)}  INT {len(int1)}/{len(int2)}  "
                      f"[{len(done)}/{len(ids)}]", file=sys.stderr)

            await asyncio.gather(*(one(task_id) for task_id in ids))

    asyncio.run(run_all())

    missing = [t for t in ids if t not in done]
    if missing:
        print(f"incomplete: {len(done)}/{len(ids)} done, {len(missing)} failed: {missing}\n"
              f"re-run with --resume to finish.", file=sys.stderr)
        return 1
    if out_path is not None:
        print(f"Wrote {len(done)} predictions to {out_path}", file=sys.stderr)
        return 0

    from turnbench.data import resolve_dataset  # only for optional in-process dev scoring
    from turnbench.score import score_submission, task_cells

    scores = score_submission(assembled(), resolve_dataset(source=args.dataset, skip_audio=True))
    print(f"openai_{vad_mode} — {len(done)} conversations")
    for task_name, score in (("EOT", scores.task_eot), ("INT", scores.task_int)):
        recall, fp_rate, latency = task_cells(score)
        print(f"  {task_name}: recall={recall} fp_rate={fp_rate} latency_ms={latency}")
    return 0
