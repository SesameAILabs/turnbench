#!/usr/bin/env python3
"""
Headless LiveKit client: stream a local audio file (wav/flac/…) into a LiveKit
room and capture the agent's audio response – no browser or web UI required.

This script acts as a "user" participant. It joins the same LiveKit room that
the agent (lk_agent.py) is listening on, publishes audio from the input file,
and saves the agent's spoken response time-aligned with the input — both as
.wav and .flac.

Usage:
    python lk_audio_client.py -i speaker_1_audio.flac -o output.wav \
        [--room ROOM_NAME] [--silence SEC]

The response is written to OUTPUT and to the same path with a .flac suffix
(e.g. -o output.wav  →  output.wav + output.flac).

Requirements:
    pip install "livekit[crypto]~=1.0" python-dotenv numpy soundfile scipy

Environment variables (in .env.local):
    LIVEKIT_URL          – wss://your-project.livekit.cloud
    LIVEKIT_API_KEY      – your LiveKit API key
    LIVEKIT_API_SECRET   – your LiveKit API secret
"""

import argparse
import asyncio
import logging
import os
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from dotenv import load_dotenv
from livekit import api, rtc

load_dotenv(Path(__file__).resolve().parent / ".env.local")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("lk_audio_client")

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

SAMPLE_WIDTH = 2  # 16-bit PCM = 2 bytes per sample

# LiveKit WebRTC uses 48 kHz by default for audio tracks.
PUBLISH_SAMPLE_RATE = 48000
CAPTURE_SAMPLE_RATE = 24000  # sample rate we request when receiving agent audio
CHUNK_DURATION_MS = 20  # 20 ms chunks (standard for WebRTC)


def read_audio_pcm16(path: str, target_rate: int, silence_sec: float = 0.0) -> bytes:
    """
    Read any audio file and return mono PCM-16 bytes at *target_rate* Hz,
    optionally followed by *silence_sec* seconds of trailing silence.
    """
    data, rate = sf.read(path, dtype="float64")

    if data.ndim > 1:
        data = data.mean(axis=1)

    if rate != target_rate:
        log.info("Resampling '%s' from %d Hz to %d Hz …", path, rate, target_rate)
        g = gcd(target_rate, rate)
        data = resample_poly(data, target_rate // g, rate // g)

    pcm = np.clip(np.round(data * 32767), -32768, 32767).astype(np.int16)

    if silence_sec > 0:
        pcm = np.concatenate([pcm, np.zeros(int(target_rate * silence_sec), dtype=np.int16)])

    return pcm.tobytes()


def write_outputs(output_path: str, samples: np.ndarray, sample_rate: int) -> str:
    """Write the response to OUTPUT_PATH and its .flac sibling. Returns flac path."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, samples, sample_rate, subtype="PCM_16")
    flac_path = str(Path(output_path).with_suffix(".flac"))
    sf.write(flac_path, samples, sample_rate, format="FLAC", subtype="PCM_16")
    return flac_path


# ---------------------------------------------------------------------------
# Main client logic
# ---------------------------------------------------------------------------

async def run(input_path: str, output_path: str, room_name: str, silence_sec: float):
    url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not all([url, api_key, api_secret]):
        raise RuntimeError(
            "Missing environment variables. Set LIVEKIT_URL, LIVEKIT_API_KEY, "
            "and LIVEKIT_API_SECRET in .env.local"
        )

    # ── Read input audio (with optional trailing silence) ─────────────
    pcm_data = read_audio_pcm16(input_path, PUBLISH_SAMPLE_RATE, silence_sec)
    total_samples = len(pcm_data) // SAMPLE_WIDTH
    duration_sec = total_samples / PUBLISH_SAMPLE_RATE
    log.info(
        "Input: %s  (%.2fs incl. %.1fs silence, %d Hz, mono 16-bit)",
        input_path, duration_sec, silence_sec, PUBLISH_SAMPLE_RATE,
    )

    # ── Generate access token ─────────────────────────────────────────
    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity("audio-file-user")
        .with_name("Audio File User")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
            )
        )
        .to_jwt()
    )

    # ── Pre-allocate output buffer (same length as input) ───────────
    # The output will be exactly the same duration as the (padded) input.
    # Agent audio is written into this buffer at the time offset it
    # arrives (relative to when we start streaming the input).
    target_samples = int(duration_sec * CAPTURE_SAMPLE_RATE)
    output_buf = np.zeros(target_samples, dtype=np.int16)
    write_pos = 0  # next sample position to write into output_buf
    recording_started = asyncio.Event()
    recording_stop = asyncio.Event()

    # ── Connect to room ───────────────────────────────────────────────
    room = rtc.Room()

    receiving_task: asyncio.Task | None = None
    agent_audio_ready = asyncio.Event()

    async def _receive_agent_audio(track: rtc.Track):
        """Collect agent audio frames into output_buf in real-time."""
        nonlocal write_pos
        stream = rtc.AudioStream(
            track,
            sample_rate=CAPTURE_SAMPLE_RATE,
            num_channels=1,
        )
        log.info("Receiving agent audio …")
        agent_audio_ready.set()

        # Wait until we actually start streaming input so the clocks align
        await recording_started.wait()

        try:
            async for frame_event in stream:
                if recording_stop.is_set():
                    break

                frame = frame_event.frame
                samples = np.frombuffer(bytes(frame.data), dtype=np.int16)
                n = len(samples)
                remaining = target_samples - write_pos
                if remaining <= 0:
                    break
                to_write = min(n, remaining)
                output_buf[write_pos : write_pos + to_write] = samples[:to_write]
                write_pos += to_write
        except Exception as e:
            log.warning("Audio receive error: %s", e)
        finally:
            await stream.aclose()
            log.info(
                "Recording stopped at %.2fs of %.2fs.",
                write_pos / CAPTURE_SAMPLE_RATE,
                duration_sec,
            )

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            log.info(
                "Subscribed to audio track from '%s' (%s)",
                participant.identity, participant.name,
            )
            nonlocal receiving_task
            receiving_task = asyncio.create_task(_receive_agent_audio(track))

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        log.info("Participant joined: %s (%s)", participant.identity, participant.name)

    @room.on("disconnected")
    def on_disconnected():
        log.info("Disconnected from room.")

    log.info("Connecting to room '%s' at %s …", room_name, url)
    await room.connect(
        url,
        token,
        options=rtc.RoomOptions(auto_subscribe=True),
    )
    log.info("Connected to room '%s'. Waiting for agent to join …", room.name)

    # ── Publish audio track (but don't stream yet) ────────────────────
    source = rtc.AudioSource(PUBLISH_SAMPLE_RATE, 1)
    local_track = rtc.LocalAudioTrack.create_audio_track("audio-input", source)

    pub_opts = rtc.TrackPublishOptions()
    pub_opts.source = rtc.TrackSource.SOURCE_MICROPHONE
    await room.local_participant.publish_track(local_track, pub_opts)
    log.info("Published audio track.")

    # ── Wait for agent to join and audio channel to be ready ──────────
    AGENT_TIMEOUT = 60
    try:
        await asyncio.wait_for(agent_audio_ready.wait(), timeout=AGENT_TIMEOUT)
    except asyncio.TimeoutError:
        log.error("Agent did not join within %ds — aborting.", AGENT_TIMEOUT)
        await room.disconnect()
        raise RuntimeError(f"Agent did not join room '{room_name}' within {AGENT_TIMEOUT}s")
    log.info("Agent audio channel ready. Starting playback …")

    # Brief pause so the agent's realtime model is fully initialized
    await asyncio.sleep(1)

    # ── Stream input & record output in parallel ──────────────────────
    samples_per_chunk = PUBLISH_SAMPLE_RATE * CHUNK_DURATION_MS // 1000
    chunk_bytes = samples_per_chunk * SAMPLE_WIDTH

    # Signal recording start — the receive task starts writing from here
    stream_start_time = asyncio.get_event_loop().time()
    recording_started.set()
    log.info("Recording started (%.2fs window).", duration_sec)

    offset = 0
    chunks_sent = 0

    while offset < len(pcm_data):
        end = min(offset + chunk_bytes, len(pcm_data))
        chunk = pcm_data[offset:end]
        num_samples = len(chunk) // SAMPLE_WIDTH

        frame = rtc.AudioFrame(
            data=chunk,
            sample_rate=PUBLISH_SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=num_samples,
        )
        await source.capture_frame(frame)
        offset = end
        chunks_sent += 1
        await asyncio.sleep(CHUNK_DURATION_MS / 1000)

    log.info("Finished streaming %d chunks (%.2fs of audio).", chunks_sent, duration_sec)

    # ── Wait until exactly input-duration has elapsed since start ─────
    elapsed = asyncio.get_event_loop().time() - stream_start_time
    remaining_wait = duration_sec - elapsed
    if remaining_wait > 0:
        log.info("Waiting %.2fs for recording window to finish …", remaining_wait)
        await asyncio.sleep(remaining_wait)

    recording_stop.set()
    log.info("Recording window closed.")

    # Give the receive task a moment to finish its current frame
    if receiving_task and not receiving_task.done():
        await asyncio.sleep(0.1)
        if not receiving_task.done():
            receiving_task.cancel()
            try:
                await receiving_task
            except asyncio.CancelledError:
                pass

    # ── Save outputs (exact same duration as input) ────────────────────
    flac_path = write_outputs(output_path, output_buf, CAPTURE_SAMPLE_RATE)
    actual_speech = np.count_nonzero(output_buf) / CAPTURE_SAMPLE_RATE
    log.info(
        "Saved: %s + %s  (%.2fs, %d Hz, ~%.2fs of non-silence)",
        output_path, flac_path, duration_sec, CAPTURE_SAMPLE_RATE, actual_speech,
    )

    # ── Cleanup ───────────────────────────────────────────────────────
    await room.disconnect()
    log.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stream a local audio file to a LiveKit room and capture the agent's "
            "audio response. Requires lk_agent.py running in a separate terminal."
        ),
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to the input audio file to send (wav, flac, …).",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Path for the output WAV file (a .flac copy is written alongside).",
    )
    parser.add_argument(
        "--room", default="test-room",
        help="LiveKit room name (default: test-room).",
    )
    parser.add_argument(
        "--silence", type=float, default=0.0,
        help="Trailing silence (seconds) appended to the input stream (default: 0).",
    )
    args = parser.parse_args()
    asyncio.run(run(args.input, args.output, args.room, args.silence))


if __name__ == "__main__":
    main()
