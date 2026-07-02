#!/usr/bin/env python3
"""Stream one audio file to Gemini Live directly (no LiveKit) and record
the agent's response sample-aligned with the input.

Bypasses LiveKit by talking to the Gemini Live API straight through
`google-genai`. Reads GEMINI_API_KEY (or GOOGLE_API_KEY) from a .env file.

Usage:
    python direct_gemini_client.py \\
        -i /path/to/speaker_1_audio.wav \\
        -o output.wav \\
        --model gemini-3.1-flash-live-preview \\
        --env-file pipeline/.env.local
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from scipy.signal import resample_poly

from google import genai
from google.genai import types
import google.genai.live as _genai_live
from websockets.exceptions import ConnectionClosed

_orig_ws_connect = _genai_live.ws_connect


def _patched_ws_connect(uri, **kwargs):
    kwargs.setdefault("ping_interval", None)
    kwargs.setdefault("close_timeout", 5)
    kwargs.setdefault("max_size", None)
    return _orig_ws_connect(uri, **kwargs)


_genai_live.ws_connect = _patched_ws_connect

INPUT_SR = 16_000
OUTPUT_SR = 24_000
CHUNK_MS = 20
SAMPLE_WIDTH = 2

DEFAULT_SYSTEM_PROMPT = (
    "Keep your responses concise and conversational since they will be "
    "spoken aloud. Respond naturally to whatever the user says or asks."
)


def _read_pcm16(path: Path, target_sr: int,
                normalize_peak_dbfs: float | None = None) -> bytes:
    data, sr = sf.read(str(path), dtype="float64")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        g = gcd(target_sr, sr)
        data = resample_poly(data, target_sr // g, sr // g)
    if normalize_peak_dbfs is not None:
        peak = np.abs(data).max()
        if peak > 0:
            target = 10 ** (normalize_peak_dbfs / 20.0)
            data = data * (target / peak)
    pcm = np.clip(np.round(data * 32767), -32768, 32767).astype(np.int16)
    return pcm.tobytes()


async def _send_audio(session, pcm: bytes, t_anchor: float, start_chunk: int,
                      counter: list, log: logging.Logger) -> None:
    """Stream input PCM at wall-clock pace, starting at chunk index
    `start_chunk`. `counter[0]` tracks the next chunk to send so the
    caller can resume across reconnects when this raises.
    """
    samples_per_chunk = INPUT_SR * CHUNK_MS // 1000
    chunk_bytes = samples_per_chunk * SAMPLE_WIDTH
    total_chunks = len(pcm) // chunk_bytes
    counter[0] = start_chunk
    while counter[0] < total_chunks:
        idx = counter[0]
        target_t = t_anchor + idx * (CHUNK_MS / 1000)
        slack = target_t - time.monotonic()
        if slack > 0:
            await asyncio.sleep(slack)
        b0 = idx * chunk_bytes
        b1 = b0 + chunk_bytes
        await session.send_realtime_input(
            audio=types.Blob(data=pcm[b0:b1], mime_type=f"audio/pcm;rate={INPUT_SR}")
        )
        counter[0] = idx + 1


async def _recv_audio(session, out_buf: np.ndarray, t_start: float,
                      stop_at: float, log: logging.Logger) -> None:
    """Sample-align agent audio to the input timeline.

    Gemini Live bursts audio faster than real-time as variable-size
    chunks (one model turn = many frames over a shorter wall-clock
    window). Each turn's frames form a contiguous stream: we
    concatenate them and anchor the start of each turn at the first
    frame's wall-clock arrival offset. `session.receive()` only yields
    one turn — we call it in an outer loop until the input duration
    elapses so multi-turn responses are all captured.

    When Gemini barges-in / cancels its own turn it sets
    `server_content.interrupted=True`. The audio it already sent ahead
    of real-time would NOT have reached the user, so we truncate the
    turn at the wall-clock moment of the interrupt event.
    """
    target_samples = out_buf.shape[0]
    grand_total = 0
    turns = 0
    interrupted_count = 0

    def _flush(anchor: int, frames: list[np.ndarray],
               cut_samples: int | None) -> int:
        if not frames:
            return 0
        block = np.concatenate(frames)
        if cut_samples is not None:
            block = block[:max(0, cut_samples)]
        n = min(len(block), target_samples - anchor)
        if n > 0:
            out_buf[anchor:anchor + n] = block[:n]
        return n

    while time.monotonic() < stop_at:
        turn_buf: list[np.ndarray] = []
        turn_anchor_pos: int | None = None
        cut_samples: int | None = None
        try:
            async for response in session.receive():
                if time.monotonic() > stop_at:
                    break
                sc = getattr(response, "server_content", None)
                if sc is not None and getattr(sc, "interrupted", False):
                    if turn_anchor_pos is not None and cut_samples is None:
                        played = int(
                            (time.monotonic() - t_start) * OUTPUT_SR) - turn_anchor_pos
                        cut_samples = max(0, played)
                        interrupted_count += 1
                data = response.data
                if data:
                    samples = np.frombuffer(data, dtype=np.int16)
                    if turn_anchor_pos is None:
                        turn_anchor_pos = int(
                            (time.monotonic() - t_start) * OUTPUT_SR)
                        if turn_anchor_pos >= target_samples:
                            break
                    turn_buf.append(samples)
        except Exception as e:
            log.warning("receive() error: %s", e)
            break
        if turn_anchor_pos is not None:
            written = _flush(turn_anchor_pos, turn_buf, cut_samples)
            grand_total += written
            turns += 1
            tag = " (INTERRUPTED)" if cut_samples is not None else ""
            log.info("Turn %d: %.2fs at %.2fs offset%s.", turns,
                     written / OUTPUT_SR, turn_anchor_pos / OUTPUT_SR, tag)
    log.info("Received %.2fs of model audio across %d turn(s), %d interrupted.",
             grand_total / OUTPUT_SR, turns, interrupted_count)


async def run(input_path: Path, output_path: Path, model: str,
              system_prompt: str, voice: str, log: logging.Logger,
              normalize_peak_dbfs: float | None) -> bool:
    """Stream `input_path` into Gemini Live; write output.wav/.flac.
    Returns True iff every input chunk was sent (clean, complete run) —
    False means the saved audio is partial (reconnect budget exhausted),
    so batch drivers must NOT mark this direction done."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY/GOOGLE_API_KEY not set (check --env-file).")

    pcm_in = _read_pcm16(input_path, INPUT_SR, normalize_peak_dbfs)
    input_duration_s = len(pcm_in) / SAMPLE_WIDTH / INPUT_SR
    log.info("Input: %s (%.2fs at %d Hz mono pcm16, normalize=%s)",
             input_path, input_duration_s, INPUT_SR,
             f"peak={normalize_peak_dbfs}dBFS" if normalize_peak_dbfs is not None else "none")

    out_samples = int(input_duration_s * OUTPUT_SR)
    out_buf = np.zeros(out_samples, dtype=np.int16)

    client = genai.Client(api_key=api_key)
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=types.Content(parts=[types.Part(text=system_prompt)]),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
    )

    samples_per_chunk = INPUT_SR * CHUNK_MS // 1000
    chunk_bytes = samples_per_chunk * SAMPLE_WIDTH
    total_chunks = len(pcm_in) // chunk_bytes

    t_anchor = time.monotonic()
    stop_at = t_anchor + input_duration_s
    counter = [0]
    session_num = 0

    def _save() -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), out_buf, OUTPUT_SR, subtype="PCM_16")

    async def _checkpoint(period_s: float = 30.0) -> None:
        while True:
            await asyncio.sleep(period_s)
            _save()
            nonzero_s = np.count_nonzero(out_buf) / OUTPUT_SR
            elapsed = time.monotonic() - t_anchor
            log.info("Checkpoint @ %.0fs wall (%.2fs non-silence saved).",
                     elapsed, nonzero_s)

    chkpt_task = asyncio.create_task(_checkpoint())

    recv_task = None
    try:
        while counter[0] < total_chunks:
            session_num += 1
            start_chunk = counter[0]
            log.info("Session %d: connecting (resume at chunk %d / %d, %.1fs in) …",
                     session_num, start_chunk, total_chunks,
                     start_chunk * CHUNK_MS / 1000)
            try:
                async with client.aio.live.connect(
                        model=model, config=config) as session:
                    recv_task = asyncio.create_task(
                        _recv_audio(session, out_buf, t_anchor, stop_at, log))
                    send_task = asyncio.create_task(
                        _send_audio(session, pcm_in, t_anchor, start_chunk,
                                    counter, log))
                    await send_task
                    log.info("Session %d: send complete at chunk %d.",
                             session_num, counter[0])
                    remaining = stop_at - time.monotonic()
                    if remaining > 0:
                        log.info("Holding recv for trailing %.2fs …", remaining)
                        await asyncio.sleep(remaining)
                    recv_task.cancel()
                    try:
                        await recv_task
                    except asyncio.CancelledError:
                        pass
            except ConnectionClosed as e:
                # Reap this session's receiver before reconnecting — otherwise
                # it leaks and races the next session's receiver on out_buf.
                if recv_task is not None:
                    recv_task.cancel()
                    try:
                        await recv_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    recv_task = None
                log.warning("Session %d closed at chunk %d / %d (%.1fs/%.1fs): %s",
                            session_num, counter[0], total_chunks,
                            counter[0] * CHUNK_MS / 1000, input_duration_s,
                            str(e)[:120])
                await asyncio.sleep(0.5)
            if session_num >= 20:
                log.warning("Hit max session count; stopping.")
                break
    finally:
        chkpt_task.cancel()
        try:
            await chkpt_task
        except asyncio.CancelledError:
            pass
        _save()
        flac_path = output_path.with_suffix(".flac")
        sf.write(str(flac_path), out_buf, OUTPUT_SR, format="FLAC",
                 subtype="PCM_16")
        nonzero_s = np.count_nonzero(out_buf) / OUTPUT_SR
        log.info("Saved %s + %s (%.2fs total, %.2fs non-silence, %d session(s)).",
                 output_path, flac_path, input_duration_s, nonzero_s,
                 session_num)
    return counter[0] >= total_chunks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", type=Path, required=True)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--model", default="gemini-3.1-flash-live-preview")
    ap.add_argument("--voice", default=os.environ.get("GOOGLE_VOICE", "Puck"))
    ap.add_argument("--env-file", type=Path,
                    default=Path(__file__).resolve().parent / ".env.local",
                    help="dotenv with GOOGLE_API_KEY (see .env.local.example)")
    ap.add_argument("--system-prompt-file", type=Path,
                    default=Path(__file__).resolve().parent / "system_prompt.txt")
    ap.add_argument("--normalize-peak-dbfs", type=float, default=-3.0,
                    help="Normalize input peak to this dBFS before sending; "
                         "pass NaN to disable (default: -3 dBFS)")
    args = ap.parse_args()
    norm = None if args.normalize_peak_dbfs != args.normalize_peak_dbfs else args.normalize_peak_dbfs

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("direct_gemini")

    if args.env_file.exists():
        load_dotenv(args.env_file, override=False)
        log.info("Loaded env from %s", args.env_file)

    if args.system_prompt_file.exists():
        system_prompt = args.system_prompt_file.read_text().strip()
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    complete = asyncio.run(run(args.input, args.output, args.model, system_prompt,
                               args.voice, log, norm))
    if not complete:
        log.error("Recording INCOMPLETE — not every input chunk was sent; "
                  "exit 1 so batch drivers do not mark this direction done.")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
