#!/usr/bin/env python3
"""
Moshi inference script for the dev_release_flac dataset (continuous streaming).

Streams each speaker's audio to the Moshi server in real time while recording
the model's response. No turn-taking: input is never paused.

Dataset layout (input):
    <input_dir>/<number>/speaker_1_audio.flac
    <input_dir>/<number>/speaker_2_audio.flac

Output layout (mirrors the dataset):
    <output_dir>/<number>/speaker_1/speaker_1_audio.flac   (copy of input)
    <output_dir>/<number>/speaker_1/output.wav             (Moshi response)
    <output_dir>/<number>/speaker_1/output.flac            (same, FLAC)
    <output_dir>/<number>/speaker_2/...                    (same for speaker 2)

Usage:
    python inference_moshi_dev_release.py --server_ip <ip:port> \
        --input /path/to/dev_release_flac --output /path/to/results \
        [--silence-threshold 15] [--post-input-wait 20000] \
        [--final-silence-timeout 10000] [--max-jobs 1]
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import gcd
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import sphn
from scipy.signal import resample_poly
import websockets
import websockets.exceptions as wsex

# ──────────────────────────────────────────────────────────────────────────────
# Audio parameters (matching Moshi server expectations)
# ──────────────────────────────────────────────────────────────────────────────
SEND_SR = 24_000  # Server expected sample rate
FRAME_SMP = 1_920  # 80 ms @ 24 kHz
FRAME_MS = 80  # Frame duration in milliseconds
SKIP_FRAMES = 1  # Matches server's skip_frames setting
FRAME_SEC = FRAME_SMP / SEND_SR

DEFAULT_SILENCE_THRESHOLD = 15  # RMS threshold - samples below this are considered silent
DEFAULT_POST_INPUT_WAIT_MS = 10000  # Max wait for model to start responding after input finishes
DEFAULT_FINAL_SILENCE_TIMEOUT_MS = 5000  # Silence after model response = end conversation

SPEAKER_FILES = ["speaker_1_audio.flac", "speaker_2_audio.flac"]


# ──────────────────────────────────────────────────────────────────────────────
# CLI — tunable arguments
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Moshi streaming client for the dev_release_flac dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python inference_moshi_dev_release.py --server_ip localhost:8998 \\
        --input ./dev_release_flac --output ./results
        """,
    )
    ap.add_argument("--server_ip", required=True, help="Moshi server host[:port] or full URL")
    ap.add_argument("--input", required=True, help="Dataset root with numbered folders containing speaker_{1,2}_audio.flac")
    ap.add_argument("--output", required=True, help="Output directory (mirrors dataset hierarchy)")
    ap.add_argument("--max-jobs", type=int, default=1, help="Maximum parallel jobs (default: 1)")
    ap.add_argument(
        "--silence-threshold", type=int, default=DEFAULT_SILENCE_THRESHOLD,
        help=f"RMS threshold for model silence detection (default: {DEFAULT_SILENCE_THRESHOLD})"
    )
    ap.add_argument(
        "--post-input-wait", type=int, default=DEFAULT_POST_INPUT_WAIT_MS,
        help=f"Max wait in ms for model to respond after input ends (default: {DEFAULT_POST_INPUT_WAIT_MS})"
    )
    ap.add_argument(
        "--final-silence-timeout", type=int, default=DEFAULT_FINAL_SILENCE_TIMEOUT_MS,
        help=f"Silence in ms after model response to end conversation (default: {DEFAULT_FINAL_SILENCE_TIMEOUT_MS})"
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    ap.add_argument("--start-from", type=int, default=0, help="Start processing from this folder number")
    ap.add_argument("--end-at", type=int, default=None, help="Stop processing at this folder number")
    ap.add_argument("--sleep-between", type=int, default=15, help="Sleep in seconds between inferences (default: 15)")

    args = ap.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    server_url = _ws_url(args.server_ip)

    print("==========================================")
    print("Moshi Inference Configuration:")
    print(f"  Server URL:             {server_url}")
    print(f"  Input directory:        {input_dir}")
    print(f"  Output directory:       {output_dir}")
    print(f"  Max parallel jobs:      {args.max_jobs}")
    print(f"  Silence threshold:      {args.silence_threshold} (RMS)")
    print(f"  Post-input wait:        {args.post_input_wait} ms")
    print(f"  Final silence timeout:  {args.final_silence_timeout} ms")
    print(f"  Start from folder:      {args.start_from}")
    print(f"  End at folder:          {args.end_at if args.end_at is not None else 'unlimited'}")
    print(f"  Sleep between jobs:     {args.sleep_between}s")
    print(f"  Overwrite:              {args.overwrite}")
    print("==========================================")
    print()

    error_log = output_dir / "errors.log"
    with open(error_log, "w") as f:
        f.write(f"Processing started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Server URL: {server_url}\n\n")

    jobs, skipped_count = collect_jobs(input_dir, output_dir, args.start_from, args.end_at, args.overwrite)
    if not jobs and skipped_count == 0:
        print(f"No speaker audio files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(jobs)} inference jobs to run ({skipped_count} skipped)")
    print()

    processed_count = 0
    failed_count = 0

    def record_result(job_name: str, success: bool, message: str):
        nonlocal processed_count, failed_count
        if success:
            processed_count += 1
        else:
            failed_count += 1
            with open(error_log, "a") as f:
                f.write(f"[{job_name}] {message}\n")

    if args.max_jobs == 1:
        for i, (job_name, input_flac, work_dir) in enumerate(jobs):
            if i > 0 and args.sleep_between > 0:
                print(f"[SLEEP] Waiting {args.sleep_between}s before next job...")
                time.sleep(args.sleep_between)
            record_result(*process_speaker(
                server_url=server_url,
                input_flac=input_flac,
                work_dir=work_dir,
                job_name=job_name,
                silence_threshold=args.silence_threshold,
                post_input_wait_ms=args.post_input_wait,
                final_silence_timeout_ms=args.final_silence_timeout,
            ))
    else:
        with ThreadPoolExecutor(max_workers=args.max_jobs) as executor:
            futures = [
                executor.submit(
                    process_speaker,
                    server_url=server_url,
                    input_flac=input_flac,
                    work_dir=work_dir,
                    job_name=job_name,
                    silence_threshold=args.silence_threshold,
                    post_input_wait_ms=args.post_input_wait,
                    final_silence_timeout_ms=args.final_silence_timeout,
                )
                for job_name, input_flac, work_dir in jobs
            ]
            for future in as_completed(futures):
                record_result(*future.result())

    print()
    print("==========================================")
    print("Processing complete!")
    print(f"  Processed:              {processed_count} jobs")
    print(f"  Skipped:                {skipped_count} jobs")
    print(f"  Failed:                 {failed_count} jobs")
    print(f"  Output directory:       {output_dir}")
    if failed_count > 0:
        print(f"  Check {error_log} for errors.")
    print("==========================================")


# ──────────────────────────────────────────────────────────────────────────────
# sphn API compatibility
# ──────────────────────────────────────────────────────────────────────────────
def _check_sphn_api() -> bool:
    """Newer sphn versions use separate append/read methods."""
    return hasattr(sphn.OpusStreamWriter(SEND_SR), "read_bytes")


SPHN_HAS_SEPARATE_READ = _check_sphn_api()


# ──────────────────────────────────────────────────────────────────────────────
# Audio utilities
# ──────────────────────────────────────────────────────────────────────────────
def _mono(x: np.ndarray) -> np.ndarray:
    """Ensure mono audio."""
    return x if x.ndim == 1 else x.mean(axis=1)


def _resample(x: np.ndarray, sr: int, tgt: int) -> np.ndarray:
    """int16 → int16 polyphase resample using scipy."""
    if sr == tgt:
        return x
    g = gcd(sr, tgt)
    y = resample_poly(x.astype(np.float64), tgt // g, sr // g)
    return np.clip(y, -32768, 32767).astype(np.int16)


def _chunk(sig: np.ndarray) -> List[np.ndarray]:
    """Split into 80ms frames (zero-padded to a whole number of frames)."""
    pad = (-len(sig)) % FRAME_SMP
    if pad:
        sig = np.concatenate([sig, np.zeros(pad, sig.dtype)])
    return [sig[i : i + FRAME_SMP] for i in range(0, len(sig), FRAME_SMP)]


def calculate_rms(samples: np.ndarray) -> float:
    """RMS of audio samples, scaled to int16 range for float input."""
    if len(samples) == 0:
        return 0.0
    if samples.dtype in (np.float32, np.float64):
        samples = samples * 32768
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def format_time() -> str:
    return time.strftime("%H:%M:%S")


# ──────────────────────────────────────────────────────────────────────────────
# Moshi client (continuous streaming)
# ──────────────────────────────────────────────────────────────────────────────
class MoshiClient:
    def __init__(
        self,
        ws_url: str,
        inp: Path,
        out: Path,
        silence_threshold: int = DEFAULT_SILENCE_THRESHOLD,
        post_input_wait_ms: int = DEFAULT_POST_INPUT_WAIT_MS,
        final_silence_timeout_ms: int = DEFAULT_FINAL_SILENCE_TIMEOUT_MS,
    ):
        self.url = ws_url
        self.out = out
        self.silence_threshold = silence_threshold
        self.post_input_wait_ms = post_input_wait_ms
        self.final_silence_timeout_ms = final_silence_timeout_ms

        sig, sr = sf.read(inp, dtype="int16")
        self.sig24 = _resample(_mono(sig), sr, SEND_SR)
        self.max_samples = len(self.sig24)
        self.input_duration_sec = len(self.sig24) / SEND_SR

        self.writer = sphn.OpusStreamWriter(SEND_SR)
        self.reader = sphn.OpusStreamReader(SEND_SR)

        # Shared state between send/recv coroutines
        self.input_finished = False
        self.last_model_speak_time: float | None = None
        self.should_stop = False
        self.state_lock = asyncio.Lock()

        self.silence_frame = np.zeros(FRAME_SMP, dtype=np.int16)

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        pcm = frame.astype(np.float32) / 32768
        if SPHN_HAS_SEPARATE_READ:
            self.writer.append_pcm(pcm)
            return self.writer.read_bytes()
        return self.writer.append_pcm(pcm)

    def _decode_audio(self, payload: bytes) -> np.ndarray:
        if SPHN_HAS_SEPARATE_READ:
            self.reader.append_bytes(payload)
            chunks = []
            while True:
                pcm = self.reader.read_pcm()
                if pcm.size == 0:
                    break
                chunks.append(pcm)
            return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        pcm = self.reader.append_bytes(payload)
        return pcm if pcm is not None else np.array([], dtype=np.float32)

    async def _send(self, ws):
        """Stream input frames in real time, then silence until the receiver stops."""
        tx_frames = _chunk(self.sig24)
        total_frames = len(tx_frames)
        last_progress_log = time.time()

        print(f"[{format_time()}] [SEND] Streaming {total_frames} frames ({self.input_duration_sec:.2f}s)...")

        for frame_idx, frame in enumerate(tx_frames):
            if self.should_stop:
                break
            frame_start = time.time()

            encoded = self._encode_frame(frame)
            if encoded:
                await ws.send(b"\x01" + encoded)

            if time.time() - last_progress_log > 5.0:
                progress = (frame_idx + 1) / total_frames * 100
                elapsed_sec = (frame_idx + 1) * FRAME_MS / 1000
                print(f"[{format_time()}] [SEND] Progress: {progress:.1f}% ({elapsed_sec:.1f}s / {self.input_duration_sec:.1f}s)")
                last_progress_log = time.time()

            await asyncio.sleep(max(0.0, FRAME_SEC - (time.time() - frame_start)))

        print(f"[{format_time()}] [SEND] Input streaming finished.")
        async with self.state_lock:
            self.input_finished = True

        # Keep the connection alive with silence while waiting for the model to finish
        wait_start = time.time()
        max_wait_sec = (self.post_input_wait_ms + self.final_silence_timeout_ms) / 1000 + 5
        while not self.should_stop:
            encoded = self._encode_frame(self.silence_frame)
            if encoded:
                await ws.send(b"\x01" + encoded)
            await asyncio.sleep(FRAME_SEC)

            if time.time() - wait_start > max_wait_sec:
                print(f"[{format_time()}] [SEND] Max wait time exceeded, stopping")
                async with self.state_lock:
                    self.should_stop = True

        async with self.state_lock:
            self.should_stop = True

    async def _recv(self, ws):
        """Record model audio; end after the model falls silent post-input."""
        samples_written = 0
        first_pcm_seen = False
        model_responded_after_input = False

        print(f"[{format_time()}] [RECV] Starting receiver...")

        with sf.SoundFile(self.out, "w", samplerate=SEND_SR, channels=1, subtype="PCM_16") as fout:
            try:
                while not self.should_stop:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue

                    if not isinstance(msg, (bytes, bytearray)) or len(msg) == 0:
                        continue
                    if msg[0] != 1:  # only audio messages
                        continue

                    pcm = self._decode_audio(msg[1:])
                    if pcm.size == 0:
                        continue

                    # Pad for the server's skipped frames so output aligns with input
                    if not first_pcm_seen:
                        pad = min(SKIP_FRAMES * FRAME_SMP, self.max_samples)
                        fout.write(np.zeros(pad, dtype=np.int16))
                        samples_written += pad
                        first_pcm_seen = True

                    fout.write((pcm * 32768).astype(np.int16))
                    samples_written += pcm.size

                    # Track model voice activity (used only for end-of-conversation)
                    if calculate_rms(pcm) > self.silence_threshold:
                        async with self.state_lock:
                            self.last_model_speak_time = time.time()
                        if self.input_finished and not model_responded_after_input:
                            model_responded_after_input = True
                            print(f"[{format_time()}] [RECV] Model responding after input; will end after {self.final_silence_timeout_ms / 1000}s of silence.")

                    # End once the model has been silent long enough after the input
                    if (
                        self.input_finished
                        and model_responded_after_input
                        and self.last_model_speak_time is not None
                        and (time.time() - self.last_model_speak_time) * 1000 >= self.final_silence_timeout_ms
                    ):
                        print(f"[{format_time()}] [RECV] Model finished responding. Ending.")
                        async with self.state_lock:
                            self.should_stop = True
                        break

            except wsex.ConnectionClosedError as e:
                print(f"[{format_time()}] [RECV] Connection closed: {e}")
            except Exception as e:
                print(f"[{format_time()}] [RECV] Error: {e}")
                import traceback
                traceback.print_exc()

            # Pad with zeros if output is shorter than the input
            if samples_written < self.max_samples:
                fout.write(np.zeros(self.max_samples - samples_written, dtype=np.int16))
                samples_written = self.max_samples

            print(f"[{format_time()}] [RECV] Written {samples_written} samples to {self.out}")

        async with self.state_lock:
            self.should_stop = True

    async def _run(self):
        print(f"[{format_time()}] Input duration: {self.input_duration_sec:.2f}s")
        async with websockets.connect(self.url, max_size=None) as ws:
            await asyncio.gather(self._send(ws), self._recv(ws), return_exceptions=True)
        output_duration = sf.info(self.out).duration
        print(f"[{format_time()}] [DONE] input len = {self.input_duration_sec:.2f}s | output len = {output_duration:.2f}s")

    def run(self) -> bool:
        try:
            asyncio.run(self._run())
            return True
        except wsex.ConnectionClosedError as e:
            print(f"[{format_time()}] [WARN] Connection closed: {e}")
            return False
        except Exception as e:
            print(f"[{format_time()}] [ERR] {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return False


# ──────────────────────────────────────────────────────────────────────────────
# Batch processing
# ──────────────────────────────────────────────────────────────────────────────
def _ws_url(addr: str) -> str:
    """Convert user input address to ws://…/api/chat format."""
    if "://" in addr:
        proto, rest = addr.split("://", 1)
        proto = "ws" if proto in {"http", "ws"} else "wss"
        return f"{proto}://{rest.rstrip('/')}/api/chat"
    if ":" not in addr:
        addr += ":8998"
    return f"ws://{addr}/api/chat"


def process_speaker(
    server_url: str,
    input_flac: Path,
    work_dir: Path,
    job_name: str,
    silence_threshold: int,
    post_input_wait_ms: int,
    final_silence_timeout_ms: int,
) -> tuple[str, bool, str]:
    """Run one inference on a speaker file. Returns (job_name, success, message)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    output_wav = work_dir / "output.wav"
    output_flac = work_dir / "output.flac"

    print(f"[{format_time()}] Processing {job_name}: {input_flac}")

    try:
        shutil.copy2(input_flac, work_dir / input_flac.name)
    except Exception as e:
        return (job_name, False, f"Failed to copy input file: {e}")

    client = MoshiClient(
        ws_url=server_url,
        inp=input_flac,
        out=output_wav,
        silence_threshold=silence_threshold,
        post_input_wait_ms=post_input_wait_ms,
        final_silence_timeout_ms=final_silence_timeout_ms,
    )
    if not client.run():
        return (job_name, False, "Moshi inference failed")
    if not output_wav.exists():
        return (job_name, False, "output.wav not found after inference")

    try:
        audio, sr = sf.read(output_wav, dtype="int16")
        sf.write(output_flac, audio, sr, format="FLAC")
    except Exception as e:
        return (job_name, False, f"Failed to write output.flac: {e}")

    print(f"[{format_time()}] Completed {job_name}: output={sf.info(output_wav).duration:.2f}s")
    return (job_name, True, "Success")


def collect_jobs(input_dir: Path, output_dir: Path, start_from: int, end_at: int | None, overwrite: bool):
    """Yield (job_name, input_flac, work_dir) for every speaker file to process."""
    jobs = []
    skipped = 0
    for folder in sorted(input_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        index = int(folder.name)
        if index < start_from or (end_at is not None and index > end_at):
            continue
        for speaker_file in SPEAKER_FILES:
            input_flac = folder / speaker_file
            if not input_flac.exists():
                print(f"[WARN] Missing {input_flac}, skipping", file=sys.stderr)
                continue
            speaker = speaker_file.rsplit("_audio", 1)[0]  # "speaker_1" / "speaker_2"
            work_dir = output_dir / folder.name / speaker
            if not overwrite and (work_dir / "output.flac").exists():
                print(f"[SKIP] {folder.name}/{speaker} already exists, skipping...")
                skipped += 1
                continue
            jobs.append((f"{folder.name}/{speaker}", input_flac, work_dir))
    return jobs, skipped


if __name__ == "__main__":
    main()
