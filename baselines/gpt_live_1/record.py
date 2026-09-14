#!/usr/bin/env python3
"""Record a benchmark split through OpenAI GPT-Live-1 (generative protocol).

GPT-Live-1 (`gpt-live-1`, the /v1/live/sessions API) is a full-duplex voice
model with no detector interface: it emits no speech-start/stop or turn
events, only transcript deltas and output audio. Like `baselines/gemini_vad`
and `baselines/moshi_vad`, its turn-taking is therefore read off the audio it
produces: each conversation x direction streams speaker_K's channel into a
Live session paced at real time and records the agent's output audio
sample-aligned with the input, into the layout the readouts expect:

    <out>/<task_id>/speaker_K/output.flac      (agent audio, 24 kHz mono)
    <out>/<task_id>/speaker_K/events.jsonl     (transcripts, delegations, timing)

Alignment: output audio arrives as `session.output_audio.delta` events with
no playback timing. The recorder simulates a jitter-free client player: each
delta is placed at max(arrival offset on the input clock, end of the previous
delta), which is what a listener would hear. The event's own `start_ms`
(session timeline) is logged alongside for cross-checking.

Delegation: the session runs in client-delegation mode with no backend. The
prompt tells the model there is none; if it delegates anyway, the recorder
answers the delegation with a quiet `thinking.append` saying so, so the
conversation does not stall.

Resume-safe at direction granularity: `.done` is written only after every
input chunk was delivered; rerun the same command to redo unfinished
directions. Pilot with `--limit 1 --max-seconds 90` first.

    uv run --exclude-newer 2026-12-31 --with 'openai[realtime]==3.13.0' \
        --with scipy --with python-dotenv \
        python baselines/gpt_live_1/record.py --split dev --limit 1 --max-seconds 90

    uv run --exclude-newer 2026-12-31 --with 'openai[realtime]==3.13.0' \
        --with scipy --with python-dotenv \
        python baselines/gpt_live_1/record.py --split test --parallel 20
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import snapshot_download
from scipy.signal import resample_poly

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

from dotenv import load_dotenv  # noqa: E402

from turnbench.data import DEV_DATASET, PINNED_REVISIONS  # noqa: E402

TEST_DATASET = "mundo-ai/turn-benchmark-test"  # public: audio yes, labels no
MODEL = "gpt-live-1"
VOICE = "marin"
SR = 24_000  # Live API: one PCM16 format for input and output
CHUNK_MS = 20
SAMPLE_WIDTH = 2
CHECKPOINT_S = 30.0
CLOSE_TIMEOUT_S = 15.0
MAX_SESSIONS = 6  # reconnect budget per direction
DEFAULT_OUT = _HERE / "recordings"
DEFAULT_PROMPT = (_HERE / "system_prompt.txt").read_text().strip()
NO_BACKEND_NOTE = (
    "There is no backend in this session. Answer from your own knowledge, "
    "or say briefly that you do not know, and continue the conversation."
)


def _api_key() -> str:
    load_dotenv(_REPO / ".env")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("OPENAI_API_KEY not set (env or repo-root .env)")
    return key


# ---- input audio ----------------------------------------------------------------

def _pcm16_mono(data: np.ndarray, sr: int, peak_dbfs: float | None) -> bytes:
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        g = gcd(SR, sr)
        data = resample_poly(data, SR // g, sr // g)
    if peak_dbfs is not None:
        peak = np.abs(data).max()
        if peak > 0:
            data = data * (10 ** (peak_dbfs / 20.0) / peak)
    return np.clip(np.round(data * 32767), -32768, 32767).astype(np.int16).tobytes()


def _shard_files(source: str) -> list[str]:
    if Path(source).is_dir():
        return sorted(str(p) for p in Path(source).glob("*.parquet"))
    snapshot = snapshot_download(
        source, repo_type="dataset", revision=PINNED_REVISIONS.get(source),
        allow_patterns="*.parquet",
    )
    return sorted(str(p) for p in Path(snapshot).rglob("*.parquet"))


def dataset_index(source: str) -> dict[str, tuple[str, int]]:
    """{conversation_id: (parquet_path, row_group)} — reads only the id column."""
    index: dict[str, tuple[str, int]] = {}
    for shard in _shard_files(source):
        parquet = pq.ParquetFile(shard)
        assert parquet.metadata.num_rows == parquet.metadata.num_row_groups, shard
        ids = parquet.read(columns=["conversation_id"])["conversation_id"].to_pylist()
        for row_group, cid in enumerate(ids):
            index[cid] = (shard, row_group)
    return index


def _load_input(shard: str, row_group: int, speaker: int,
                max_seconds: float | None, peak_dbfs: float | None) -> bytes:
    table = pq.ParquetFile(shard).read_row_group(
        row_group, columns=[f"speaker_{speaker}_audio"]
    )
    cell = table[f"speaker_{speaker}_audio"][0].as_py()
    raw = cell["bytes"] if isinstance(cell, dict) else cell
    data, sr = sf.read(io.BytesIO(raw), dtype="float64", always_2d=False)
    if max_seconds is not None:
        data = data[: int(max_seconds * sr)]
    return _pcm16_mono(data, sr, peak_dbfs)


# ---- one Live session ------------------------------------------------------------

class _Recording:
    """Output buffer + simulated player cursor for one direction."""

    def __init__(self, n_samples: int, t_anchor: float, events_path: Path):
        self.buf = np.zeros(n_samples, dtype=np.int16)
        self.t_anchor = t_anchor
        self.cursor = 0          # next free sample of the simulated player
        self.deltas = 0
        self.dropped = 0
        self.events_path = events_path
        self._events = events_path.open("a", encoding="utf-8")

    def input_pos(self) -> int:
        return int((time.monotonic() - self.t_anchor) * SR)

    def place(self, pcm: bytes, start_ms: int | None) -> None:
        samples = np.frombuffer(pcm, dtype=np.int16)
        arrival = self.input_pos()
        pos = max(arrival, self.cursor)
        n = min(len(samples), len(self.buf) - pos)
        if n > 0:
            self.buf[pos:pos + n] = samples[:n]
        else:
            self.dropped += 1
        self.cursor = pos + len(samples)
        self.deltas += 1
        if self.deltas <= 3 or self.deltas % 200 == 0:
            self.log({"type": "audio.delta", "arrival_s": round(arrival / SR, 3),
                      "placed_s": round(pos / SR, 3), "n": int(len(samples)),
                      "start_ms": start_ms})

    def log(self, obj: dict) -> None:
        obj["wall_s"] = round(time.monotonic() - self.t_anchor, 3)
        self._events.write(json.dumps(obj) + "\n")
        self._events.flush()

    def close(self) -> None:
        self._events.close()


def _session_config(instructions: str) -> dict:
    return {
        "model": MODEL,
        "instructions": instructions,
        "audio": {"format": {"type": "audio/pcm", "rate": SR},
                  "output": {"voice": VOICE}},
        "delegation": {"type": "client"},
    }


async def _receive(conn, rec: _Recording, stop_at: float, log: logging.Logger,
                   started: asyncio.Event, closed: asyncio.Event) -> None:
    async for event in conn:
        et = event.type
        if et == "session.output_audio.delta":
            rec.place(base64.b64decode(event.delta), getattr(event, "start_ms", None))
        elif et in ("session.input_transcript.delta", "session.output_transcript.delta"):
            rec.log({"type": et, "delta": event.delta,
                     "start_ms": event.start_ms, "end_ms": event.end_ms})
        elif et == "session.started":
            started.set()
            rec.log({"type": et, "session_id": event.session.id})
        elif et == "session.delegation.created":
            did = event.delegation.id
            rec.log({"type": et, "delegation_id": did, "offset_ms": event.offset_ms})
            try:
                await conn.session.thinking.append(delegation_id=did, content=NO_BACKEND_NOTE)
            except Exception as e:  # noqa: BLE001
                log.warning("thinking.append failed: %s", e)
        elif et == "error":
            rec.log({"type": et, "error": event.error.model_dump()})
            log.warning("server error: %s", event.error)
        elif et == "session.closed":
            rec.log({"type": et, "reason": getattr(event, "reason", None),
                     "usage": event.usage.model_dump() if event.usage else None})
            closed.set()
            return
        elif et in ("session.usage.updated", "session.thinking.appended", "info"):
            pass
        else:
            rec.log({"type": et})


async def _send(conn, pcm: bytes, rec: _Recording, counter: list[int]) -> None:
    chunk_bytes = SR * CHUNK_MS // 1000 * SAMPLE_WIDTH
    total = len(pcm) // chunk_bytes
    while counter[0] < total:
        idx = counter[0]
        slack = rec.t_anchor + idx * CHUNK_MS / 1000 - time.monotonic()
        if slack > 0:
            await asyncio.sleep(slack)
        b0 = idx * chunk_bytes
        await conn.session.input_audio.append(
            audio=base64.b64encode(pcm[b0:b0 + chunk_bytes]).decode("ascii"))
        counter[0] = idx + 1


async def run_direction(pcm_in: bytes, out_dir: Path, instructions: str,
                        log: logging.Logger) -> bool:
    """Stream one channel through Live; write output.flac. True iff every
    input chunk was delivered (partial audio is saved without .done)."""
    from openai import AsyncOpenAI

    out_dir.mkdir(parents=True, exist_ok=True)
    duration_s = len(pcm_in) / SAMPLE_WIDTH / SR
    t_anchor = time.monotonic()
    stop_at = t_anchor + duration_s
    rec = _Recording(int(duration_s * SR), t_anchor, out_dir / "events.jsonl")
    wav_path, flac_path = out_dir / "output.wav", out_dir / "output.flac"
    chunk_bytes = SR * CHUNK_MS // 1000 * SAMPLE_WIDTH
    total = len(pcm_in) // chunk_bytes
    counter = [0]

    def _save(path: Path) -> None:
        sf.write(str(path), rec.buf, SR, subtype="PCM_16")

    async def _checkpoint() -> None:
        while True:
            await asyncio.sleep(CHECKPOINT_S)
            _save(wav_path)
            log.info("checkpoint %.0fs: %d chunks sent, %d deltas, %.1fs audio",
                     time.monotonic() - t_anchor, counter[0], rec.deltas,
                     np.count_nonzero(rec.buf) / SR)

    client = AsyncOpenAI(api_key=_api_key())
    ckpt = asyncio.create_task(_checkpoint())
    try:
        for session_num in range(1, MAX_SESSIONS + 1):
            if counter[0] >= total:
                break
            log.info("session %d: resume at chunk %d/%d", session_num, counter[0], total)
            started, closed = asyncio.Event(), asyncio.Event()
            try:
                async with client.live.connect() as conn:
                    recv = asyncio.create_task(
                        _receive(conn, rec, stop_at, log, started, closed))
                    await conn.session.start(session=_session_config(instructions))
                    await asyncio.wait_for(started.wait(), timeout=30)
                    await _send(conn, pcm_in, rec, counter)
                    remaining = stop_at - time.monotonic()
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                    await conn.session.close()
                    try:
                        await asyncio.wait_for(closed.wait(), timeout=CLOSE_TIMEOUT_S)
                    except asyncio.TimeoutError:
                        log.warning("no session.closed within %.0fs", CLOSE_TIMEOUT_S)
                    recv.cancel()
                    await asyncio.gather(recv, return_exceptions=True)
            except Exception as e:  # noqa: BLE001  transport / server errors
                log.warning("session %d ended early at chunk %d: %r",
                            session_num, counter[0], e)
                await asyncio.sleep(min(30.0, 2.0 ** session_num))
    finally:
        ckpt.cancel()
        await asyncio.gather(ckpt, return_exceptions=True)
        rec.log({"type": "summary", "chunks_sent": counter[0], "chunks_total": total,
                 "deltas": rec.deltas, "dropped": rec.dropped,
                 "audio_s": round(float(np.count_nonzero(rec.buf) / SR), 2)})
        rec.close()
        _save(flac_path)
        wav_path.unlink(missing_ok=True)
    complete = counter[0] >= total
    log.info("%s: %s, %.1fs agent audio", out_dir,
             "complete" if complete else "PARTIAL", np.count_nonzero(rec.buf) / SR)
    return complete


# ---- batch driver ----------------------------------------------------------------

_CONNECT_GATE = asyncio.Lock()
_CONNECT_SPACING_S = 0.25


async def _record(tid: str, speaker: int, shard: str, row_group: int,
                  out_root: Path, instructions: str, sem: asyncio.Semaphore,
                  log: logging.Logger, max_seconds: float | None) -> bool:
    out_dir = out_root / tid / f"speaker_{speaker}"
    if (out_dir / ".done").exists():
        return True
    async with sem:
        pcm = await asyncio.to_thread(_load_input, shard, row_group, speaker,
                                      max_seconds, -3.0)
        async with _CONNECT_GATE:
            await asyncio.sleep(_CONNECT_SPACING_S)
        dlog = logging.getLogger(f"gpt_live.{tid}.s{speaker}")
        complete = await run_direction(pcm, out_dir, instructions, dlog)
    if complete:
        (out_dir / ".done").touch()
    else:
        log.error("PARTIAL %s/speaker_%d — rerun to retry", tid, speaker)
    return complete


def _split_ids(split: str) -> list[str]:
    text = (_REPO / "turnbench" / "splits" / f"{split}.txt").read_text()
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]


async def _main_async(args: argparse.Namespace, log: logging.Logger) -> int:
    source = args.dataset or (TEST_DATASET if args.split == "test" else DEV_DATASET)
    index = dataset_index(source)
    ids = _split_ids(args.split)
    missing = sorted(set(ids) - set(index), key=int)
    if missing:
        sys.exit(f"{len(missing)} {args.split} conversations missing from {source}: {missing}")
    wanted = ids[: args.limit] if args.limit else ids
    instructions = args.system_prompt_file.read_text().strip()
    sem = asyncio.Semaphore(args.parallel)
    jobs = [_record(tid, k, *index[tid], args.out, instructions, sem, log,
                    args.max_seconds)
            for tid in wanted for k in (1, 2)]
    log.info("split=%s conversations=%d directions=%d parallel=%d out=%s",
             args.split, len(wanted), len(jobs), args.parallel, args.out)
    results = await asyncio.gather(*jobs)
    n_ok = sum(results)
    log.info("finished: %d/%d directions complete", n_ok, len(results))
    return 0 if n_ok == len(results) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--dataset", default=None, help="HF repo id or local parquet dir")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--parallel", type=int, default=4, help="concurrent Live sessions")
    ap.add_argument("--limit", type=int, default=None, help="first N conversations")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="truncate input (pilot only — never for a real run)")
    ap.add_argument("--system-prompt-file", type=Path, default=_HERE / "system_prompt.txt")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    return asyncio.run(_main_async(args, logging.getLogger("gpt_live")))


if __name__ == "__main__":
    raise SystemExit(main())
