#!/usr/bin/env python3
"""Record a whole split through Gemini Live, straight from the HF dataset.

Batch driver for `direct_gemini_client.run`: audio comes from the benchmark
parquet shards, read lazily one row group at a time exactly like
baselines/openai_realtime.py — no local Mundo delivery layout, and no
materialising the whole split in RAM (~24 GB for test; loading it eagerly
gets the process OOM-killed). Each conversation × direction streams
speaker_K's channel into a Gemini Live session in real time and records the
agent's output sample-aligned with the input, into the layout the readouts
expect:

    <out>/<task_id>/speaker_K/output.{wav,flac}

Resume-safe at direction granularity: a direction is done only when its
`.done` marker exists, written strictly after `run()` reports a complete
send (every input chunk delivered). A crash / kill / exhausted reconnect
budget leaves partial audio WITHOUT the marker, so rerunning the driver
redoes exactly the unfinished directions and skips the rest. Rerun the same
command after any interruption; `--limit 1` first for a pilot.

    # pilot: one conversation of the test split
    uv run --extra eval --with google-genai --with python-dotenv --with scipy \
        python baselines/gemini/pipeline/run_split.py --split test --limit 1

    # full run, 4 concurrent Live sessions
    uv run --extra eval --with google-genai --with python-dotenv --with scipy \
        python baselines/gemini/pipeline/run_split.py --split test --parallel 4
"""
from __future__ import annotations

import argparse
import asyncio
import io
import logging
import sys
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import snapshot_download

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE))

from dotenv import load_dotenv  # noqa: E402

from direct_gemini_client import DEFAULT_SYSTEM_PROMPT, run  # noqa: E402

DEV_DATASET = "mundo-ai/turn-benchmark-dev"
TEST_DATASET = "mundo-ai/turn-benchmark-test"  # public: audio yes, labels no
_DEFAULT_OUT = _HERE.parent / "sample_runs"


def _split_ids(split: str) -> list[str]:
    split_file = _REPO / "turnbench" / "splits" / f"{split}.txt"
    return [ln.strip() for ln in split_file.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]


# ---- lazy parquet access (mirrors baselines/openai_realtime.py) --------------

def _shard_files(source: str) -> list[str]:
    """Parquet shards for a split — a local directory, or an HF dataset snapshot."""
    if Path(source).is_dir():
        return sorted(str(p) for p in Path(source).glob("*.parquet"))
    from turnbench.data import PINNED_REVISIONS
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
        assert parquet.metadata.num_rows == parquet.metadata.num_row_groups, (
            f"{shard}: expected one conversation per row group"
        )
        ids = parquet.read(columns=["conversation_id"])["conversation_id"].to_pylist()
        for row_group, cid in enumerate(ids):
            index[cid] = (shard, row_group)
    return index


def _materialize_input(shard: str, row_group: int, speaker: int, path: Path,
                       max_seconds: float | None = None) -> Path:
    """Write speaker_K's channel from one row group to a wav the client reads.
    Idempotent; one row group in RAM at a time. `max_seconds` truncates the
    input (pilot runs only — streaming is real-time, so audio length = wall
    clock); never use it for a real recording run."""
    if path.exists():
        return path
    table = pq.ParquetFile(shard).read_row_group(
        row_group, columns=[f"speaker_{speaker}_audio"]
    )
    cell = table[f"speaker_{speaker}_audio"][0].as_py()
    data = cell["bytes"] if isinstance(cell, dict) else cell
    samples, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if max_seconds is not None:
        samples = samples[: int(max_seconds * sample_rate)]
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")
    return path


# ---- batch driver -------------------------------------------------------------

_CONNECT_GATE = asyncio.Lock()  # stagger session opens — a thundering herd of
_CONNECT_SPACING_S = 0.25       # TLS handshakes times every one of them out
_MAX_ATTEMPTS = 5


async def _record_direction(
    tid: str, speaker: int, shard: str, row_group: int, out_root: Path,
    work_dir: Path, model: str, voice: str, system_prompt: str,
    sem: asyncio.Semaphore, log: logging.Logger,
    max_seconds: float | None = None,
) -> tuple[str, int, bool]:
    """One conversation × direction. Returns (task_id, speaker, ok)."""
    out_dir = out_root / tid / f"speaker_{speaker}"
    done = out_dir / ".done"
    if done.exists():
        log.info("skip %s/speaker_%d (done)", tid, speaker)
        return tid, speaker, True

    async with sem:
        # decode off the event loop — blocking IO would stall other sessions
        input_wav = await asyncio.to_thread(
            _materialize_input, shard, row_group, speaker,
            work_dir / f"{tid}_speaker_{speaker}.wav", max_seconds,
        )
        task_log = logging.getLogger(f"gemini.{tid}.s{speaker}")
        handler = logging.FileHandler(work_dir / f"{tid}_speaker_{speaker}.log")
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        task_log.addHandler(handler)
        task_log.propagate = False
        complete = False
        try:
            for attempt in range(1, _MAX_ATTEMPTS + 1):
                async with _CONNECT_GATE:  # stagger connection opens
                    await asyncio.sleep(_CONNECT_SPACING_S)
                try:
                    complete = await run(input_wav, out_dir / "output.wav", model,
                                         system_prompt, voice, task_log,
                                         normalize_peak_dbfs=-3.0)
                except Exception as e:  # handshake timeout, quota, transport …
                    backoff = min(60.0, 2.0 ** attempt)
                    log.warning("retry %s/speaker_%d (attempt %d/%d in %.0fs): %s",
                                tid, speaker, attempt, _MAX_ATTEMPTS, backoff, e)
                    await asyncio.sleep(backoff)
                    continue
                break  # run() returned (complete or partial) — don't re-run a partial
        finally:
            handler.close()
            task_log.removeHandler(handler)
            input_wav.unlink(missing_ok=True)  # ~100 MB per direction; recreated on retry
    if complete:
        done.touch()  # marker strictly after a complete send
        log.info("done %s/speaker_%d", tid, speaker)
    else:
        log.error("PARTIAL %s/speaker_%d — will retry on next invocation", tid, speaker)
    return tid, speaker, complete


async def _main_async(args: argparse.Namespace, log: logging.Logger) -> int:
    source = args.dataset or (TEST_DATASET if args.split == "test" else DEV_DATASET)
    index = dataset_index(source)
    wanted = [t for t in _split_ids(args.split) if t in index]
    missing = sorted(set(_split_ids(args.split)) - set(index), key=int)
    if missing:
        sys.exit(f"{len(missing)} {args.split} conversations missing from {source}: {missing}")
    if args.limit:
        wanted = wanted[: args.limit]

    system_prompt = (args.system_prompt_file.read_text().strip()
                     if args.system_prompt_file.exists() else DEFAULT_SYSTEM_PROMPT)
    work_dir = args.out / ".batch"
    work_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.parallel)
    jobs = [
        _record_direction(tid, k, *index[tid], args.out, work_dir,
                          args.model, args.voice, system_prompt, sem, log,
                          args.max_seconds)
        for tid in wanted for k in (1, 2)
    ]
    log.info("split=%s conversations=%d directions=%d parallel=%d out=%s",
             args.split, len(wanted), len(jobs), args.parallel, args.out)
    results = await asyncio.gather(*jobs)
    failed = [(t, k) for t, k, ok in results if not ok]
    log.info("finished: %d/%d directions complete", len(results) - len(failed), len(results))
    if failed:
        log.error("incomplete (rerun the same command to retry): %s",
                  [f"{t}/s{k}" for t, k in failed])
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("dev", "test"), default="test")
    ap.add_argument("--dataset", default=None,
                    help="HF repo or local parquet dir (default: the split's public repo)")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                    help=f"recording root (default: {_DEFAULT_OUT})")
    ap.add_argument("--parallel", type=int, default=4,
                    help="concurrent Gemini Live sessions (default: 4)")
    ap.add_argument("--limit", type=int, default=None,
                    help="record only the first N conversations (pilot)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="truncate each input to this many seconds (PILOT ONLY — "
                         "never for a real recording run)")
    ap.add_argument("--model", default="gemini-3.1-flash-live-preview")
    ap.add_argument("--voice", default="Puck")
    ap.add_argument("--system-prompt-file", type=Path,
                    default=_HERE / "system_prompt.txt")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("gemini.batch")
    load_dotenv(_REPO / ".env", override=False)
    return asyncio.run(_main_async(args, log))


if __name__ == "__main__":
    sys.exit(main())
