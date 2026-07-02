#!/usr/bin/env python3
"""Gemini VAD-based turn-taking predictor.

Runs pyannote VAD on Gemini Live's recorded output audio and on the user's
input audio, then reads events off the region boundaries. No ASR pass; the
paired `baselines/gemini` baseline reads a word-level ASR transcript instead
and so misses non-lexical vocalisations.

Events, per direction K ∈ {1, 2}:

    eot_speaker_K          = agent VAD onsets in direction K while
                             user_K is VAD-inactive at that time.
                             Committed at the onset (no lookahead).

EOT only — this baseline commits empty interruption lists. The natural
VAD-side interruption readout (agent VAD *offset* while the user is active
nearby, i.e. the agent yielding the floor) measures the agent's yield
decision, not the barge-in onset: its commit times are offset-anchored, so
its latencies aren't comparable with the onset-anchored INT convention every
other baseline uses (user speech onset; cf. baselines/openai_realtime.py),
and it also fires at ordinary turn exchanges (agent stops, user replies
within the window). The INT track is therefore out of scope here; the ASR
readout in baselines/gemini remains the INT source for Gemini.

Runs per-file pyannote VAD once and caches the resulting regions to
`--cache-dir` (default: `baselines/gemini/.vad_cache`). The cache key is
the file only — delete the cache dir after changing VAD_PARAMS or
MERGE_GAP_S, or stale regions will be served.

    python -m baselines.gemini_vad.predict                    # score dev in-place
    python -m baselines.gemini_vad.predict \
        --out baselines/gemini_vad/predictions-dev.json       # write JSON
    python -m baselines.gemini_vad.predict \
        --dataset mundo-ai/turn-benchmark-test \
        --out baselines/gemini_vad/predictions-test.json      # test split
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# eval/ + baselines/ are two levels up (baselines/gemini_vad/predict.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import io  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

from eval.data import DEV_DATASET, PINNED_REVISIONS  # noqa: E402
from eval.submission import (  # noqa: E402
    SCHEMA_VERSION,
    ConversationPrediction,
    SpeakerEvents,
    Submission,
)

FRAME_RATE_HZ = 12.5
MERGE_GAP_S = 0.5
VAD_PARAMS = {
    "onset": 0.5,
    "offset": 0.363,
    "min_duration_on": 0.0,
    "min_duration_off": 0.0,
}

_HERE = Path(__file__).resolve().parent
_DEFAULT_SAMPLE_RUNS = _HERE.parent / "gemini" / "sample_runs"
_DEFAULT_CACHE = _HERE.parent / "gemini" / ".vad_cache"


def _merge_regions(
    regions: list[tuple[float, float]], max_gap_s: float = MERGE_GAP_S
) -> list[tuple[float, float]]:
    if not regions:
        return []
    regions = sorted(regions)
    merged = [list(regions[0])]
    for start, end in regions[1:]:
        if start - merged[-1][1] < max_gap_s:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _active_at(regions: list[tuple[float, float]], t: float) -> bool:
    return any(s <= t < e for s, e in regions)


class VadCache:
    """Reads pyannote-VAD regions from disk; runs pyannote lazily on cache miss."""

    def __init__(self, cache_dir: Path, device: str | None = None) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device  # None = auto (cuda > mps > cpu)
        self._vad = None

    def _pyannote(self):
        if self._vad is None:
            self._vad = _build_pyannote_vad(self.device)
        return self._vad

    def regions_for(self, wav_path: Path, key: str) -> list[tuple[float, float]]:
        cf = self.cache_dir / f"{key}.json"
        if cf.exists():
            return [(s, e) for s, e in json.loads(cf.read_text())]
        annotation = self._pyannote()({"audio": str(wav_path)})
        raw = [(seg.start, seg.end) for seg in annotation.itersegments()]
        merged = _merge_regions(raw, MERGE_GAP_S)
        cf.write_text(json.dumps(merged))
        return merged


def _build_pyannote_vad(device: str | None = None):
    """Lazy: only imported when a cache miss forces us to run pyannote.
    `device` pins the pipeline (e.g. "cuda:3" for multi-GPU worker sharding);
    None picks the best available."""
    import torch
    import huggingface_hub as hh

    orig_hub_download = hh.hf_hub_download
    orig_snap_download = hh.snapshot_download

    def _shim(fn):
        def _wrapped(*args, **kwargs):
            if "use_auth_token" in kwargs and "token" not in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            elif "use_auth_token" in kwargs:
                kwargs.pop("use_auth_token")
            return fn(*args, **kwargs)

        return _wrapped

    hh.hf_hub_download = _shim(orig_hub_download)
    hh.snapshot_download = _shim(orig_snap_download)
    import pyannote.audio.core.io as _pa_io
    import pyannote.audio.pipelines.utils.hook as _pa_hook

    for mod in (_pa_io, _pa_hook):
        if hasattr(mod, "hf_hub_download"):
            mod.hf_hub_download = hh.hf_hub_download

    orig_torch_load = torch.load

    def _torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return orig_torch_load(*args, **kwargs)

    torch.load = _torch_load

    from pyannote.audio.pipelines import VoiceActivityDetection

    vad = VoiceActivityDetection(segmentation="pyannote/segmentation")
    vad.instantiate(VAD_PARAMS)
    # run on an accelerator when present — CPU is ~2h for the test split's
    # ~92h of audio, CUDA minutes. Regions are identical across devices.
    if device is not None:
        vad.to(torch.device(device))
    elif torch.cuda.is_available():
        vad.to(torch.device("cuda"))
    elif torch.backends.mps.is_available():
        vad.to(torch.device("mps"))
    return vad


# ---- parallel cache warming ---------------------------------------------------
# The 464 VAD passes (2 user channels + 2 agent outputs × 116 conversations) are
# embarrassingly parallel: each worker process builds its own pyannote pipeline
# and fills the shared region cache (idempotent, keyed per file), then the main
# predict loop runs entirely on cache hits. With multiple GPUs, workers are
# assigned devices round-robin (cuda:0, cuda:1, …), so e.g. --vad-workers 16 on
# an 8-GPU node puts two pipelines on each GPU.

_WORKER_CACHE: VadCache | None = None
_WORKER_DEVICE: str | None = None


def _init_worker(device_queue) -> None:
    """Pool initializer: claim this worker's device assignment (or None = auto)."""
    global _WORKER_DEVICE
    try:
        _WORKER_DEVICE = device_queue.get_nowait()
    except Exception:
        _WORKER_DEVICE = None


def _warm_one(job: tuple[str, str, str]) -> str:
    """Worker: ensure one file's VAD regions are cached. job=(wav, key, cache_dir)."""
    global _WORKER_CACHE
    wav, key, cache_dir = job
    if _WORKER_CACHE is None:
        _WORKER_CACHE = VadCache(Path(cache_dir), device=_WORKER_DEVICE)
    _WORKER_CACHE.regions_for(Path(wav), key)
    return key


def warm_cache(index: dict[str, tuple[str, int]], sample_runs: Path,
               cache_dir: Path, workers: int) -> None:
    """Materialise user wavs, then VAD every uncached file across `workers`
    processes. Safe to interrupt/rerun — the cache is per-file and idempotent."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import torch

    jobs: list[tuple[str, str, str]] = []
    for task_id in sorted(index, key=int):
        shard, row_group = index[task_id]
        for speaker in (1, 2):
            user_wav, _ = _materialize_user_wav(shard, row_group, task_id, speaker)
            agent_wav = sample_runs / task_id / f"speaker_{speaker}" / "output.flac"
            for wav, key in ((user_wav, f"{task_id}_user{speaker}"),
                             (agent_wav, f"{task_id}_agent{speaker}")):
                if not (cache_dir / f"{key}.json").exists():
                    jobs.append((str(wav), key, str(cache_dir)))

    n_gpus = torch.cuda.device_count()
    ctx = multiprocessing.get_context("spawn")  # CUDA-safe; also macOS default
    device_queue = ctx.Queue()
    for i in range(workers):
        device_queue.put(f"cuda:{i % n_gpus}" if n_gpus else None)
    print(f"warming VAD cache: {len(jobs)} uncached files, {workers} workers, "
          f"{n_gpus} GPU(s)", file=sys.stderr)
    if not jobs:
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                             initializer=_init_worker, initargs=(device_queue,)) as pool:
        for i, key in enumerate(pool.map(_warm_one, jobs, chunksize=1), 1):
            print(f"vad [{i}/{len(jobs)}] {key}", file=sys.stderr)


def _quantize(times_s: set[float], duration_s: float) -> list[float]:
    """Snap to the 12.5 Hz grid, drop anything outside [0, duration_s), sort."""
    n_frames = int(duration_s * FRAME_RATE_HZ)
    frames = sorted(
        {int(t * FRAME_RATE_HZ) for t in times_s if 0.0 <= int(t * FRAME_RATE_HZ) < n_frames}
    )
    return [round(f / FRAME_RATE_HZ, 3) for f in frames]


# ---- lazy parquet access (mirrors baselines/openai_realtime.py) --------------
# Materialising a whole split with audio needs ~24 GB for test and gets the
# process OOM-killed; read one conversation's row group at a time instead.

def _shard_files(source: str) -> list[str]:
    """Parquet shards for a split — a local directory, or an HF dataset snapshot."""
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
        assert parquet.metadata.num_rows == parquet.metadata.num_row_groups, (
            f"{shard}: expected one conversation per row group"
        )
        ids = parquet.read(columns=["conversation_id"])["conversation_id"].to_pylist()
        for row_group, cid in enumerate(ids):
            index[cid] = (shard, row_group)
    return index


_TEMP_WAV_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "gemini_vad_user_wavs"


def _materialize_user_wav(shard: str, row_group: int, task_id: str, speaker: int) -> tuple[Path, float]:
    """Write speaker_K's channel from one row group to a temp wav (pyannote wants
    a file path) and return (path, duration_s). Idempotent per (task, speaker)."""
    import soundfile as sf

    _TEMP_WAV_DIR.mkdir(parents=True, exist_ok=True)
    path = _TEMP_WAV_DIR / f"{task_id}_speaker_{speaker}.wav"
    if path.exists():
        return path, sf.info(str(path)).duration
    table = pq.ParquetFile(shard).read_row_group(
        row_group, columns=[f"speaker_{speaker}_audio"]
    )
    cell = table[f"speaker_{speaker}_audio"][0].as_py()
    data = cell["bytes"] if isinstance(cell, dict) else cell
    samples, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")
    return path, len(samples) / sample_rate


def predict_conversation(
    task_id: str,
    shard: str,
    row_group: int,
    sample_runs: Path,
    cache: VadCache,
) -> ConversationPrediction:
    """One conversation → ConversationPrediction, EOT for both speakers."""
    user1_wav, dur_s = _materialize_user_wav(shard, row_group, task_id, 1)
    user2_wav, _ = _materialize_user_wav(shard, row_group, task_id, 2)

    agent1_wav = sample_runs / task_id / "speaker_1" / "output.flac"
    agent2_wav = sample_runs / task_id / "speaker_2" / "output.flac"
    for path in (agent1_wav, agent2_wav):
        if not path.exists():
            sys.exit(
                f"missing Gemini output audio: {path}\n"
                "run the inference stage first (see README.md)."
            )

    user1 = cache.regions_for(user1_wav, f"{task_id}_user1")
    user2 = cache.regions_for(user2_wav, f"{task_id}_user2")
    agent1 = cache.regions_for(agent1_wav, f"{task_id}_agent1")
    agent2 = cache.regions_for(agent2_wav, f"{task_id}_agent2")

    # temp user wavs are only needed for pyannote; drop them once regions exist
    user1_wav.unlink(missing_ok=True)
    user2_wav.unlink(missing_ok=True)

    eot: dict[int, set[float]] = {1: set(), 2: set()}

    # EOT[K]: agent onset in direction K while user_K is VAD-inactive there.
    # Committed at the onset — the detector only needs audio up to `onset`.
    for k, agent_r, user_r in ((1, agent1, user1), (2, agent2, user2)):
        for onset_s, _ in agent_r:
            if not _active_at(user_r, onset_s):
                eot[k].add(onset_s)

    # INT is out of scope for this readout (see module docstring): the VAD-side
    # signal is the agent's *yield* (offset-anchored), not the barge-in onset,
    # so it isn't comparable with the other baselines' INT convention.
    return ConversationPrediction(
        conversation_id=task_id,
        speaker_1=SpeakerEvents(eot=_quantize(eot[1], dur_s), interruption=[]),
        speaker_2=SpeakerEvents(eot=_quantize(eot[2], dur_s), interruption=[]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=DEV_DATASET,
        help="HF dataset repo id, or a local directory of parquet shards",
    )
    parser.add_argument(
        "--sample-runs",
        type=Path,
        default=_DEFAULT_SAMPLE_RUNS,
        help="Gemini Live output audio root (see baselines/gemini/pipeline).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_DEFAULT_CACHE,
        help="Per-file pyannote-VAD region cache.",
    )
    parser.add_argument(
        "--out", default=None, help="Write a predictions JSON here instead of scoring."
    )
    parser.add_argument(
        "--vad-workers", type=int, default=1,
        help="parallel VAD worker processes for cache warming (CPU nodes; "
             "the 464 files are independent — one worker per core scales)",
    )
    args = parser.parse_args()

    index = dataset_index(args.dataset)
    cache = VadCache(args.cache_dir)
    if args.vad_workers > 1:
        warm_cache(index, args.sample_runs, args.cache_dir, args.vad_workers)

    predictions = []
    for i, task_id in enumerate(sorted(index, key=int), 1):
        predictions.append(
            predict_conversation(task_id, *index[task_id], args.sample_runs, cache)
        )
        print(f"[{i}/{len(index)}] {task_id}", file=sys.stderr)
    submission = Submission(schema_version=SCHEMA_VERSION, predictions=predictions)

    if args.out is not None:
        Path(args.out).write_text(submission.model_dump_json(indent=2), encoding="utf-8")
        print(
            f"Wrote {len(submission.predictions)} predictions to {args.out}",
            file=sys.stderr,
        )
        return 0

    # scoring needs gold labels — score the written JSON with eval.score instead
    from eval.data import resolve_dataset  # noqa: E402  (dev-sized, labels only)
    from eval.score import score_submission, task_cells  # noqa: E402

    scores = score_submission(submission, resolve_dataset(source=args.dataset, skip_audio=True))
    print(f"gemini_vad — {len(submission.predictions)} conversations")
    for task_name, score in (("EOT", scores.task_eot), ("INT", scores.task_int)):
        recall, fp_rate, latency = task_cells(score)
        print(f"  {task_name}: recall={recall} fp_rate={fp_rate} latency_ms={latency}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
