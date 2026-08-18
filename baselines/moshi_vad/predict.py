#!/usr/bin/env python3
"""Moshi VAD-based turn-taking predictor.

Runs pyannote VAD on Moshi's recorded output audio and on the user's input
audio, then reads events off the region boundaries. Same readout strategy as
`baselines/gemini_vad` (which introduced it for Gemini Live); the paired
`baselines/moshi` baseline reads a word-level ASR transcript instead and so
misses non-lexical vocalisations — a large share of Moshi's floor-holding
(hums, backchannels, laughs).

Events, per direction K ∈ {1, 2} (Moshi conversing with dataset speaker K):

    eot_speaker_K          = agent VAD onsets in direction K while user_K is
                             VAD-inactive at that time. Committed at the onset
                             (no lookahead in the readout rules; pyannote
                             itself is ~2 s non-causal, as in gemini_vad).

EOT only — the interruption lists are committed empty, as in gemini_vad. A
"user onset inside an agent VAD region" readout is onset-anchored and was
previously committed here, but for a generative model it measures passive
floor-overlap, not interruption *detection*: the model contributes only
"was I speaking there". The detection signal a full-duplex model could
exhibit — yielding after a barge-in — is measurable and Moshi shows none:
across 1,927 test barge-ins its time-to-stop after a user onset (median
1.38 s, 37% within 1 s) is indistinguishable from the counterfactual
time-to-stop at a random moment in the same speech region (1.44 s, 39%).
Scoring that yield is also outside the benchmark's causal readout rules
(offset-anchored timestamps, or lookahead at commit time), so the INT track
is out of scope for VAD readouts of generative models.

Operating point (PARAMS below) swept on dev per the repo's protocol —
highest recall subject to fp_rate ≤ 0.1. Moshi holds the floor only
~4–8% of frames, so unlike Gemini the stock pyannote thresholds sit far from
its optimum: the best readout fragments its brief, quiet floor-taking into
many crisp onsets (high onset, narrow hysteresis, small merge gap).

The cache stores the segmentation model's per-frame *scores* (one model pass
per file); thresholds and merge gaps are applied at read time, so changing
PARAMS never serves stale regions and needs no recompute.

    python -m baselines.moshi_vad.predict --sample-runs <moshi_out>/dev
    python -m baselines.moshi_vad.predict --sample-runs <moshi_out>/dev \
        --out baselines/moshi_vad/predictions-dev.json
    python -m baselines.moshi_vad.predict \
        --dataset mundo-ai/turn-benchmark-test --sample-runs <moshi_out>/test \
        --out baselines/moshi_vad/predictions-test.json
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

# eval/ + baselines/ are two levels up (baselines/moshi_vad/predict.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

from turnbench.data import DEV_DATASET, PINNED_REVISIONS  # noqa: E402
from turnbench.submission import (  # noqa: E402
    SCHEMA_VERSION,
    ConversationPrediction,
    SpeakerEvents,
    Submission,
)

FRAME_RATE_HZ = 12.5

# Operating point, swept on dev (recall at fp_rate <= 0.1):
#   EOT 0.212/0.066 on dev at these settings.
PARAMS = {
    "eot": {"onset": 0.88, "offset": 0.862, "merge_gap_s": 0.15},
}

_HERE = Path(__file__).resolve().parent
_DEFAULT_CACHE = _HERE / ".vad_cache"


def _merge_regions(
    regions: list[tuple[float, float]], max_gap_s: float
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


def _build_segmentation_inference(device: str | None):
    """pyannote/segmentation scorer (max over speaker dims -> speech score).
    Lazy import; shims for huggingface_hub/torch API drift as in gemini_vad."""
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

    from pyannote.audio import Inference
    from pyannote.audio.pipelines.utils import get_model

    model = get_model("pyannote/segmentation")
    if device is not None:
        model.to(torch.device(device))
    return Inference(
        model, pre_aggregation_hook=lambda s: np.max(s, axis=-1, keepdims=True)
    )


class VadCache:
    """Per-file segmentation *scores* on disk (fp16 npz); regions are derived
    at read time by hysteresis-thresholding the scores with a task's PARAMS.
    One model pass per file, any thresholds for free."""

    def __init__(self, cache_dir: Path, device: str | None = None) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device  # None = pyannote picks (cuda > mps > cpu)
        self._inference = None

    def _scores(self, wav_path: Path, key: str):
        from pyannote.core import SlidingWindow, SlidingWindowFeature

        cf = self.cache_dir / f"{key}.npz"
        if not cf.exists():
            if self._inference is None:
                self._inference = _build_segmentation_inference(self.device)
            scores = self._inference({"audio": str(wav_path)})
            sw = scores.sliding_window
            np.savez_compressed(
                cf, data=scores.data.astype(np.float16).squeeze(-1),
                start=sw.start, duration=sw.duration, step=sw.step,
            )
        z = np.load(cf)
        return SlidingWindowFeature(
            z["data"].astype(np.float32)[:, None],
            SlidingWindow(start=float(z["start"]), duration=float(z["duration"]),
                          step=float(z["step"])),
        )

    def regions_for(self, wav_path: Path, key: str, params: dict) -> list[tuple[float, float]]:
        from pyannote.audio.utils.signal import Binarize

        binarize = Binarize(onset=params["onset"], offset=params["offset"],
                            min_duration_on=0.0, min_duration_off=0.0)
        annotation = binarize(self._scores(wav_path, key))
        raw = [(seg.start, seg.end) for seg in annotation.itersegments()]
        return _merge_regions(raw, params["merge_gap_s"])


# ---- parallel cache warming (one scores pass per file) ------------------------

_WORKER_CACHE: VadCache | None = None
_WORKER_DEVICE: str | None = None


def _init_worker(device_queue) -> None:
    global _WORKER_DEVICE
    try:
        _WORKER_DEVICE = device_queue.get(timeout=60)
    except Exception:
        _WORKER_DEVICE = None


def _warm_one(job: tuple[str, str, str]) -> str:
    global _WORKER_CACHE
    wav, key, cache_dir = job
    if _WORKER_CACHE is None:
        _WORKER_CACHE = VadCache(Path(cache_dir), device=_WORKER_DEVICE)
    _WORKER_CACHE._scores(Path(wav), key)
    return key


def warm_cache(index: dict[str, tuple[str, int]], sample_runs: Path,
               cache_dir: Path, workers: int) -> None:
    """Materialise user wavs, then score every uncached file across `workers`
    processes (round-robin over GPUs when present). Idempotent."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import torch

    jobs: list[tuple[str, str, str]] = []
    for task_id in sorted(index, key=int):
        shard, row_group = index[task_id]
        for speaker in (1, 2):
            user_wav, _ = _materialize_user_wav(shard, row_group, task_id, speaker)
            agent_wav = sample_runs / task_id / f"speaker_{speaker}" / "output.flac"
            if not agent_wav.exists():
                sys.exit(
                    f"missing Moshi output audio: {agent_wav}\n"
                    "run the inference stage first (see README.md)."
                )
            for wav, key in ((user_wav, f"{task_id}_user{speaker}"),
                             (agent_wav, f"{task_id}_agent{speaker}")):
                if not (cache_dir / f"{key}.npz").exists():
                    jobs.append((str(wav), key, str(cache_dir)))

    n_gpus = torch.cuda.device_count()
    ctx = multiprocessing.get_context("spawn")
    device_queue = ctx.Queue()
    for i in range(workers):
        device_queue.put(f"cuda:{i % n_gpus}" if n_gpus else "cpu")
    print(f"warming VAD scores: {len(jobs)} uncached files, {workers} workers, "
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


# ---- lazy parquet access (mirrors baselines/gemini_vad/predict.py) ------------

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
        assert parquet.metadata.num_rows == parquet.metadata.num_row_groups, (
            f"{shard}: expected one conversation per row group"
        )
        ids = parquet.read(columns=["conversation_id"])["conversation_id"].to_pylist()
        for row_group, cid in enumerate(ids):
            index[cid] = (shard, row_group)
    return index


_TEMP_WAV_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "moshi_vad_user_wavs"


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
    """One conversation → ConversationPrediction, EOT + INT for both speakers."""
    user1_wav, dur_s = _materialize_user_wav(shard, row_group, task_id, 1)
    user2_wav, _ = _materialize_user_wav(shard, row_group, task_id, 2)
    user_wavs = {1: user1_wav, 2: user2_wav}

    agent_wavs = {}
    for k in (1, 2):
        agent_wavs[k] = sample_runs / task_id / f"speaker_{k}" / "output.flac"
        if not agent_wavs[k].exists():
            sys.exit(
                f"missing Moshi output audio: {agent_wavs[k]}\n"
                "run the inference stage first (see README.md)."
            )

    eot: dict[int, set[float]] = {1: set(), 2: set()}
    for k in (1, 2):
        # EOT[K]: agent onset while user_K is VAD-inactive there.
        agent_r = cache.regions_for(agent_wavs[k], f"{task_id}_agent{k}", PARAMS["eot"])
        user_r = cache.regions_for(user_wavs[k], f"{task_id}_user{k}", PARAMS["eot"])
        eot[k].update(on for on, _ in agent_r if not _active_at(user_r, on))

    # temp user wavs are only needed for pyannote; drop them once scores exist
    user1_wav.unlink(missing_ok=True)
    user2_wav.unlink(missing_ok=True)

    # INT is out of scope for this readout (see module docstring): committed empty.
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
        required=True,
        help="Moshi output audio root (see baselines/moshi/pipeline/run_fleet.py).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_DEFAULT_CACHE,
        help="Per-file segmentation-score cache (thresholds applied at read time).",
    )
    parser.add_argument(
        "--out", default=None, help="Write a predictions JSON here instead of scoring."
    )
    parser.add_argument(
        "--vad-workers", type=int, default=1,
        help="parallel scoring worker processes for cache warming "
             "(round-robin over GPUs when present)",
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

    from turnbench.data import resolve_dataset  # noqa: E402  (dev-sized, labels only)
    from turnbench.score import score_submission, task_cells  # noqa: E402

    scores = score_submission(submission, resolve_dataset(source=args.dataset, skip_audio=True))
    print(f"moshi_vad — {len(submission.predictions)} conversations")
    for task_name, score in (("EOT", scores.task_eot), ("INT", scores.task_int)):
        recall, fp_rate, latency = task_cells(score)
        print(f"  {task_name}: recall={recall} fp_rate={fp_rate} latency_ms={latency}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
