"""Resolve the benchmark dataset (HuggingFace parquet) to in-memory conversations.

The scorer takes a predictions JSON and a dataset. By default the dataset is the
public dev set on HuggingFace: parquet shards, one row per conversation, each row
carrying the two per-speaker audio channels and the raw three-annotator tracks
per speaker. `--dataset` accepts any HF dataset repo id, or a local directory of
parquet shards. The private test set is just a private HF repo, scored by the
same code server-side. Authentication uses ambient HF credentials (HF_TOKEN /
`huggingface-cli login`, or an HF_TOKEN in a repo-root `.env`); the public sets
are gated, so callers must have been granted access to the repo.

Shards are read with pyarrow and wrapped via the HFDataset constructor, not
`datasets.load_dataset` — see resolve_dataset for why (load_dataset overflows on
the full test split).
"""

import hashlib
import io
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from datasets import Audio, Dataset as HFDataset
from huggingface_hub import HfApi, HfFileSystem, snapshot_download

from eval.durations import load_durations_for_source

DEV_DATASET = "mundo-ai/turn-benchmark-dev"
DEV_REVISION = "8fa18a24be51528a45397b35cbcaecd84202062b"
GOLD_DATASET = "mundo-ai/turn-benchmark-test-golden"
GOLD_REVISION = "7b34876db0a883fdba5d1b6d67e9c3bf3303569a"

# The public splits that together reconstruct the full annotated corpus
# (154 dialogues): dev plus the private golden test set.
FULL_CORPUS_SOURCES = (DEV_DATASET, GOLD_DATASET)

# Pinned revisions for reproducibility; sources not listed float to latest.
# Pinning also lets the projected gold-column read cache locally (_read_gold_columns),
# so scoring a pinned source is a one-time fetch then instant.
PINNED_REVISIONS = {DEV_DATASET: DEV_REVISION, GOLD_DATASET: GOLD_REVISION}

ANNOTATORS = ("a", "b", "c")
SPEAKERS = (1, 2)

# Scoring, sweeping and gold export need only these columns. Audio columns are
# added only for callers that don't pass skip_audio (inference). Projecting the
# rest away is what lets the skip_audio path skip ~99.96% of every shard.
GOLD_COLUMNS = ["conversation_id"] + [
    f"speaker_{speaker}_annotation_{annotator}"
    for speaker in SPEAKERS
    for annotator in ANNOTATORS
]
AUDIO_COLUMNS = [f"speaker_{speaker}_audio" for speaker in SPEAKERS]

_GOLD_CACHE_DIR = Path.home() / ".cache" / "tt-benchmark" / "gold"

# One annotated segment as it lives in the parquet annotation columns:
# (start_s, end_s, fine_label, text). The verbatim annotator label is mapped to
# the canonical taxonomy later (eval/gold.py); the transcript text is used by
# the corpus statistics (data_analysis/per_conversation.py).
Annotation = tuple[float, float, str, str]


@dataclass(frozen=True)
class Conversation:
    """One conversation's gold inputs.

    `annotations` maps (speaker, annotator) to that track's segments in file
    order, before any canonical label mapping. `audio_bytes` holds each speaker
    channel's encoded bytes (WAV or FLAC); `audio()` decodes on demand so a loop
    over conversations keeps at most one conversation's audio in memory.
    """

    conversation_id: str
    duration_s: float
    annotations: dict[tuple[int, str], list[Annotation]]
    audio_bytes: dict[int, bytes]

    def audio(self, speaker: int) -> tuple[np.ndarray, int]:
        """Decode one speaker channel to (mono float32 samples, sample_rate)."""
        samples, sample_rate = sf.read(
            io.BytesIO(self.audio_bytes[speaker]), dtype="float32", always_2d=False
        )
        return samples, sample_rate


@dataclass(frozen=True)
class Dataset:
    """A loaded benchmark split: the HF rows, a conversation_id -> row index, and
    per-conversation durations from the committed durations artifact (empty for
    sources without one, in which case duration falls back to the audio header)."""

    rows: HFDataset
    index: dict[str, int]
    durations: dict[str, float]


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _seed_hf_token_from_env() -> None:
    """Seed HF_TOKEN from a repo-root `.env` so scoring the gated splits needs no
    manual export. A token already in the environment wins; no-ops without `.env`."""
    env_path = _REPO_ROOT / ".env"
    if os.environ.get("HF_TOKEN") or not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key.strip() == "HF_TOKEN":
            os.environ["HF_TOKEN"] = value.strip().strip("\"'")
            return


def resolve_dataset(
    source: str = DEV_DATASET, revision: str | None = None, *, skip_audio: bool = False
) -> Dataset:
    """Load one benchmark split (Arrow-backed) and index it by conversation id.

    Audio is loaded by default, so any predictor can call resolve_dataset(...) and
    read conv.audio(...) unchanged. Pass skip_audio=True for scoring, sweeping and
    gold export: only the gold columns (conversation_id + the three annotator
    tracks per speaker) are read, and remote sources are read column-projected over
    HTTP range requests — never downloading the audio, which is ~99.96% of each
    shard. The tiny projected result is cached locally per (source, revision).

    `source` is an HF dataset repo id or a local directory of parquet shards.
    `revision` pins the HF download; when omitted it defaults to the pinned dev
    revision for the dev dataset, else the repo's latest. Authentication uses
    ambient HF credentials.

    Shards are wrapped via the HFDataset constructor rather than
    `datasets.load_dataset`: load_dataset's writer calls `combine_chunks()`, and
    the embedded audio is a `binary` column (32-bit offsets, ~2 GB per array) that
    overflows on the full test split. The constructor wraps the chunked table
    as-is — chunks are never combined — and scores identically to load_dataset.
    """
    _seed_hf_token_from_env()
    # skip_audio only avoids the remote *download*; local shards have nothing to
    # download and their header is the duration fallback, so always read them whole.
    is_local = Path(source).is_dir()
    load_audio = is_local or not skip_audio
    columns = GOLD_COLUMNS + AUDIO_COLUMNS if load_audio else GOLD_COLUMNS
    if not is_local and revision is None:
        revision = PINNED_REVISIONS.get(source)
    if skip_audio and not is_local:
        table = _read_gold_columns(source, revision)
    else:
        if is_local:
            files = [str(path) for path in Path(source).glob("*.parquet")]
        else:
            snapshot = snapshot_download(
                source, repo_type="dataset", revision=revision, allow_patterns="*.parquet"
            )
            files = [str(path) for path in Path(snapshot).rglob("*.parquet")]
        table = pa.concat_tables([pq.read_table(file, columns=columns) for file in sorted(files)])
    rows = HFDataset(table)
    if load_audio:
        for speaker in SPEAKERS:
            rows = rows.cast_column(f"speaker_{speaker}_audio", Audio(decode=False))
    index = {row_id: i for i, row_id in enumerate(rows["conversation_id"])}
    return Dataset(rows=rows, index=index, durations=load_durations_for_source(source))


def _read_gold_columns(source: str, revision: str | None) -> pa.Table:
    return read_columns_projected(source, revision, GOLD_COLUMNS)


def read_columns_projected(source: str, revision: str | None, columns: list[str]) -> pa.Table:
    """Read `columns` from every shard of a remote dataset, projected over HTTP
    range requests (`pre_buffer` coalesces the per-column-chunk reads) — never
    downloading the rest of the shard (the audio is ~99.96% of it). Shards are
    read in parallel, and the tiny result is cached locally per
    (source, revision, columns) so repeat runs are instant. When `revision` is
    None the source's pinned revision is used, falling back to the repo's
    current commit sha (one cheap API call) — a stable cache key that
    invalidates itself when the dataset is updated."""
    revision = revision or PINNED_REVISIONS.get(source) or HfApi().dataset_info(source).sha
    cache = _projected_cache_path(source, revision, columns) if revision else None
    if cache is not None and cache.exists():
        return pq.read_table(cache)

    files = sorted(
        name
        for name in HfApi().list_repo_files(source, revision=revision, repo_type="dataset")
        if name.endswith(".parquet")
    )
    fs = HfFileSystem()

    def read_shard(name: str) -> pa.Table:
        with fs.open(f"datasets/{source}/{name}", "rb", revision=revision) as handle:
            return pq.ParquetFile(handle, pre_buffer=True).read(columns=columns)

    with ThreadPoolExecutor(max_workers=max(1, len(files))) as pool:
        table = pa.concat_tables(list(pool.map(read_shard, files)))

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, cache)
    return table


def _projected_cache_path(source: str, revision: str, columns: list[str]) -> Path:
    """Local cache file for a source's projected columns at a pinned revision. The
    column set is folded into the name so a change to the columns invalidates it."""
    columns_tag = hashlib.sha1(",".join(columns).encode()).hexdigest()[:8]
    slug = source.replace("/", "__")
    return _GOLD_CACHE_DIR / f"{slug}@{revision}.{columns_tag}.parquet"


def conversation_ids(dataset: Dataset) -> list[str]:
    """The split's conversation ids (numeric task ids), sorted numerically."""
    return sorted(dataset.index, key=int)


def conversation(dataset: Dataset, conversation_id: str) -> Conversation:
    """Materialise one conversation: its annotation tracks, duration, and (unless
    the dataset was loaded with skip_audio) its audio bytes. Reads exactly one row.

    Duration comes from the committed durations artifact; it falls back to the
    audio header only for sources without one, which must be loaded with audio."""
    row = dataset.rows[dataset.index[conversation_id]]
    annotations = {
        (speaker, annotator): [
            (event["start_s"], event["end_s"], event["label"], event["text"])
            for event in row[f"speaker_{speaker}_annotation_{annotator}"]
        ]
        for speaker in SPEAKERS
        for annotator in ANNOTATORS
    }
    has_audio = f"speaker_{SPEAKERS[0]}_audio" in dataset.rows.column_names
    audio_bytes = (
        {speaker: row[f"speaker_{speaker}_audio"]["bytes"] for speaker in SPEAKERS}
        if has_audio
        else {}
    )
    duration_s = dataset.durations.get(conversation_id)
    if duration_s is None:
        if not audio_bytes:
            raise KeyError(
                f"conversation {conversation_id}: no committed duration for this source "
                "and audio not loaded — load with skip_audio=False or add a durations artifact"
            )
        info = {speaker: sf.info(io.BytesIO(audio_bytes[speaker])) for speaker in SPEAKERS}
        assert info[1].frames == info[2].frames, (
            f"conversation {conversation_id}: speaker channels differ in length "
            f"({info[1].frames} vs {info[2].frames} samples); the data is corrupt"
        )
        duration_s = info[1].frames / info[1].samplerate
    return Conversation(
        conversation_id=conversation_id,
        duration_s=duration_s,
        annotations=annotations,
        audio_bytes=audio_bytes,
    )
