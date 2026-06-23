"""Resolve the benchmark dataset (HuggingFace parquet) to in-memory conversations.

The scorer takes a predictions JSON and a dataset. By default the dataset is the
public dev set on HuggingFace: parquet shards, one row per conversation, each row
carrying the two per-speaker audio channels and the raw three-annotator tracks
per speaker. `--dataset` accepts any HF dataset repo id, or a local directory of
parquet shards. The private test set is just a private HF repo, scored by the
same code server-side. Authentication uses ambient HF credentials (HF_TOKEN /
`huggingface-cli login`); the public sets are gated, so callers must have been
granted access to the repo.

Shards are read with pyarrow and wrapped via the HFDataset constructor, not
`datasets.load_dataset` — see resolve_dataset for why (load_dataset overflows on
the full test split).
"""

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from datasets import Audio, Dataset as HFDataset
from huggingface_hub import snapshot_download

DEV_DATASET = "mundo-ai/turn-benchmark-dev"
DEV_REVISION = "8fa18a24be51528a45397b35cbcaecd84202062b"

# Pinned revisions for reproducibility; sources not listed float to latest.
PINNED_REVISIONS = {DEV_DATASET: DEV_REVISION}

ANNOTATORS = ("a", "b", "c")
SPEAKERS = (1, 2)

# One annotated segment as it lives in the parquet annotation columns, reduced to
# what consensus needs: (start_s, end_s, fine_label). The verbatim annotator label
# is mapped to the canonical taxonomy later (eval/gold.py).
Annotation = tuple[float, float, str]


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
    """A loaded benchmark split: the HF rows plus a conversation_id -> row index."""

    rows: HFDataset
    index: dict[str, int]


def resolve_dataset(source: str = DEV_DATASET, revision: str | None = None) -> Dataset:
    """Load the dataset's single split (Arrow-backed) and index it by id.

    `source` is an HF dataset repo id (snapshot-cached) or a local directory of
    parquet shards. `revision` pins the HF download; when omitted it defaults to
    the pinned dev revision for the dev dataset, else the dataset's latest.
    Authentication uses ambient HF credentials. Audio columns are kept undecoded —
    duration is read from the header and full samples are decoded only on demand
    (Conversation.audio).

    The shards are read with pyarrow and wrapped via the HFDataset constructor
    rather than `datasets.load_dataset`. load_dataset writes an Arrow cache, and
    its writer calls `combine_chunks()`, merging each column's chunks into one
    contiguous array. The embedded audio is a `binary` column (32-bit offsets,
    ~2 GB per array), so the full test split overflows it (`ArrowInvalid: offset
    overflow while concatenating arrays`); the 38-conv dev set is small enough to
    dodge it. The constructor wraps the chunked table as-is — chunks are never
    combined — and is verified to score identically to load_dataset on dev.
    """
    if Path(source).is_dir():
        files = sorted(str(path) for path in Path(source).glob("*.parquet"))
    else:
        if revision is None:
            revision = PINNED_REVISIONS.get(source)
        snapshot = snapshot_download(
            source, repo_type="dataset", revision=revision, allow_patterns="*.parquet"
        )
        files = sorted(str(path) for path in Path(snapshot).rglob("*.parquet"))
    rows = HFDataset(pa.concat_tables([pq.read_table(file) for file in files]))
    # The dataset also carries Opus `_preview` columns for the web viewer; the
    # scorer doesn't use them and must not try to decode them, so drop them.
    rows = rows.remove_columns(
        [name for name in rows.column_names if name.endswith("_preview")]
    )
    for speaker in SPEAKERS:
        rows = rows.cast_column(f"speaker_{speaker}_audio", Audio(decode=False))
    index = {row_id: i for i, row_id in enumerate(rows["conversation_id"])}
    return Dataset(rows=rows, index=index)


def conversation_ids(dataset: Dataset) -> list[str]:
    """The split's conversation ids (numeric task ids), sorted numerically."""
    return sorted(dataset.index, key=int)


def conversation(dataset: Dataset, conversation_id: str) -> Conversation:
    """Materialise one conversation: its annotation tracks, audio duration, and
    audio bytes (decoded on demand). Reads exactly one row."""
    row = dataset.rows[dataset.index[conversation_id]]
    annotations = {
        (speaker, annotator): [
            (event["start_s"], event["end_s"], event["label"])
            for event in row[f"speaker_{speaker}_annotation_{annotator}"]
        ]
        for speaker in SPEAKERS
        for annotator in ANNOTATORS
    }
    audio_bytes = {
        speaker: row[f"speaker_{speaker}_audio"]["bytes"] for speaker in SPEAKERS
    }
    info = {speaker: sf.info(io.BytesIO(audio_bytes[speaker])) for speaker in SPEAKERS}
    assert info[1].frames == info[2].frames, (
        f"conversation {conversation_id}: speaker channels differ in length "
        f"({info[1].frames} vs {info[2].frames} samples); the data is corrupt"
    )
    return Conversation(
        conversation_id=conversation_id,
        duration_s=info[1].frames / info[1].samplerate,
        annotations=annotations,
        audio_bytes=audio_bytes,
    )
