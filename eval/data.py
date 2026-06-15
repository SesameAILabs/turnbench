"""Resolve the benchmark dataset to a local directory of conversation dirs.

The scorer takes a predictions JSON and a dataset. By default the dataset is
the public dev set on HuggingFace (snapshot-downloaded into the local HF cache,
cache-first then network); `--dataset` accepts any HF dataset repo id, or a
local directory. The private test set is just a private HF repo, scored by the
same code server-side.
"""

from pathlib import Path

import soundfile as sf
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

DEV_DATASET = "freeman-sesame/turn-benchmark-dev"
DEV_REVISION = "e9be61f7737cc8b3518aee5b1f5f6e8f74ed8f53"


def resolve_dataset(source: str = DEV_DATASET, revision: str | None = None) -> Path:
    """Local path to the conversation directories to score.

    `source` is either a local directory (returned as-is) or an HF dataset repo
    id, snapshot-downloaded cache-first then over the network. `revision` pins
    the HF download; when omitted it defaults to the pinned dev revision for the
    dev dataset, else the dataset's latest revision.
    """
    local = Path(source)
    if local.is_dir():
        return local
    if revision is None and source == DEV_DATASET:
        revision = DEV_REVISION
    try:
        return Path(
            snapshot_download(
                source,
                repo_type="dataset",
                revision=revision,
                local_files_only=True,
            )
        )
    except LocalEntryNotFoundError:
        return Path(snapshot_download(source, repo_type="dataset", revision=revision))


def dataset_dir() -> Path:
    """The public dev dataset — convenience for tools that aren't dataset-parameterised
    (baselines, the gold CLI, the inspector)."""
    return resolve_dataset()


def conversation_ids(data_dir: Path) -> list[str]:
    """Conversation dir names under `data_dir` (numeric task ids), sorted numerically."""
    return sorted((d.name for d in data_dir.iterdir() if d.is_dir()), key=int)


def conversation_duration_s(conversation_dir: Path) -> float:
    """Duration (s) of the conversation's time-aligned speaker channels."""
    speaker_1 = sf.info(conversation_dir / "speaker_1_audio.flac")
    speaker_2 = sf.info(conversation_dir / "speaker_2_audio.flac")
    assert speaker_1.frames == speaker_2.frames, (
        f"speaker channels differ in length ({speaker_1.frames} vs "
        f"{speaker_2.frames} samples); the data is corrupt"
    )
    return speaker_1.frames / speaker_1.samplerate
