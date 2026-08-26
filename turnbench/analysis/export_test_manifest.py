"""Export the public test split's viewer manifest, for the leaderboard site.

The website lists the test split on /conversations and its viewer streams
test audio, but the test labels are withheld — so instead of a gold
artifact the site vendors a tiny manifest: per conversation, the audio
duration (from the committed durations-test.json), the dataset `metadata`
struct (conversation_type, per-speaker actor id/gender), and audio_status.
The manifest is stamped with the dataset revision it was read at; the site
pins its audio range-reads to that revision.

    uv run python -m turnbench.analysis.export_test_manifest > test-manifest.json

Copy the output to the website repo as `public/test-manifest.json`.
"""

import json

from huggingface_hub import HfApi

from turnbench.data import read_columns_projected
from turnbench.durations import load_durations

TEST_DATASET = "mundo-ai/turn-benchmark-test"


def main() -> None:
    revision = HfApi().dataset_info(TEST_DATASET).sha
    table = read_columns_projected(
        TEST_DATASET, revision, ["conversation_id", "metadata", "audio_status"]
    )
    ids = table["conversation_id"].to_pylist()
    metadata = table["metadata"].to_pylist()
    audio_status = table["audio_status"].to_pylist()
    durations = load_durations("test")
    conversations = {
        cid: {
            "duration_s": durations[cid],
            "audio_status": status,
            "metadata": meta,
        }
        for cid, meta, status in sorted(
            zip(ids, metadata, audio_status), key=lambda row: int(row[0])
        )
    }
    print(
        json.dumps(
            {
                "dataset": TEST_DATASET,
                "dataset_revision": revision,
                "conversations": conversations,
            }
        )
    )


if __name__ == "__main__":
    main()
