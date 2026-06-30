"""Per-conversation audio durations as a tiny committed artifact.

Validation (eval.check, eval.sweep) needs only conversation_ids + per-conversation
duration — never the audio samples. The dataset ships no duration column, so we
compute durations once from the audio headers and commit them as
`durations-{dev,test}.json` ({conversation_id: seconds}). The author check then
runs with no dataset download; the eval server still uses the real dataset to
score.

    python -m eval.durations                    # regenerate both splits (maintainer)
"""

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import pyarrow.parquet as pq
import soundfile as sf
import typer
from huggingface_hub import snapshot_download

_DIR = Path(__file__).resolve().parent
SPLITS = {"dev": "mundo-ai/turn-benchmark-dev", "test": "mundo-ai/turn-benchmark-test"}


def path_for(split: str) -> Path:
    return _DIR / f"durations-{split}.json"


def load_durations(split: str) -> dict[str, float]:
    """{conversation_id: duration_s} for a split, from the committed JSON."""
    return json.loads(path_for(split).read_text(encoding="utf-8"))["durations"]


def _compute(source: str, revision: str | None) -> dict[str, float]:
    """Compute durations one parquet shard at a time (bounded memory, and avoids
    the offset overflow that concatenating the full audio column hits)."""
    snapshot = snapshot_download(
        source, repo_type="dataset", revision=revision, allow_patterns="*.parquet"
    )
    durations: dict[str, float] = {}
    for shard in sorted(Path(snapshot).rglob("*.parquet")):
        table = pq.read_table(shard, columns=["conversation_id", "speaker_1_audio"])
        for cid, audio in zip(
            table["conversation_id"].to_pylist(), table["speaker_1_audio"].to_pylist()
        ):
            data = audio["bytes"] if isinstance(audio, dict) else audio
            info = sf.info(io.BytesIO(data))
            durations[cid] = info.frames / info.samplerate
    return dict(sorted(durations.items(), key=lambda kv: int(kv[0])))


app = typer.Typer(add_completion=False)


@app.command()
def generate(
    revision: Annotated[
        str | None, typer.Option(help="HF revision override (defaults to pinned)")
    ] = None,
) -> None:
    """Recompute durations-{dev,test}.json from the datasets (maintainer only)."""
    from eval.data import PINNED_REVISIONS

    for split, source in SPLITS.items():
        resolved = revision or PINNED_REVISIONS.get(source)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "revision": resolved,
            "durations": _compute(source, resolved),
        }
        path_for(split).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"wrote {path_for(split).name} — {len(payload['durations'])} conversations")


if __name__ == "__main__":
    app()
