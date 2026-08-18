"""Fetch the baseline probability files from their pinned HF dataset.

The per-baseline probs JSONs (38 files, ~1 GB — each baseline's per-frame
EOT/INT scores over dev and test) live in the public HF dataset REPO_ID, not
in git. This command materialises them into baselines/<baseline>/<file>.json
at the pinned REVISION; after that, every consumer (turnbench.sweep, turnbench.check,
turnbench/analysis/{plot_sweep,finalize_ops,merge_prob_shards}) reads plain local
files. Nothing fetches at scoring time.

    uv run python -m turnbench.probs            # download missing files
    uv run python -m turnbench.probs --force    # re-download everything

To add or refresh probs files: upload to REPO_ID (huggingface_hub
upload_folder over baselines/ with allow_patterns=["*probs*.json"]), then bump
REVISION to the new commit oid.
"""

import shutil
from pathlib import Path
from typing import Annotated

import typer
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "freemanjiang/turnbench-baseline-probs"
REVISION = "e3cd4caa2a45b85ef4d6714d79d5c7c40471ae79"

BASELINES_DIR = Path(__file__).resolve().parent.parent / "baselines"
FETCH_HINT = f"run `uv run python -m turnbench.probs` to download it from {REPO_ID}"

app = typer.Typer(add_completion=False)


@app.command()
def fetch(
    force: Annotated[
        bool, typer.Option(help="re-download files that already exist locally")
    ] = False,
) -> None:
    """Download every probs file from REPO_ID at REVISION into baselines/."""
    names = [
        name
        for name in list_repo_files(REPO_ID, repo_type="dataset", revision=REVISION)
        if name.endswith(".json") and "probs" in name
    ]
    fetched = 0
    for name in names:
        target = BASELINES_DIR / name
        if target.exists() and not force:
            continue
        cached = hf_hub_download(
            REPO_ID, filename=name, repo_type="dataset", revision=REVISION
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, target)
        fetched += 1
    typer.echo(
        f"{fetched} fetched, {len(names) - fetched} already present "
        f"({REPO_ID}@{REVISION[:10]})"
    )


if __name__ == "__main__":
    app()
