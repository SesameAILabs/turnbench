"""Emit the website's parity bundle from THIS (canonical) scorer.

The leaderboard site runs a TypeScript port of this scorer in the browser.
Parity is a data contract, not shared code: this writes `dev-gold.json` plus
two `(predictions, expected-scores)` test vectors, and the site's vitest suite
asserts its TS scorer reproduces the expected scores exactly. The vectors are
`vad` (fires on every silence — exercises TPs, FPs, one-to-one claiming) and
`no_events` (the smallest valid submission).

After any scorer or gold change: regenerate, copy the output into the site
(`site/public/dev-gold.json` and `site/src/lib/parity/`), and re-run the site's
vitest. `dev-gold.json` carries `scorer_sha`, so a mismatch against the site's
vendored copy is the tripwire that the TS port needs re-syncing.

    uv run python -m eval.parity [OUT_DIR]   # default: ./parity_out
"""
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from eval.data import Dataset, conversation, conversation_ids, resolve_dataset
from eval.score import ConversationScores, TaskScore, merge, score_conversation
from eval.submission import (
    SCHEMA_VERSION,
    ConversationPrediction,
    SpeakerEvents,
    Submission,
    load_submission,
)

REPO = Path(__file__).resolve().parent.parent


def task_dict(score: TaskScore) -> dict:
    """One task's score in the parity wire format (latency None when no TPs)."""
    latency = score.latency()
    return {
        "tp": score.tp,
        "fn": score.fn,
        "fp": score.fp,
        "tn": score.tn,
        "latencies_ms": score.latencies_ms,
        "latency_ms": None
        if math.isnan(latency.p50)
        else {"p10": latency.p10, "p50": latency.p50, "p90": latency.p90},
    }


def expected_scores(predictions_path: Path, dataset: Dataset) -> dict:
    """Score a predictions file with this scorer, in the parity wire format."""
    by_conversation = load_submission(predictions_path).by_conversation()
    totals = ConversationScores(TaskScore(), TaskScore())
    conversations = {}
    for task_id in conversation_ids(dataset):
        scores = score_conversation(by_conversation[task_id], conversation(dataset, task_id))
        merge(totals.task_eot, scores.task_eot)
        merge(totals.task_int, scores.task_int)
        conversations[task_id] = {
            "eot": task_dict(scores.task_eot),
            "int": task_dict(scores.task_int),
        }
    return {
        "conversations": conversations,
        "aggregate": {
            "eot": task_dict(totals.task_eot),
            "int": task_dict(totals.task_int),
        },
    }


def write_no_events_predictions(path: Path, dataset: Dataset) -> None:
    empty = SpeakerEvents(eot=[], interruption=[])
    submission = Submission(
        schema_version=SCHEMA_VERSION,
        predictions=[
            ConversationPrediction(conversation_id=task_id, speaker_1=empty, speaker_2=empty)
            for task_id in conversation_ids(dataset)
        ],
    )
    path.write_text(submission.model_dump_json(indent=2), encoding="utf-8")


def run_module(module: str, *args: str) -> str:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def export(
    out_dir: Annotated[Path, typer.Argument(help="where to write the bundle")] = Path("parity_out"),
) -> None:
    """Write dev-gold.json + the parity test vectors to OUT_DIR."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = resolve_dataset(skip_audio=True)

    (out_dir / "dev-gold.json").write_text(run_module("eval.gold", "export"), encoding="utf-8")
    run_module("baselines.rms_vad.predict", "--out", str(out_dir / "vad_predictions.json"))
    write_no_events_predictions(out_dir / "no_events_predictions.json", dataset)

    for name in ("vad", "no_events"):
        expected = expected_scores(out_dir / f"{name}_predictions.json", dataset)
        (out_dir / f"{name}_expected.json").write_text(json.dumps(expected), encoding="utf-8")

    print(f"wrote parity bundle to {out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    typer.run(export)
