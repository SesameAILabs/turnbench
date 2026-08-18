"""Pre-submission check: validate every file a baseline commits, in one command.

A baseline commits up to four files (see baselines/README.md):

    predictions-dev.json   predictions-test.json   probs-eot.json   probs-int.json

`python -m turnbench.check baselines/<name>` validates each one that is present and
prints a pass/fail line per file, so a submitter can confirm a fresh clone is
submission-ready before opening a PR. It needs no audio and no download — only the
committed per-conversation durations (eval/durations-{dev,test}.json) for coverage,
in-audio event times, and the probs frame grid. Validation only; it does not score
(dev scores are turnbench.score). predictions-test.json can *only* be validated this way
— it can't be scored locally, since test labels are withheld.
"""

from pathlib import Path
from typing import Annotated, Callable

import typer

from turnbench.durations import load_durations
from turnbench.submission import load_submission, validate_coverage, validate_event_times
from turnbench.sweep import load_probs, validate_probs


def check_predictions(path: Path, durations: dict[str, float]) -> None:
    """Schema + coverage + in-audio event times for a predictions file."""
    submission = load_submission(path)
    validate_coverage(submission, list(durations))
    for prediction in submission.predictions:
        validate_event_times(prediction, durations[prediction.conversation_id])


def check_probs(path: Path, durations: dict[str, float]) -> None:
    """Schema + coverage + exact frame counts for a probabilities file."""
    validate_probs(load_probs(path), durations)


def main(
    baseline_dir: Annotated[Path, typer.Argument(help="a baselines/<name>/ directory")],
) -> None:
    """Validate every submission file present in a baseline directory."""
    dev = load_durations("dev")
    test = load_durations("test")

    # (filename, validator, absent_note). absent_note=None marks a required file —
    # missing is an error. Otherwise missing is a warning: a probs file is absent
    # only when the model has no such per-frame probability (e.g. rms_vad has no
    # probabilities at all), so the author should confirm that's deliberate.
    files: list[tuple[str, Callable[[Path], None], str | None]] = [
        ("predictions-dev.json", lambda p: check_predictions(p, dev), None),
        ("predictions-test.json", lambda p: check_predictions(p, test), None),
        (
            "probs-eot.json",
            lambda p: check_probs(p, dev),
            "omit only if your model does not emit per-frame EOT probabilities",
        ),
        (
            "probs-int.json",
            lambda p: check_probs(p, dev),
            "omit only if your model does not emit per-frame INT probabilities",
        ),
    ]

    failures = 0
    warnings = 0
    for name, validate, absent_note in files:
        path = baseline_dir / name
        if not path.exists():
            if absent_note is None:
                typer.secho(f"✗ {name}: missing (required)", fg="red")
                failures += 1
            else:
                typer.secho(f"⚠ {name}: absent — {absent_note}", fg="yellow")
                warnings += 1
            continue
        try:
            validate(path)
            typer.secho(f"✓ {name}", fg="green")
        except (
            Exception
        ) as error:  # report each file's failure rather than crash on the first
            typer.secho(f"✗ {name}: {error}", fg="red")
            failures += 1

    if failures:
        raise typer.Exit(code=1)
    if warnings:
        typer.secho(
            f"\nvalid, but {warnings} optional file(s) absent",
            fg="yellow",
            bold=True,
        )
    else:
        typer.secho("all files valid — ready to submit", fg="green", bold=True)


if __name__ == "__main__":
    typer.run(main)
