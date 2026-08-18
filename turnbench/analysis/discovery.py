"""Where the artifact scripts find predictions to score.

Baselines are committed under baselines/<name>/. External submissions are the
same <name>/predictions-{dev,test}.json layout in a directory that lives
OUTSIDE any public repo — their test predictions must never be published, or
the next submitter could copy a leaderboard position — passed to the scripts
via --submissions. leaderboard.py, results_by_conversation_type.py, and
export_dev_predictions.py all discover through this module, so the two
sources stay in lockstep everywhere a model list is built.
"""
from __future__ import annotations

from pathlib import Path

BASELINES_DIR = Path(__file__).resolve().parents[2] / "baselines"


def discover(split: str, submissions: Path | None = None) -> dict[str, Path]:
    """{label: predictions_path} for every model with a predictions file for
    `split`: the committed baselines plus, when given, an external submissions
    directory. `-variant` files keep their suffix in the label
    (`name/variant`). A submission whose label collides with an existing model
    is an error rather than a silent shadow."""
    found: dict[str, Path] = {}
    roots = [BASELINES_DIR] + ([submissions] if submissions else [])
    for root in roots:
        for path in sorted(root.glob(f"*/predictions-{split}*.json")):
            variant = path.stem[len(f"predictions-{split}"):].lstrip("-")
            label = path.parent.name + (f"/{variant}" if variant else "")
            if label in found:
                raise SystemExit(
                    f"duplicate model label {label!r}: {found[label]} and {path}"
                )
            found[label] = path
    return found
