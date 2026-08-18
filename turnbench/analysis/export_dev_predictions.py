#!/usr/bin/env python3
"""Export the committed dev predictions as a website-consumable folder.

The website renders a per-model page for every baseline: it fetches that
model's predictions-dev.json, scores it in the browser against the vendored
dev gold (the same parity-locked TS scorer the /dev page uses), and overlays
the fires in the conversation viewer. This script produces the folder the
site serves: one validated predictions JSON per baseline plus an index.json
manifest listing what exists. The folder is vendored into the website repo
(like leaderboard.json) — regenerate and re-copy when baselines change:

    uv run python turnbench/analysis/export_dev_predictions.py <out>
    # then copy into turn-benchmark:
    #   <out>/* -> site/public/predictions-dev/

index.json shape (consumed by the site's lib/model-predictions.ts):

    {"generated_at_sha": "<tt-benchmark HEAD>",
     "models": [{"name": "vap", "file": "vap.json"}, ...]}
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


from turnbench.analysis.discovery import BASELINES_DIR, discover  # noqa: E402
from turnbench.submission import load_submission  # noqa: E402


def export(out: Path, submissions: Path | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=BASELINES_DIR.parent, capture_output=True, text=True, check=True,
    ).stdout.strip()

    models = []
    for label, path in discover("dev", submissions).items():
        load_submission(path)  # validate before publishing
        filename = label.replace("/", "-") + ".json"
        shutil.copyfile(path, out / filename)
        models.append({"name": label, "file": filename})
    (out / "index.json").write_text(
        json.dumps({"generated_at_sha": sha, "models": models}, indent=2) + "\n"
    )
    print(f"exported {len(models)} models -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path, help="output directory for the export")
    ap.add_argument("--submissions", type=Path, default=None,
                    help="external submissions dir (same <name>/predictions-dev.json layout)")
    args = ap.parse_args()
    export(args.out, submissions=args.submissions)


if __name__ == "__main__":
    main()
