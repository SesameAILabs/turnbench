"""Shared runner for baseline scripts.

Each baseline's `predict.py` implements one function, `predict_for_agent`,
that returns turn-taking event predictions assuming the model is the
agent listening to the OTHER speaker's channel. This module:

  - loads `TT_BENCHMARK_DATA` from the repo's `.env` (the dataset root,
    containing one subdirectory per `task_id`);
  - iterates over every sample directory;
  - runs `predict_for_agent` twice per sample (once with `agent_speaker=1`,
    once with `agent_speaker=2`) — bidirectional evaluation;
  - writes a single JSONL per sample to `predictions/<baseline>/<task_id>.jsonl`,
    with one `{"time": float, "speaker": int, "label": str}` per line.

A baseline's entry point is typically:

    from baselines.runner import run

    def predict_for_agent(sample_dir, agent_speaker):
        ...
        return [{"time": ..., "speaker": agent_speaker, "label": "EOT"}, ...]

    if __name__ == "__main__":
        run("gemini", predict_for_agent)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable


_REPO = Path(__file__).resolve().parent.parent


def load_env(p: Path = _REPO / ".env") -> dict[str, str]:
    env: dict[str, str] = {}
    if not p.exists():
        return env
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def data_root() -> Path:
    env = load_env()
    if "TT_BENCHMARK_DATA" not in env or not env["TT_BENCHMARK_DATA"]:
        sys.exit("Set TT_BENCHMARK_DATA in .env — see .env.example.")
    return Path(env["TT_BENCHMARK_DATA"])


def run(baseline_name: str,
        predict_for_agent: Callable[[Path, int], list[dict]]) -> int:
    root = data_root()
    out_dir = _REPO / "predictions" / baseline_name
    out_dir.mkdir(parents=True, exist_ok=True)
    n_samples = 0
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        preds: list[dict] = []
        for agent in (1, 2):
            preds.extend(predict_for_agent(d, agent))
        with (out_dir / f"{d.name}.jsonl").open("w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        n_samples += 1
    print(f"Wrote predictions for {n_samples} samples to {out_dir}",
          file=sys.stderr)
    return 0
