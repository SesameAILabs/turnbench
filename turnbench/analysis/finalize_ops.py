#!/usr/bin/env python3
"""Finalize every prob-bearing baseline at its swept operating point — dev AND test.

For each baseline directory that has committed dev probs:

  1. dev: sweep `probs-{eot,int}.json` with turnbench.sweep (candidate thresholds =
     score quantiles ∪ uniform 0.01 grid; op = highest recall at fp_rate ≤ 0.1)
     → θ_eot, θ_int → regenerate `predictions-dev.json` via the central
     `commit_events` rule. Consistent-by-construction with the committed probs.
  2. test: take the model's per-frame TEST probs (emitted once by its predict.py
     — see --test-probs-src), round to 6 decimals, and bank them as
     `probs-test-{eot,int}.json` next to the dev probs. Regenerate
     `predictions-test.json` = commit_events(banked test probs, θ_dev).
     Banked test probs make any future op/grid change a re-threshold, not a
     re-inference; committing predictions FROM the rounded file keeps the
     committed artifacts exactly self-reproducing.
  3. validate: probs-test files against the canonical test frame grid.

Test-probs sources default to the staging dirs used by the run scripts; espnet's
are banked in-place by its submit.py. Score with --score (dev locally, test
against the gold dataset — needs the mundo-ai HF token).

    uv run --extra eval python turnbench/analysis/finalize_ops.py [--write] [--score]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from turnbench.data import DEV_DATASET, resolve_dataset
from turnbench.durations import load_durations
from turnbench.sweep import (
    ProbsFile,
    commit_events,
    load_probs,
    operating_point,
    sweep,
    validate_probs,
)

BASE = Path(__file__).resolve().parents[2] / "baselines"
GOLD_DATASET = "mundo-ai/turn-benchmark-test-golden"

# baseline -> staging dir holding its raw TEST probs (probs-{eot,int}.json).
# espnet's submit.py banks probs-test-*.json into the baseline dir directly.
TEST_PROBS_SRC = {
    "espnet_turntaking": None,               # banked in-place
    "espnet_turntaking_perchannel": None,    # banked in-place
    "vap": "predictions/vap_test_probs",
    "mimi_endpointer": "predictions/mimi_test_probs",
    "kyutai_semantic_vad": "predictions/kyutai_test_probs_merged",
    "smart_turn_v3": "predictions/smart_turn_test_probs",
    "wavlm_base_causal": "predictions/wavlm_probs/wavlm_base_causal_test",
    "wavlm_large_causal": "predictions/wavlm_probs/wavlm_large_causal_test",
    "wavlm_large_anchor": "predictions/wavlm_probs/wavlm_large_anchor_test",
}


def _round_probs(pf: ProbsFile, decimals: int = 6) -> ProbsFile:
    """Round every per-frame value for compact storage. 1e-6 resolution is far
    below any operating point in use; boundary flips affect a vanishing fraction
    of frames and the rounded file is the artifact of record (predictions are
    committed FROM it, so everything stays exactly reproducible)."""
    for entry in pf.probs:
        entry.speaker_1.prob = [round(v, decimals) for v in entry.speaker_1.prob]
        entry.speaker_2.prob = [round(v, decimals) for v in entry.speaker_2.prob]
    return pf


def _predictions(thetas: dict[str, float], probs_by_task: dict[str, ProbsFile]) -> dict:
    """Central commit at the per-task thetas → a Submission-shaped dict."""
    eot_pf, int_pf = probs_by_task["eot"], probs_by_task["int"]
    int_by_id = int_pf.by_conversation()
    preds = []
    for entry in eot_pf.probs:
        ie = int_by_id[entry.conversation_id]
        preds.append({
            "conversation_id": entry.conversation_id,
            "speaker_1": {
                "eot": commit_events(entry.speaker_1.prob, eot_pf.frame_rate_hz, thetas["eot"]),
                "interruption": commit_events(ie.speaker_1.prob, int_pf.frame_rate_hz, thetas["int"]),
            },
            "speaker_2": {
                "eot": commit_events(entry.speaker_2.prob, eot_pf.frame_rate_hz, thetas["eot"]),
                "interruption": commit_events(ie.speaker_2.prob, int_pf.frame_rate_hz, thetas["int"]),
            },
        })
    return {"schema_version": 1, "predictions": preds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write predictions + banked test probs")
    ap.add_argument("--score", action="store_true", help="score dev + test(vs gold) after finalizing")
    ap.add_argument("--baselines", nargs="*", default=list(TEST_PROBS_SRC),
                    help="subset of baselines to finalize")
    args = ap.parse_args()

    dev = resolve_dataset(source=DEV_DATASET, skip_audio=True)
    test_durations = load_durations("test")
    repo = BASE.parent

    all_thetas: dict[str, dict[str, float]] = {}
    for name in args.baselines:
        d = BASE / name
        # ---- dev: op per task + regenerated predictions-dev ----
        dev_probs, thetas = {}, {}
        for task in ("eot", "int"):
            pf = load_probs(d / f"probs-{task}.json")
            op = operating_point(sweep(pf, dev))
            if op is None:
                print(f"[{name}/{task}] NO operating point at fp<=0.1 — skipping baseline")
                break
            dev_probs[task], thetas[task] = pf, op.theta
            print(f"[{name}/{task}] dev op: theta={op.theta!r} recall={op.recall:.3f} fp={op.fp_rate:.3f}")
        else:
            all_thetas[name] = thetas
            if args.write:
                (d / "predictions-dev.json").write_text(
                    json.dumps(_predictions(thetas, dev_probs), indent=2))
                print(f"[{name}] wrote predictions-dev.json")

            # ---- test: bank rounded probs + predictions from them ----
            src = TEST_PROBS_SRC.get(name)
            test_probs = {}
            for task in ("eot", "int"):
                raw = (repo / src / f"probs-{task}.json") if src else (d / f"probs-test-{task}.json")
                if not raw.exists():
                    print(f"[{name}/{task}] test probs missing at {raw} — test skipped")
                    break
                pf = _round_probs(load_probs(raw))
                validate_probs(pf, test_durations)
                test_probs[task] = pf
                if args.write:
                    (d / f"probs-test-{task}.json").write_text(pf.model_dump_json())
                    print(f"[{name}] banked probs-test-{task}.json")
            if len(test_probs) == 2 and args.write:
                (d / "predictions-test.json").write_text(
                    json.dumps(_predictions(thetas, test_probs), indent=2))
                print(f"[{name}] wrote predictions-test.json @ {thetas}")

    if args.score:
        from turnbench.score import score_submission, task_cells
        from turnbench.submission import Submission
        gold = resolve_dataset(source=GOLD_DATASET, skip_audio=True)
        print(f"\n{'baseline':<30} {'split':<5} recall/fp  (EOT | INT)")
        print("-" * 80)
        for name in args.baselines:
            for split, ds in (("dev", dev), ("test", gold)):
                p = BASE / name / f"predictions-{split}.json"
                if not p.exists():
                    print(f"{name:<30} {split:<5} (missing)"); continue
                s = score_submission(Submission.model_validate_json(p.read_text()), ds)
                e, i = task_cells(s.task_eot), task_cells(s.task_int)
                print(f"{name:<30} {split:<5} EOT {e[0]}/{e[1]}  INT {i[0]}/{i[1]}")


if __name__ == "__main__":
    main()
