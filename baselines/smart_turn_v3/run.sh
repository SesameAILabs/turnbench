#!/usr/bin/env bash
# baselines/smart_turn_v3/run.sh — Smart Turn v3 prediction entry point
#
# Usage:
#   bash baselines/smart_turn_v3/run.sh          # default: dev + test + turnbench.check
#   bash baselines/smart_turn_v3/run.sh --dev    # dev: infer + probs + sweep + predictions-dev.json
#   bash baselines/smart_turn_v3/run.sh --test   # test: sweep existing probs → pick theta → predictions-test.json
#   bash baselines/smart_turn_v3/run.sh --probs  # dev inference → probs files only (no predictions)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

_HERE="baselines/smart_turn_v3"

MODE="${1:-}"
TEST_DATASET=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('mundo-ai/turn-benchmark-test', repo_type='dataset', local_files_only=True))")/data

# ── Helper: pick threshold from existing dev probs via turnbench.sweep ──────────
_pick_threshold() {
    local task="$1"
    local fallback="$2"
    local probs_path="${_HERE}/probs-${task}.json"
    [[ -f "$probs_path" ]] || { echo "Missing ${probs_path} — run --dev first" >&2; exit 1; }
    python -c "
import sys
from pathlib import Path
from turnbench.sweep import load_probs, sweep, operating_point
from turnbench.data import resolve_dataset, DEV_DATASET
probs = load_probs(Path('$probs_path'))
rows = sweep(probs, resolve_dataset(source=DEV_DATASET))
op = operating_point(rows)
if op is not None:
    print(f'[sweep] $task theta={op.theta} recall={op.recall:.3f} fp_rate={op.fp_rate:.3f}', file=sys.stderr)
    print(op.theta)
else:
    print(f'[sweep] no operating point found for $task, using fallback=$fallback', file=sys.stderr)
    print($fallback)
"
}

# ── _run_dev: inference + probs + auto-threshold + predictions-dev.json ──────
_run_dev() {
    rm -f "${_HERE}/probs-eot.json" "${_HERE}/probs-int.json"
    rm -f "${_HERE}/predictions-dev.json"
    python -m baselines.smart_turn_v3.predict
}

case "$MODE" in

--dev)
    _run_dev
    ;;

--test)
    FALLBACK_EOT=$(python -c "from baselines.smart_turn_v3.predict import CHECKPOINT_DEFAULTS; print(CHECKPOINT_DEFAULTS['pretrained'][0])")
    FALLBACK_INT=$(python -c "from baselines.smart_turn_v3.predict import CHECKPOINT_DEFAULTS; print(CHECKPOINT_DEFAULTS['pretrained'][1])")
    THR_EOT=$(_pick_threshold eot "$FALLBACK_EOT")
    THR_INT=$(_pick_threshold int "$FALLBACK_INT")
    echo "Thresholds from sweep: eot=${THR_EOT}  int=${THR_INT}"

    rm -f "${_HERE}/predictions-test.json"
    python -m baselines.smart_turn_v3.predict \
        --dataset "$TEST_DATASET" \
        --threshold-eot "$THR_EOT" \
        --threshold-int "$THR_INT"
    ;;

--probs)
    rm -f "${_HERE}/probs-eot.json" "${_HERE}/probs-int.json"
    python -m baselines.smart_turn_v3.predict --probs-only
    ;;

"")
    _run_dev

    FALLBACK_EOT=$(python -c "from baselines.smart_turn_v3.predict import CHECKPOINT_DEFAULTS; print(CHECKPOINT_DEFAULTS['pretrained'][0])")
    FALLBACK_INT=$(python -c "from baselines.smart_turn_v3.predict import CHECKPOINT_DEFAULTS; print(CHECKPOINT_DEFAULTS['pretrained'][1])")
    THR_EOT=$(_pick_threshold eot "$FALLBACK_EOT")
    THR_INT=$(_pick_threshold int "$FALLBACK_INT")
    echo "Thresholds from sweep: eot=${THR_EOT}  int=${THR_INT}"

    rm -f "${_HERE}/predictions-test.json"
    python -m baselines.smart_turn_v3.predict \
        --dataset "$TEST_DATASET" \
        --threshold-eot "$THR_EOT" \
        --threshold-int "$THR_INT"

    python -m turnbench.check "$_HERE"
    ;;

*)
    echo "Unknown mode: $MODE" >&2
    echo "Usage: $0 [--dev|--test|--probs]" >&2
    exit 1
    ;;
esac
