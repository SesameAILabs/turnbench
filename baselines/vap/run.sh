#!/usr/bin/env bash
# baselines/vap/run.sh — VAP prediction entry point
#
# Usage:
#   bash baselines/vap/run.sh                         # default: dev + test (oto checkpoint, no prefix)
#   bash baselines/vap/run.sh --dev                   # dev only, oto checkpoint
#   bash baselines/vap/run.sh --dev  --pretrained     # dev, pretrained checkpoint (pretrained- prefix)
#   bash baselines/vap/run.sh --dev  --swbd           # dev, swbd checkpoint
#   bash baselines/vap/run.sh --dev  --swbd-oto       # dev, swbd_oto checkpoint
#   bash baselines/vap/run.sh --test                  # test: sweep existing probs → pick theta → predictions-test.json
#   bash baselines/vap/run.sh --test --pretrained     # test, pretrained checkpoint
#   bash baselines/vap/run.sh --probs                 # dev inference → probs files only (no predictions)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

_HERE="baselines/vap"

# ── Parse mode + checkpoint flag ──────────────────────────────────────────────
MODE="${1:-}"; shift || true; CKP_FLAG="${1:-}"

case "$CKP_FLAG" in
    --pretrained) RUN_NAME="pretrained"; CKP_ARGS="--run-name pretrained" ;;
    --swbd)       RUN_NAME="swbd";       CKP_ARGS="--run-name swbd"       ;;
    --swbd-oto)   RUN_NAME="swbd_oto";   CKP_ARGS="--run-name swbd_oto"   ;;
    "")           RUN_NAME="oto";        CKP_ARGS=""                      ;;
    *) echo "Unknown checkpoint flag: $CKP_FLAG" >&2; exit 1 ;;
esac

PFX=""; [[ "$RUN_NAME" != "oto" ]] && PFX="${RUN_NAME}-"

TEST_DATASET="mundo-ai/turn-benchmark-test"

# ── Helper: pick threshold from existing probs file via turnbench.sweep ─────────
_pick_threshold() {
    local task="$1"
    local fallback="$2"
    local probs_path="${_HERE}/${PFX}probs-${task}.json"
    [[ -f "$probs_path" ]] || { echo "Missing ${probs_path}" >&2; exit 1; }
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
    rm -f "${_HERE}/${PFX}probs-eot.json" "${_HERE}/${PFX}probs-int.json"
    rm -f "${_HERE}/${PFX}predictions-dev.json"
    python -m baselines.vap.predict $CKP_ARGS
}

# ── Modes ─────────────────────────────────────────────────────────────────────
case "$MODE" in

--dev)
    _run_dev
    ;;

--test)
    FALLBACK_EOT=$(python -c "
from baselines.vap.predict import CHECKPOINT_DEFAULTS
d = CHECKPOINT_DEFAULTS.get('$RUN_NAME', (0.5, 0.5))
print(d[0])
")
    FALLBACK_INT=$(python -c "
from baselines.vap.predict import CHECKPOINT_DEFAULTS
d = CHECKPOINT_DEFAULTS.get('$RUN_NAME', (0.5, 0.5))
print(d[1])
")
    THR_EOT=$(_pick_threshold eot "$FALLBACK_EOT")
    THR_INT=$(_pick_threshold int "$FALLBACK_INT")
    echo "Thresholds from sweep: eot=${THR_EOT}  int=${THR_INT}"

    rm -f "${_HERE}/${PFX}predictions-test.json"
    python -m baselines.vap.predict $CKP_ARGS \
        --dataset "$TEST_DATASET" \
        --threshold-eot "$THR_EOT" \
        --threshold-int "$THR_INT"
    ;;

--probs)
    rm -f "${_HERE}/${PFX}probs-eot.json" "${_HERE}/${PFX}probs-int.json"
    python -m baselines.vap.predict $CKP_ARGS --probs-only
    ;;

"")
    _run_dev

    FALLBACK_EOT=$(python -c "
from baselines.vap.predict import CHECKPOINT_DEFAULTS
d = CHECKPOINT_DEFAULTS.get('$RUN_NAME', (0.5, 0.5))
print(d[0])
")
    FALLBACK_INT=$(python -c "
from baselines.vap.predict import CHECKPOINT_DEFAULTS
d = CHECKPOINT_DEFAULTS.get('$RUN_NAME', (0.5, 0.5))
print(d[1])
")
    THR_EOT=$(_pick_threshold eot "$FALLBACK_EOT")
    THR_INT=$(_pick_threshold int "$FALLBACK_INT")
    echo "Thresholds from sweep: eot=${THR_EOT}  int=${THR_INT}"

    rm -f "${_HERE}/${PFX}predictions-test.json"
    python -m baselines.vap.predict $CKP_ARGS \
        --dataset "$TEST_DATASET" \
        --threshold-eot "$THR_EOT" \
        --threshold-int "$THR_INT"

    python -m turnbench.check "$_HERE"
    ;;

*)
    echo "Unknown mode: $MODE" >&2
    echo "Usage: $0 [--dev|--test|--probs] [--pretrained|--swbd|--swbd-oto]" >&2
    exit 1
    ;;
esac
