#!/usr/bin/env bash
# baselines/mimi_endpointer/run.sh — Mimi Endpointer prediction entry point
#
# Usage:
#   bash baselines/mimi_endpointer/run.sh --dev  [--oto|--swbd|--swbd-oto]  # dev: probs + predictions + score + sweep tables
#   bash baselines/mimi_endpointer/run.sh --test [--oto|--swbd|--swbd-oto]  # test: sweep existing probs → pick threshold → predictions
#   bash baselines/mimi_endpointer/run.sh        [--oto|--swbd|--swbd-oto]  # full: --dev then --test + turnbench.check
#   bash baselines/mimi_endpointer/run.sh --sweep [--oto|--swbd|--swbd-oto] # probs-only + turnbench.sweep tables
#   bash baselines/mimi_endpointer/run.sh --probs [--oto|--swbd|--swbd-oto] # probs-only
#
# Checkpoint flag sets run name and output prefix; omit for pretrained (default).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# ── parse mode and checkpoint flag ──────────────────────────────────────────
MODE="${1:-}"
shift || true
CKP_FLAG="${1:-}"

case "$CKP_FLAG" in
    --oto)      RUN_NAME="oto_d1f";      CKP_ARGS="--run-name oto_d1f" ;;
    --swbd)     RUN_NAME="swbd_d1f";     CKP_ARGS="--run-name swbd_d1f" ;;
    --swbd-oto) RUN_NAME="swbd_oto_d1f"; CKP_ARGS="--run-name swbd_oto_d1f" ;;
    "")         RUN_NAME="pretrained";   CKP_ARGS="" ;;
    *)          echo "Unknown checkpoint flag: $CKP_FLAG" >&2
                echo "Valid: --oto  --swbd  --swbd-oto" >&2; exit 1 ;;
esac

# output prefix: empty for pretrained, "{run_name}-" for others
PFX=""
[[ "$RUN_NAME" != "pretrained" ]] && PFX="${RUN_NAME}-"

TEST_DATASET=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('mundo-ai/turn-benchmark-test', repo_type='dataset', local_files_only=True))")/data
OUT_DIR="baselines/mimi_endpointer"
METRICS_FILE="$OUT_DIR/${PFX}predictions-dev-metrics.json"

# Run turnbench.sweep on an existing probs file and return the operating-point theta.
# Falls back to the predefined default if no threshold satisfies fp_budget ≤ 0.1.
_pick_threshold() {
    local probs_path="$1"
    local fallback="$2"
    python -c "
from pathlib import Path
from turnbench.sweep import load_probs, sweep, operating_point
from turnbench.data import resolve_dataset, DEV_DATASET
probs = load_probs(Path('$probs_path'))
rows  = sweep(probs, resolve_dataset(source=DEV_DATASET))
op    = operating_point(rows)
print(op.theta if op is not None else $fallback)
"
}

_run_dev() {
    echo "=== dev: $RUN_NAME ==="
    # shellcheck disable=SC2086
    python -m baselines.mimi_endpointer.predict \
        $CKP_ARGS \
        --out "$OUT_DIR/${PFX}predictions-dev.json"

    echo "=== turnbench.sweep: EOT ($RUN_NAME) ==="
    python -m turnbench.sweep "$OUT_DIR/${PFX}probs-eot.json"
    echo "=== turnbench.sweep: INT ($RUN_NAME) ==="
    python -m turnbench.sweep "$OUT_DIR/${PFX}probs-int.json"
}

case "$MODE" in

# ── dev only ─────────────────────────────────────────────────────────────────
--dev)
    _run_dev
    ;;

# ── test only: sweep existing probs → pick threshold → test predictions ──────
--test)
    [[ -f "$OUT_DIR/${PFX}probs-eot.json" ]] || { echo "Missing $OUT_DIR/${PFX}probs-eot.json — run --dev or --sweep first" >&2; exit 1; }
    [[ -f "$OUT_DIR/${PFX}probs-int.json" ]] || { echo "Missing $OUT_DIR/${PFX}probs-int.json — run --dev or --sweep first" >&2; exit 1; }

    # Get CHECKPOINT_DEFAULTS fallbacks for this run
    FALLBACK_EOT=$(python -c "from baselines.mimi_endpointer.predict import CHECKPOINT_DEFAULTS; print(CHECKPOINT_DEFAULTS.get('$RUN_NAME', CHECKPOINT_DEFAULTS['pretrained'])[0])")
    FALLBACK_INT=$(python -c "from baselines.mimi_endpointer.predict import CHECKPOINT_DEFAULTS; print(CHECKPOINT_DEFAULTS.get('$RUN_NAME', CHECKPOINT_DEFAULTS['pretrained'])[1])")

    echo "=== turnbench.sweep: EOT ($RUN_NAME) ==="
    THR_EOT=$(_pick_threshold "$OUT_DIR/${PFX}probs-eot.json" "$FALLBACK_EOT")
    python -m turnbench.sweep "$OUT_DIR/${PFX}probs-eot.json"

    echo "=== turnbench.sweep: INT ($RUN_NAME) ==="
    THR_INT=$(_pick_threshold "$OUT_DIR/${PFX}probs-int.json" "$FALLBACK_INT")
    python -m turnbench.sweep "$OUT_DIR/${PFX}probs-int.json"

    echo "Thresholds: eot=$THR_EOT  int=$THR_INT"

    echo "=== test: $RUN_NAME ==="
    # shellcheck disable=SC2086
    python -m baselines.mimi_endpointer.predict \
        $CKP_ARGS \
        --dataset "$TEST_DATASET" \
        --out "$OUT_DIR/${PFX}predictions-test.json" \
        --threshold-eot "$THR_EOT" \
        --threshold-int "$THR_INT"
    ;;

# ── probs-only + turnbench.sweep tables ──────────────────────────────────────────
--sweep)
    echo "=== probs-only: $RUN_NAME ==="
    # shellcheck disable=SC2086
    python -m baselines.mimi_endpointer.predict \
        $CKP_ARGS \
        --probs-only

    echo "=== turnbench.sweep: EOT ($RUN_NAME) ==="
    python -m turnbench.sweep "$OUT_DIR/${PFX}probs-eot.json"
    echo "=== turnbench.sweep: INT ($RUN_NAME) ==="
    python -m turnbench.sweep "$OUT_DIR/${PFX}probs-int.json"
    ;;

# ── probs only ───────────────────────────────────────────────────────────────
--probs)
    # shellcheck disable=SC2086
    python -m baselines.mimi_endpointer.predict \
        $CKP_ARGS \
        --probs-only
    ;;

# ── default: dev then test + turnbench.check ──────────────────────────────────────
"")
    _run_dev

    FALLBACK_EOT=$(python -c "from baselines.mimi_endpointer.predict import CHECKPOINT_DEFAULTS; print(CHECKPOINT_DEFAULTS.get('$RUN_NAME', CHECKPOINT_DEFAULTS['pretrained'])[0])")
    FALLBACK_INT=$(python -c "from baselines.mimi_endpointer.predict import CHECKPOINT_DEFAULTS; print(CHECKPOINT_DEFAULTS.get('$RUN_NAME', CHECKPOINT_DEFAULTS['pretrained'])[1])")
    THR_EOT=$(_pick_threshold "$OUT_DIR/${PFX}probs-eot.json" "$FALLBACK_EOT")
    THR_INT=$(_pick_threshold "$OUT_DIR/${PFX}probs-int.json" "$FALLBACK_INT")
    echo "Thresholds: eot=$THR_EOT  int=$THR_INT"

    echo "=== test: $RUN_NAME ==="
    # shellcheck disable=SC2086
    python -m baselines.mimi_endpointer.predict \
        $CKP_ARGS \
        --dataset "$TEST_DATASET" \
        --out "$OUT_DIR/${PFX}predictions-test.json" \
        --threshold-eot "$THR_EOT" \
        --threshold-int "$THR_INT"

    echo "=== turnbench.check ==="
    python -m turnbench.check "$OUT_DIR"
    ;;

*)
    echo "Unknown mode: $MODE" >&2
    echo "Usage: $0 [--dev|--sweep|--probs] [--oto|--swbd|--swbd-oto]" >&2
    exit 1
    ;;
esac
