#!/usr/bin/env bash
# baselines/kyutai_semantic_vad/run.sh — Kyutai Semantic VAD prediction entry point
#
# Usage:
#   bash baselines/kyutai_semantic_vad/run.sh --dev            # dev: inference → probs + predictions + sweep tables
#   bash baselines/kyutai_semantic_vad/run.sh --test           # test: sweep existing probs → pick threshold → predictions
#   bash baselines/kyutai_semantic_vad/run.sh --dev-from-probs # dev predictions from existing probs (no inference)
#   bash baselines/kyutai_semantic_vad/run.sh                  # full: --dev then --test + turnbench.check
#   bash baselines/kyutai_semantic_vad/run.sh --probs          # probs only (no predictions)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODE="${1:-}"
TEST_DATASET=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('mundo-ai/turn-benchmark-test', repo_type='dataset', local_files_only=True))")/data
OUT_DIR="baselines/kyutai_semantic_vad"

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
    echo "=== dev: pretrained ==="
    python -m baselines.kyutai_semantic_vad.predict \
        --out "$OUT_DIR/predictions-dev.json"

    echo "=== turnbench.sweep: EOT ==="
    python -m turnbench.sweep "$OUT_DIR/probs-eot.json"
    echo "=== turnbench.sweep: INT ==="
    python -m turnbench.sweep "$OUT_DIR/probs-int.json"
}

case "$MODE" in

# ── dev only ─────────────────────────────────────────────────────────────────
--dev)
    _run_dev
    ;;

# ── dev predictions from existing probs ──────────────────────────────────────
--dev-from-probs)
    [[ -f "$OUT_DIR/probs-eot.json" ]] || { echo "Missing $OUT_DIR/probs-eot.json — run --dev first" >&2; exit 1; }
    [[ -f "$OUT_DIR/probs-int.json" ]] || { echo "Missing $OUT_DIR/probs-int.json — run --dev first" >&2; exit 1; }
    echo "=== dev predictions from probs ==="
    python -m baselines.kyutai_semantic_vad.predict \
        --from-probs \
        --out "$OUT_DIR/predictions-dev.json"
    echo "=== turnbench.sweep: EOT ==="
    python -m turnbench.sweep "$OUT_DIR/probs-eot.json"
    echo "=== turnbench.sweep: INT ==="
    python -m turnbench.sweep "$OUT_DIR/probs-int.json"
    ;;

# ── test only: sweep existing probs → pick threshold → test predictions ──────
--test)
    [[ -f "$OUT_DIR/probs-eot.json" ]] || { echo "Missing $OUT_DIR/probs-eot.json — run --dev first" >&2; exit 1; }
    [[ -f "$OUT_DIR/probs-int.json" ]] || { echo "Missing $OUT_DIR/probs-int.json — run --dev first" >&2; exit 1; }

    FALLBACK_EOT=$(python -c "from baselines.kyutai_semantic_vad.predict import CHECKPOINT_DEFAULTS; print(CHECKPOINT_DEFAULTS['pretrained'][0])")
    FALLBACK_INT=$(python -c "from baselines.kyutai_semantic_vad.predict import CHECKPOINT_DEFAULTS; print(CHECKPOINT_DEFAULTS['pretrained'][1])")

    echo "=== turnbench.sweep: EOT ==="
    THR_EOT=$(_pick_threshold "$OUT_DIR/probs-eot.json" "$FALLBACK_EOT")
    python -m turnbench.sweep "$OUT_DIR/probs-eot.json"

    echo "=== turnbench.sweep: INT ==="
    THR_INT=$(_pick_threshold "$OUT_DIR/probs-int.json" "$FALLBACK_INT")
    python -m turnbench.sweep "$OUT_DIR/probs-int.json"

    echo "Thresholds: eot=$THR_EOT  int=$THR_INT"

    echo "=== test: pretrained ==="
    python -m baselines.kyutai_semantic_vad.predict \
        --dataset "$TEST_DATASET" \
        --out "$OUT_DIR/predictions-test.json" \
        --threshold-eot "$THR_EOT" \
        --threshold-int "$THR_INT"
    ;;

# ── probs only ───────────────────────────────────────────────────────────────
--probs)
    python -m baselines.kyutai_semantic_vad.predict --probs-only
    ;;

# ── default: dev then test + turnbench.check ──────────────────────────────────────
"")
    _run_dev

    FALLBACK_EOT=$(python -c "from baselines.kyutai_semantic_vad.predict import CHECKPOINT_DEFAULTS; print(CHECKPOINT_DEFAULTS['pretrained'][0])")
    FALLBACK_INT=$(python -c "from baselines.kyutai_semantic_vad.predict import CHECKPOINT_DEFAULTS; print(CHECKPOINT_DEFAULTS['pretrained'][1])")
    THR_EOT=$(_pick_threshold "$OUT_DIR/probs-eot.json" "$FALLBACK_EOT")
    THR_INT=$(_pick_threshold "$OUT_DIR/probs-int.json" "$FALLBACK_INT")
    echo "Thresholds: eot=$THR_EOT  int=$THR_INT"

    echo "=== test: pretrained ==="
    python -m baselines.kyutai_semantic_vad.predict \
        --dataset "$TEST_DATASET" \
        --out "$OUT_DIR/predictions-test.json" \
        --threshold-eot "$THR_EOT" \
        --threshold-int "$THR_INT"

    echo "=== turnbench.check ==="
    python -m turnbench.check "$OUT_DIR"
    ;;

*)
    echo "Unknown mode: $MODE" >&2
    echo "Usage: $0 [--dev|--test|--probs]" >&2
    exit 1
    ;;
esac
