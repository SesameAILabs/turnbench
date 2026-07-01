#!/usr/bin/env bash
# baselines/espnet_turntaking_perchannel/run.sh — reproduce the submission from a fresh clone.
#
# Prereqs (see requirements.txt): the espnet env with espnet2 importable, and
# ESPNET_TT_EXP pointing at the HF checkpoint's exp dir (config.yaml +
# valid.loss.ave.pth from espnet/Turn_taking_prediction_SWBD).
#
# Usage:
#   PYTHON=/path/to/espnet-venv/bin/python \
#   ESPNET_TT_EXP=/abs/.../asr_train_asr_whisper_turn_taking_raw_en_word \
#   bash baselines/espnet_turntaking_perchannel/run.sh {dev|test} [shards_per_gpu]
#
#   dev : shard the model over GPUs -> per-frame cache -> probs-{eot,int}.json ->
#         eval.sweep operating point -> predictions-dev.json
#   test: build the test cache -> commit predictions-test.json at the *dev* op
#
# TF32 is on by default (TT_TF32=1): ~2.3x on H100 tensor cores, ~5e-3 prob delta.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

SPLIT="${1:-dev}"; SPG="${2:-2}"
PY="${PYTHON:-python}"
: "${ESPNET_TT_EXP:?set ESPNET_TT_EXP to the checkpoint exp dir}"
export TT_TF32="${TT_TF32:-1}"
HERE="baselines/espnet_turntaking_perchannel"
MOD="baselines.espnet_turntaking_perchannel"

case "$SPLIT" in
  dev)  DATASET="mundo-ai/turn-benchmark-dev";  CACHE="predictions/espnet_turntaking_perchannel/cache" ;;
  test) DATASET="mundo-ai/turn-benchmark-test"; CACHE="predictions/espnet_turntaking_perchannel_test/cache" ;;
  *) echo "usage: run.sh {dev|test} [shards_per_gpu]" >&2; exit 1 ;;
esac

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "${NGPU:-0}" -lt 1 ] && NGPU=1
NSHARD=$(( NGPU * SPG ))

# 1) per-frame cache, sharded across GPUs (cache-only; skips already-cached convs)
echo "[espnet_turntaking_perchannel:$SPLIT] $NSHARD shards over $NGPU GPU(s), TF32=$TT_TF32"
pids=()
for j in $(seq 0 $(( NSHARD - 1 ))); do
    CUDA_VISIBLE_DEVICES=$(( j % NGPU )) $PY -m "${MOD}.predict" \
        --dataset "$DATASET" --cache-dir "$CACHE" --shard "$j" --num-shards "$NSHARD" &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

# operating point from dev probs (highest recall at fp_rate <= 0.1; eval.sweep quantile candidates)
_theta() {  # task -> theta on the committed dev probs
    $PY - "$1" <<'PY'
import sys
from pathlib import Path
from eval.data import DEV_DATASET, resolve_dataset
from eval.sweep import load_probs, sweep, operating_point
task = sys.argv[1]
probs = load_probs(Path(f"baselines/espnet_turntaking_perchannel/probs-{task}.json"))
op = operating_point(sweep(probs, resolve_dataset(source=DEV_DATASET, skip_audio=True)))
print(repr(op.theta) if op else "")  # full precision: quantile ops can be tiny (e.g. 7e-4)
PY
}

if [ "$SPLIT" = "dev" ]; then
    $PY -m "${MOD}.submit" probs --task eot --cache-dir "$CACHE" --out "$HERE/probs-eot.json"
    $PY -m "${MOD}.submit" probs --task int --cache-dir "$CACHE" --out "$HERE/probs-int.json"
fi

TE=$(_theta eot); TI=$(_theta int)
[ -n "$TE" ] && [ -n "$TI" ] || { echo "no operating point at fp<=0.1 for a task (eot=$TE int=$TI)" >&2; exit 1; }
echo "[espnet_turntaking_perchannel] operating point: theta_eot=$TE theta_int=$TI"

$PY -m "${MOD}.submit" predictions --split "$SPLIT" --cache-dir "$CACHE" \
    --theta-eot "$TE" --theta-int "$TI" --out "$HERE/predictions-${SPLIT}.json"

$PY -m eval.check "$HERE" || true
