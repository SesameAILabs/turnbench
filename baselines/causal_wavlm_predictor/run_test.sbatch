#!/bin/bash
#SBATCH --job-name=tt_cwp_test
#SBATCH --account=bbjs-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Three stages:
#   1. Extract test audio from HF dataset (needs ~60GB RAM)
#   2. Run chunked predictor inference (needs GPU)
#   3. Convert to submission JSON (needs eval deps)
#
# Usage:
#   sbatch baselines/causal_wavlm_predictor/run_test.sbatch [base|large]
set -euo pipefail

MODEL=${1:-large}  # "base" or "large"

REPO=/work/nvme/bbjs/ttao3/workspace/tt-benchmark
ESPNET_BASE=/work/nvme/bbjs/ttao3/workspace/espnet/egs2/universa_unite/turn_taking1
TTBENCH_PYTHON=/work/nvme/bbjs/ttao3/miniconda3/envs/ttbench/bin/python3
ESPNET_PYTHON=/work/nvme/bbjs/ttao3/.conda/envs/espnet39/bin/python3

if [ "$MODEL" = "large" ]; then
    MODEL_DIR=${ESPNET_BASE}/exp/tt_pred_large_turn_swbd_res
    UPSTREAM=wavlm_large
    SUFFIX=large
else
    MODEL_DIR=${ESPNET_BASE}/exp/tt_pred_base_turn_swbd_res
    UPSTREAM=wavlm_base_plus
    SUFFIX=base
fi

TEST_AUDIO_DIR=${ESPNET_BASE}/data_turnbench/test_hf
DECODE_DIR=${MODEL_DIR}/decode_test_hf_chunked
OUT_JSON=${REPO}/baselines/causal_wavlm_predictor/predictions-test-${SUFFIX}.json

# --- Stage 1: Extract test audio ---
if [ ! -f "${TEST_AUDIO_DIR}/wav.scp" ]; then
    echo "Stage 1: Extracting test audio from HF dataset..."
    cd "$REPO"
    $TTBENCH_PYTHON baselines/causal_wavlm_predictor/extract_test_audio.py \
        --dataset mundo-ai/turn-benchmark-test \
        --output-dir "$TEST_AUDIO_DIR"
else
    echo "Stage 1: Skipped (wav.scp already exists)"
fi

# --- Stage 2: Predictor inference ---
echo "Stage 2: Running chunked inference (model=$MODEL)..."
mkdir -p "$DECODE_DIR"
cd "$ESPNET_BASE"
export PYTHONPATH=/work/nvme/bbjs/ttao3/workspace/espnet:/work/nvme/bbjs/ttao3/universa/espnet:/work/nvme/bbjs/ttao3/RawNet/python:${ESPNET_BASE}/pyscripts:${PYTHONPATH:-}
$ESPNET_PYTHON ${ESPNET_BASE}/pyscripts/run_predictor_inference.py \
    --model-dir "$MODEL_DIR" \
    --wavscp "${TEST_AUDIO_DIR}/wav.scp" \
    --output "${DECODE_DIR}/text" \
    --upstream "$UPSTREAM" \
    --device cuda \
    --batch-size 1 \
    --chunk-seconds 30

# --- Stage 3: Convert to submission JSON ---
echo "Stage 3: Building submission JSON..."
cd "$REPO"
$TTBENCH_PYTHON ${ESPNET_BASE}/pyscripts/eval_turnbench.py \
    --hyp "${DECODE_DIR}/text" \
    --dataset mundo-ai/turn-benchmark-test \
    --output-dir "${DECODE_DIR}/eval"

cp "${DECODE_DIR}/eval/predictions.json" "$OUT_JSON"
echo "Done: $OUT_JSON"
