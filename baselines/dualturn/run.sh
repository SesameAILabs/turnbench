#!/bin/bash

set -e

echo ""
echo "--------------------------------------------------"
echo "Current time: $(date)"
echo "hostname: $(hostname)"
start_time=$(date +%s)

# --- HuggingFace offline ---
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="/mnt/matylda4/udupa/hugging-face"
export WANDB_MODE="offline"

# --- GPU ---
export CUDA_VISIBLE_DEVICES=$(/mnt/matylda4/udupa/exps/endpointing/smart-endpointing/sge_utils/free-gpus.sh 1) || {
    echo "Could not obtain a free GPU."
    exit 1
}
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

# --- Paths ---
REPO="/mnt/matylda4/udupa/exps/endpointing/sesame_eval/tt-benchmark"

# --- Run ---
cd "$REPO"
python3 baselines/dualturn/predict.py --inspect /mnt/matylda4/udupa/exps/endpointing/sesame_eval/data/105

echo ""
echo "Job finished at: $(date)"
end_time=$(date +%s)
echo "Time taken: $(echo "scale=2; ($end_time - $start_time) / 60" | bc) minutes"
echo "--------------------------------------------------"
