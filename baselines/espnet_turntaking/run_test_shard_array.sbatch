#!/bin/bash
#SBATCH --job-name=espnet_tt_test_shard
#SBATCH --account=bbjs-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#
# One array task per shard: requests a SINGLE GPU so it backfills into the many
# free GPUs on partially-allocated nodes (a full 4-GPU node was scarce and sat
# ~27h in the queue). Each task processes conversation_ids[i::4] of the test set
# into a SHARED per-conversation cache (distinct ids -> distinct files, no
# collision). The dependent merge job emits the JSON once all four finish.
#
#   sbatch baselines/espnet_turntaking/run_test_shard_array.sbatch
set -euo pipefail

REPO=/work/nvme/bbjs/chuang14/tt-benchmark
DATASET=mundo-ai/turn-benchmark-test
CACHE=$REPO/predictions/espnet_turntaking_test/cache
export ESPNET_TT_EXP=/work/nvme/bbjs/chuang14/espnet-slu/egs2/swbd/slu1/exp/asr_train_asr_whisper_turn_taking_raw_en_word

source /u/chuang14/miniconda3/etc/profile.d/conda.sh
conda activate espnet-owsm
cd "$REPO"
mkdir -p "$CACHE"

echo "shard ${SLURM_ARRAY_TASK_ID}/4 on $(hostname), GPU=${CUDA_VISIBLE_DEVICES:-?}"
python -m baselines.espnet_turntaking.predict \
    --dataset "$DATASET" --cache-dir "$CACHE" \
    --shard "${SLURM_ARRAY_TASK_ID}" --num-shards 4
echo "shard ${SLURM_ARRAY_TASK_ID} done"
