#!/bin/bash
#SBATCH --job-name=tt_pc_test
#SBATCH --account=bbjs-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#
# Cache per-channel model probs on the TEST set (116 convs x 2 channels),
# sharded across 8 single-GPU jobs (backfill-friendly: short 1-GPU jobs schedule
# into gaps). Each task caches conversation_ids[i::8]; a final --out pass emits
# predictions-test.json from the cache.
#   sbatch baselines/espnet_turntaking_perchannel/run_test_sharded.sbatch
set -euo pipefail

REPO=/work/nvme/bbjs/chuang14/tt-benchmark
export ESPNET_TT_EXP=/work/nvme/bbjs/chuang14/espnet-slu/egs2/swbd/slu1/exp/asr_train_asr_whisper_turn_taking_raw_en_word
source /u/chuang14/miniconda3/etc/profile.d/conda.sh
conda activate espnet-owsm
cd "$REPO"

echo "pc-test shard ${SLURM_ARRAY_TASK_ID}/8 on $(hostname)"
python -m baselines.espnet_turntaking_perchannel.predict \
    --dataset mundo-ai/turn-benchmark-test \
    --shard "${SLURM_ARRAY_TASK_ID}" --num-shards 8
echo "shard ${SLURM_ARRAY_TASK_ID} done"
