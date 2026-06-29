#!/bin/bash
#SBATCH --job-name=tt_pc_merge
#SBATCH --account=bbjs-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=0:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Emit predictions-test.json from the cached per-channel probs (no model run).
# This is a separate job because resolve_dataset() loads the whole test split
# into RAM (~13 GB Arrow table); that overflows the login node, so the "final
# --out pass" must run where >60 GB of host memory is available. The GPU is
# requested only for scheduling (backfill-friendly); inference is fully cached.
#   sbatch baselines/espnet_turntaking_perchannel/merge_test_preds.sbatch
set -euo pipefail

REPO=/work/nvme/bbjs/chuang14/tt-benchmark
export ESPNET_TT_EXP=/work/nvme/bbjs/chuang14/espnet-slu/egs2/swbd/slu1/exp/asr_train_asr_whisper_turn_taking_raw_en_word
source /u/chuang14/miniconda3/etc/profile.d/conda.sh
conda activate espnet-owsm
cd "$REPO"

python -m baselines.espnet_turntaking_perchannel.predict \
    --dataset mundo-ai/turn-benchmark-test \
    --out baselines/espnet_turntaking_perchannel/predictions-test.json
echo "merge done"
