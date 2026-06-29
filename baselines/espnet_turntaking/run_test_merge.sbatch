#!/bin/bash
#SBATCH --job-name=espnet_tt_test_merge
#SBATCH --account=bbjs-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Emit the merged predictions-test.json from the now-complete shared cache.
# All 116 convs are cached by the shard array, so predict.py never loads the
# model here -> finishes in well under a minute (the 1-GPU alloc is just to
# satisfy the GPU-only partition; it backfills instantly). Submit with a
# dependency on the shard array:
#   sbatch --dependency=afterok:<array_jobid> run_test_merge.sbatch
set -euo pipefail

REPO=/work/nvme/bbjs/chuang14/tt-benchmark
DATASET=mundo-ai/turn-benchmark-test
CACHE=$REPO/predictions/espnet_turntaking_test/cache
OUT=$REPO/baselines/espnet_turntaking/predictions-test.json

source /u/chuang14/miniconda3/etc/profile.d/conda.sh
conda activate espnet-owsm
cd "$REPO"

python -m baselines.espnet_turntaking.predict \
    --dataset "$DATASET" --cache-dir "$CACHE" --out "$OUT"
python -c "import json; d=json.load(open('$OUT')); print('predictions:', len(d['predictions']))"
echo "wrote $OUT"
