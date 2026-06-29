#!/bin/bash
#SBATCH --job-name=espnet_tt_test
#SBATCH --account=bbjs-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Produce predictions on mundo-ai/turn-benchmark-test by sharding the 116
# conversations across the node's 4 GH200 GPUs. Each shard runs predict.py in
# --num-shards 4 mode, populating a SHARED per-conversation cache (distinct
# conversation ids -> distinct cache files, no collision). A final
# --num-shards 1 pass emits the merged predictions-test.json from the cache.
#
# Submit from the repo root:
#   sbatch baselines/espnet_turntaking/run_test_sharded.sbatch
set -euo pipefail

REPO=/work/nvme/bbjs/chuang14/tt-benchmark
DATASET=mundo-ai/turn-benchmark-test
CACHE=$REPO/predictions/espnet_turntaking_test/cache
OUT=$REPO/baselines/espnet_turntaking/predictions-test.json
export ESPNET_TT_EXP=/work/nvme/bbjs/chuang14/espnet-slu/egs2/swbd/slu1/exp/asr_train_asr_whisper_turn_taking_raw_en_word

source /u/chuang14/miniconda3/etc/profile.d/conda.sh
conda activate espnet-owsm
cd "$REPO"
mkdir -p "$CACHE"

echo "== launching 4 shards on $(hostname) =="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

pids=()
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i \
  python -m baselines.espnet_turntaking.predict \
    --dataset "$DATASET" --cache-dir "$CACHE" \
    --shard "$i" --num-shards 4 \
    > "shard_${i}.log" 2>&1 &
  pids+=($!)
done

# Wait for all shards; fail loudly if any shard dies.
fail=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then fail=1; fi
done
if [ "$fail" -ne 0 ]; then
  echo "!! a shard failed; cache is partial. Re-running this job resumes (cached convs are skipped)."
  exit 1
fi

echo "== all shards done; emitting merged JSON =="
python -m baselines.espnet_turntaking.predict \
  --dataset "$DATASET" --cache-dir "$CACHE" --out "$OUT"

echo "== wrote $OUT =="
python -c "import json; d=json.load(open('$OUT')); print('predictions:', len(d['predictions']))"
