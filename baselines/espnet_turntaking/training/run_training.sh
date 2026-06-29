#!/usr/bin/env bash
# Train the turn-taking model with ESPnet, using this folder's config + token
# list + prepared data dir. Portable driver: everything is parameterized by env
# vars (no hardcoded cluster paths / conda env / SLURM account).
#
# Prerequisites (see README):
#   * ESPnet installed (pinned commit) and its env activated, so
#     `python -m espnet2.bin.slu_train` works.
#   * An ESPnet egs2 SLU recipe dir with the standard symlinks (slu.sh, utils,
#     steps, ...). egs2/swbd/slu1 works; or `egs2/TEMPLATE/slu1` scaffolding.
#   * Data dirs already built by prepare_turn_data.py and copied/symlinked into
#     ${RECIPE_DIR}/data/${TRAIN_SET} and .../data/${VALID_SET}.
#   * The token list copied to
#     ${RECIPE_DIR}/data/${LANG}_token_list/word/tokens.txt
#     (so stage 5 is skipped and the class->id mapping is fixed).
#
# Env vars (override as needed):
#   RECIPE_DIR  : path to the ESPnet egs2 SLU recipe (required)
#   THIS_DIR    : this training/ folder (auto-detected)
#   TRAIN_SET   : data dir name for training   (default: turn_train)
#   VALID_SET   : data dir name for validation (default: turn_valid)
#   LANG        : recipe lang tag              (default: en)
#   NGPU        : GPUs for training            (default: 2)
#   NJ          : parallel jobs for stages 3/10 (default: 8)
#   EXP         : experiment dir               (default: exp/tt_turn)
#   STATS       : stats dir                    (default: exp/slu_stats_turn)
#   STAGE/STOP  : recipe stages                (default: 3 / 11)
#
# Usage:
#   RECIPE_DIR=/path/to/espnet/egs2/swbd/slu1 NGPU=2 \
#     bash run_training.sh
set -euo pipefail

THIS_DIR="${THIS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
: "${RECIPE_DIR:?set RECIPE_DIR to your ESPnet egs2 SLU recipe directory}"
TRAIN_SET="${TRAIN_SET:-turn_train}"
VALID_SET="${VALID_SET:-turn_valid}"
LANG="${LANG:-en}"
NGPU="${NGPU:-2}"
NJ="${NJ:-8}"
EXP="${EXP:-exp/tt_turn}"
STATS="${STATS:-exp/slu_stats_turn}"
STAGE="${STAGE:-3}"
STOP="${STOP:-11}"

# Install our config + token list into the recipe (idempotent).
mkdir -p "${RECIPE_DIR}/conf" "${RECIPE_DIR}/data/${LANG}_token_list/word"
cp "${THIS_DIR}/conf/train_turn_taking.yaml" "${RECIPE_DIR}/conf/train_turn_taking.yaml"
cp "${THIS_DIR}/tokens.txt" "${RECIPE_DIR}/data/${LANG}_token_list/word/tokens.txt"

cd "${RECIPE_DIR}"

# slu.sh (symlinked from egs2/TEMPLATE/slu1) is the training engine.
# stages: 3 format/dump -> 4 length filter -> (5 token list: SKIPPED, we supply
# tokens.txt) -> 10 collect-stats -> 11 train. use_lm false skips 6-9.
# --resume true (ESPnet default) lets a timed-out job continue from the last
# checkpoint: just re-run this script.
./slu.sh \
    --use_lm false \
    --lang "${LANG}" \
    --ngpu "${NGPU}" \
    --nj "${NJ}" \
    --token_type word \
    --feats_type raw \
    --audio_format "flac.ark" \
    --feats_normalize utterance_mvn \
    --max_wav_duration 40 \
    --no_asr_eval true \
    --slu_config conf/train_turn_taking.yaml \
    --slu_exp "${EXP}" \
    --slu_stats_dir "${STATS}" \
    --train_set "${TRAIN_SET}" \
    --valid_set "${VALID_SET}" \
    --test_sets "${VALID_SET}" \
    --stage "${STAGE}" --stop_stage "${STOP}"

echo "done -> checkpoint: ${RECIPE_DIR}/${EXP}/valid.loss.ave.pth"
echo "point ESPNET_TT_EXP at ${RECIPE_DIR}/${EXP} to use it with the espnet_turntaking baseline."
