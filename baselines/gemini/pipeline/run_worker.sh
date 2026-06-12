#!/usr/bin/env bash
set -euo pipefail

# ── Per-subset worker: processes tasks listed in a manifest sequentially ──
# Launched by run_batch.sh — not intended to be run directly.
#
# Each manifest line is a task id of the form "<folder>/<speaker>",
# e.g. "105/speaker_1". For each task the worker:
#   1. Streams DATA_DIR/<folder>/<speaker>_audio.flac through the agent
#   2. Saves OUTPUT_DIR/<folder>/<speaker>/output.wav + output.flac
#   3. Copies the input flac into the same output folder
#
# Args: <manifest> <data_dir> <output_dir> <room_name> <silence_sec> <script_dir> <worker_id>

MANIFEST="$1"
DATA_DIR="$2"
OUTPUT_DIR="$3"
ROOM_NAME="$4"
SILENCE_SEC="$5"
SCRIPT_DIR="$6"
WORKER_ID="$7"

mapfile -t TASKS < "$MANIFEST"
TOTAL=${#TASKS[@]}

DONE=0
SKIPPED=0
FAILED=0

echo "[W$WORKER_ID] Starting: $TOTAL tasks, room=$ROOM_NAME"

for task in "${TASKS[@]}"; do
    folder="${task%%/*}"     # e.g. 105
    speaker="${task##*/}"    # e.g. speaker_1
    input_file="$DATA_DIR/$folder/${speaker}_audio.flac"
    out_dir="$OUTPUT_DIR/$folder/$speaker"
    output_wav="$out_dir/output.wav"
    output_flac="$out_dir/output.flac"
    DONE=$((DONE + 1))

    if [[ -f "$output_flac" ]]; then
        SKIPPED=$((SKIPPED + 1))
        echo "[W$WORKER_ID $DONE/$TOTAL] SKIP $task (completed)"
        continue
    fi

    if [[ ! -f "$input_file" ]]; then
        FAILED=$((FAILED + 1))
        echo "[W$WORKER_ID $DONE/$TOTAL] FAIL $task (missing input: $input_file)"
        continue
    fi

    mkdir -p "$out_dir"
    rm -f "$output_wav" "$output_flac"

    # ── Run inference ─────────────────────────────────────────────────
    # Use a unique room per task so LiveKit dispatches a fresh agent session
    task_room="${ROOM_NAME}-t${DONE}"
    echo "[W$WORKER_ID $DONE/$TOTAL] RUN  $task  (room: $task_room)"
    if ! python "$SCRIPT_DIR/lk_audio_client.py" \
        -i "$input_file" \
        -o "$output_wav" \
        --silence "$SILENCE_SEC" \
        --room "$task_room"; then
        FAILED=$((FAILED + 1))
        echo "[W$WORKER_ID $DONE/$TOTAL] FAIL $task"
        continue
    fi

    # ── Copy input alongside the results ──────────────────────────────
    cp -f "$input_file" "$out_dir/${speaker}_audio.flac"

    echo "[W$WORKER_ID $DONE/$TOTAL] OK   $task"
done

echo "[W$WORKER_ID] Finished: $TOTAL total, $SKIPPED skipped, $FAILED failed"
echo "$TOTAL $SKIPPED $FAILED" > "${MANIFEST%.txt}.status"
