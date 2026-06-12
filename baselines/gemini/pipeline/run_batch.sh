#!/usr/bin/env bash
set -euo pipefail

# ── Parallel batch inference orchestrator (Gemini Live) ───────────────────
# Runs full-duplex inference over the dev_release_flac dataset:
#
#   DATA_DIR/<folder>/speaker_1_audio.flac   ─┐  two separate inferences
#   DATA_DIR/<folder>/speaker_2_audio.flac   ─┘  per dataset folder
#
# The output mirrors the dataset hierarchy:
#
#   OUTPUT_DIR/<folder>/speaker_1/{speaker_1_audio.flac, output.wav, output.flac}
#   OUTPUT_DIR/<folder>/speaker_2/{speaker_2_audio.flac, output.wav, output.flac}
#
# Usage:
#   ./run_batch.sh --provider gemini-2.5  [--max-parallel 10] [--subset-size 10]
#   ./run_batch.sh --provider gemini-3.1  [--max-parallel 10] [--subset-size 10]
#
# The script automatically:
#   1. Discovers all speaker_{1,2}_audio.flac tasks in DATA_DIR
#   2. Drops tasks whose output folder already has output.flac (skip rule)
#   3. Splits the *remaining* tasks into subsets (≤SUBSET_SIZE each)
#   4. Starts the LiveKit agent (unless --no-agent)
#   5. Launches parallel worker processes, each using a unique room name
#   6. Waits for all workers to finish and reports aggregated results

# ── Configuration ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/bathrooms/kcire/sesame/data/dev_release_flac"
OUTPUT_BASE="/bathrooms/kcire/sesame/results/audio_output/gemini"
OUTPUT_DIR=""
SILENCE_SEC=0
PROVIDER=""
MAX_PARALLEL=10
SUBSET_SIZE=10
START_AGENT=true
SYSTEM_PROMPT_FILE="$SCRIPT_DIR/system_prompt.txt"

# ── Parse arguments ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)       DATA_DIR="$2";       shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2";     shift 2 ;;
        --output-base)    OUTPUT_BASE="$2";    shift 2 ;;
        --silence)        SILENCE_SEC="$2";    shift 2 ;;
        --provider)       PROVIDER="$2";       shift 2 ;;
        --max-parallel)   MAX_PARALLEL="$2";   shift 2 ;;
        --subset-size)    SUBSET_SIZE="$2";    shift 2 ;;
        --system-prompt)  SYSTEM_PROMPT_FILE="$2"; shift 2 ;;
        --no-agent)       START_AGENT=false;   shift ;;
        -h|--help)
            echo "Usage: $0 --provider NAME [options]"
            echo ""
            echo "Required:"
            echo "  --provider NAME      Model provider (gemini-2.5 | gemini-3.1)"
            echo ""
            echo "Optional:"
            echo "  --data-dir DIR       Input dataset directory (default: $DATA_DIR)"
            echo "  --output-dir DIR     Output directory (auto: OUTPUT_BASE/<provider>)"
            echo "  --output-base DIR    Base for auto output dir (default: $OUTPUT_BASE)"
            echo "  --silence SEC        Trailing silence appended to input (default: 0)"
            echo "  --max-parallel N     Max concurrent workers (default: 10)"
            echo "  --subset-size N      Max tasks per worker (default: 10)"
            echo "  --system-prompt PATH System prompt .txt (lk_agent: LK_SYSTEM_PROMPT_FILE)"
            echo "  --no-agent           Don't start lk_agent.py (use if agent is already running)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$PROVIDER" ]]; then
    echo "ERROR: --provider is required (gemini-2.5 | gemini-3.1)"
    exit 1
fi
if [[ "$PROVIDER" != "gemini-2.5" && "$PROVIDER" != "gemini-3.1" ]]; then
    echo "ERROR: unknown provider '$PROVIDER' (supported: gemini-2.5, gemini-3.1)"
    exit 1
fi

if [[ ! -f "$SYSTEM_PROMPT_FILE" ]]; then
    echo "ERROR: system prompt file not found: $SYSTEM_PROMPT_FILE"
    exit 1
fi
PROMPT_SRC="$(realpath -e "$SYSTEM_PROMPT_FILE")"
export LK_SYSTEM_PROMPT_FILE="$PROMPT_SRC"
export LK_PROVIDER="$PROVIDER"

ROOM_BASE="${PROVIDER}-batch"
[[ -z "$OUTPUT_DIR" ]] && OUTPUT_DIR="${OUTPUT_BASE}/${PROVIDER}"

# ── Collect all tasks (one per speaker audio file) ────────────────────────
echo "Scanning for speaker audio in $DATA_DIR …"
TASKS=()
for dir in "$DATA_DIR"/*/; do
    folder="$(basename "$dir")"
    for speaker in speaker_1 speaker_2; do
        if [[ -f "$dir/${speaker}_audio.flac" ]]; then
            TASKS+=("$folder/$speaker")
        fi
    done
done
# Sort for deterministic ordering
IFS=$'\n' TASKS=($(sort <<< "${TASKS[*]}")); unset IFS
TOTAL_DISCOVERED=${#TASKS[@]}

if [[ $TOTAL_DISCOVERED -eq 0 ]]; then
    echo "No speaker_*_audio.flac files found in $DATA_DIR"
    exit 1
fi

# ── Pre-skip: align with run_worker.sh (skip = output.flac exists) ────────
PENDING=()
for task in "${TASKS[@]}"; do
    if [[ -f "$OUTPUT_DIR/$task/output.flac" ]]; then
        continue
    fi
    PENDING+=("$task")
done
TASKS=("${PENDING[@]}")
unset PENDING
TOTAL=${#TASKS[@]}
SKIPPED_PRE=$((TOTAL_DISCOVERED - TOTAL))

if [[ $TOTAL -eq 0 ]]; then
    echo "All $TOTAL_DISCOVERED task(s) already have $OUTPUT_DIR/<task>/output.flac — nothing to do."
    exit 0
fi

echo "Tasks discovered: $TOTAL_DISCOVERED  (already complete, skipped: $SKIPPED_PRE  →  pending: $TOTAL)"
echo ""

# ── Split into subsets and write manifest files ───────────────────────────
WORK_DIR="$OUTPUT_DIR/.batch_work"
mkdir -p "$WORK_DIR"

# Record the system prompt used for this run
cp "$PROMPT_SRC" "$OUTPUT_DIR/system_prompt.txt"
echo "System prompt saved to $OUTPUT_DIR/system_prompt.txt (source: $PROMPT_SRC)"

NUM_SUBSETS=$(( (TOTAL + SUBSET_SIZE - 1) / SUBSET_SIZE ))

echo "============================================"
echo "  Parallel batch inference (Gemini Live)"
echo "============================================"
echo "  Provider:      $PROVIDER"
echo "  Discovered:    $TOTAL_DISCOVERED"
echo "  Pre-skipped:   $SKIPPED_PRE (output.flac already present)"
echo "  Pending:       $TOTAL"
echo "  Subset size:   $SUBSET_SIZE"
echo "  Num subsets:   $NUM_SUBSETS"
echo "  Max parallel:  $MAX_PARALLEL"
echo "  Data dir:      $DATA_DIR"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Silence pad:   ${SILENCE_SEC}s"
echo "  Room base:     $ROOM_BASE"
echo "  Start agent:   $START_AGENT"
echo "  System prompt: $PROMPT_SRC"
echo "============================================"

for (( i=0; i<NUM_SUBSETS; i++ )); do
    start=$(( i * SUBSET_SIZE ))
    manifest="$WORK_DIR/manifest_${i}.txt"
    : > "$manifest"
    for (( j=start; j<start+SUBSET_SIZE && j<TOTAL; j++ )); do
        echo "${TASKS[$j]}" >> "$manifest"
    done
    count=$(wc -l < "$manifest")
    echo "  Subset $i: $count tasks → room ${ROOM_BASE}-${i}"
done

# ── Cleanup handler ───────────────────────────────────────────────────────
AGENT_PID=""
declare -a WORKER_PIDS=()

cleanup() {
    echo ""
    echo "Cleaning up …"
    for pid in "${WORKER_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    if [[ -n "$AGENT_PID" ]]; then
        kill "$AGENT_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Start the agent ───────────────────────────────────────────────────────
if $START_AGENT; then
    echo ""
    echo "Starting lk_agent.py dev …"
    cd "$SCRIPT_DIR"
    python lk_agent.py dev > "$WORK_DIR/agent.log" 2>&1 &
    AGENT_PID=$!
    echo "  Agent PID: $AGENT_PID  (log: $WORK_DIR/agent.log)"
    echo "  Waiting 10s for agent to register with LiveKit Cloud …"
    sleep 10

    if ! kill -0 "$AGENT_PID" 2>/dev/null; then
        echo "ERROR: Agent process died. Check $WORK_DIR/agent.log"
        tail -20 "$WORK_DIR/agent.log" || true
        exit 1
    fi
    echo "  Agent is running."
else
    echo ""
    echo "Skipping agent start (--no-agent). The agent must load the same prompt; e.g."
    echo "  export LK_SYSTEM_PROMPT_FILE=$PROMPT_SRC"
    echo "  python $SCRIPT_DIR/lk_agent.py dev"
fi

# ── Launch parallel workers ───────────────────────────────────────────────
echo ""
echo "Launching workers …"

count_alive() {
    local alive=0
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            alive=$((alive + 1))
        fi
    done
    echo "$alive"
}

for (( i=0; i<NUM_SUBSETS; i++ )); do
    # Throttle: wait until a slot is free
    while (( $(count_alive) >= MAX_PARALLEL )); do
        sleep 5
    done

    room="${ROOM_BASE}-${i}"
    manifest="$WORK_DIR/manifest_${i}.txt"
    log_file="$WORK_DIR/worker_${i}.log"

    echo "  → Worker $i  (room: $room, log: $log_file)"
    bash "$SCRIPT_DIR/run_worker.sh" \
        "$manifest" "$DATA_DIR" "$OUTPUT_DIR" "$room" \
        "$SILENCE_SEC" "$SCRIPT_DIR" "$i" \
        > "$log_file" 2>&1 &
    WORKER_PIDS+=($!)

    # Stagger launches slightly to avoid thundering herd on LiveKit Cloud
    sleep 3
done

# ── Wait for all workers ──────────────────────────────────────────────────
echo ""
echo "All $NUM_SUBSETS workers launched. Waiting for completion …"
echo "(Monitor progress with:  tail -f $WORK_DIR/worker_*.log)"
echo ""

for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

# ── Aggregate results ─────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Results"
echo "============================================"

TOTAL_ALL=0
SKIPPED_ALL=0
FAILED_ALL=0

for (( i=0; i<NUM_SUBSETS; i++ )); do
    status_file="$WORK_DIR/manifest_${i}.status"
    if [[ -f "$status_file" ]]; then
        read -r t s f < "$status_file"
        TOTAL_ALL=$((TOTAL_ALL + t))
        SKIPPED_ALL=$((SKIPPED_ALL + s))
        FAILED_ALL=$((FAILED_ALL + f))
        echo "  Worker $i: $t total, $s skipped, $f failed"
    else
        echo "  Worker $i: NO STATUS — check $WORK_DIR/worker_${i}.log"
    fi
done

PROCESSED=$((TOTAL_ALL - SKIPPED_ALL - FAILED_ALL))
echo "--------------------------------------------"
echo "  TOTAL:     $TOTAL_ALL"
echo "  PROCESSED: $PROCESSED"
echo "  SKIPPED:   $SKIPPED_ALL"
echo "  FAILED:    $FAILED_ALL"
echo "============================================"
echo ""
echo "Output:       $OUTPUT_DIR"
echo "Worker logs:  $WORK_DIR/worker_*.log"
echo "Agent log:    $WORK_DIR/agent.log"
echo "============================================"
