#!/usr/bin/env bash
# Batch runner for direct_gemini_client.py over the dev.txt split.
# Streams each speaker file into Gemini Live (no LiveKit), N at a time.
# Resume-safe: a `.done` marker is written only when the client confirms a
# COMPLETE send (exit 0); tasks with a marker are skipped. A bare
# output.flac without `.done` is treated as partial and re-recorded
# (bless pre-marker recordings you trust with `touch <dir>/.done`).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"

DATA_DIR="${DATA_DIR:?set DATA_DIR to the Mundo delivery root (numbered folders with speaker audio)}"
SPLIT_FILE="${SPLIT_FILE:-$REPO/turnbench/splits/dev.txt}"
OUT_DIR="${OUT_DIR:-$REPO/baselines/gemini/sample_runs}"
MAX_PAR="${MAX_PAR:-10}"
PY="${PY:-python}"

WORK="$OUT_DIR/.batch"
mkdir -p "$WORK"

echo "[batch] data=$DATA_DIR"
echo "[batch] split=$SPLIT_FILE"
echo "[batch] out=$OUT_DIR  max_parallel=$MAX_PAR"

# Build the task queue: (task_id, speaker_index)
QUEUE="$WORK/queue.txt"
: > "$QUEUE"
total=0
skipped=0
while read -r tid; do
    [[ -z "$tid" || "$tid" == \#* ]] && continue
    tdir="$DATA_DIR/$tid"
    [[ -d "$tdir" ]] || { echo "[batch] missing task dir: $tdir"; continue; }
    for k in 1 2; do
        infile="$tdir/speaker_${k}_audio.wav"
        donefile="$OUT_DIR/$tid/speaker_$k/.done"
        [[ -f "$infile" ]] || continue
        total=$((total + 1))
        if [[ -f "$donefile" ]]; then
            skipped=$((skipped + 1))
            continue
        fi
        printf '%s\t%s\n' "$tid" "$k" >> "$QUEUE"
    done
done < "$SPLIT_FILE"

remaining=$(wc -l < "$QUEUE")
echo "[batch] tasks: $total total, $skipped already done, $remaining to run"
[[ "$remaining" -eq 0 ]] && { echo "[batch] nothing to do"; exit 0; }

run_one() {
    local tid="$1" k="$2"
    local odir="$OUT_DIR/$tid/speaker_$k"
    local log="$WORK/${tid}_speaker_${k}.log"
    mkdir -p "$odir"
    echo "[batch] start $tid/speaker_$k"
    if "$PY" "$HERE/direct_gemini_client.py" \
            -i "$DATA_DIR/$tid/speaker_${k}_audio.wav" \
            -o "$odir/output.wav" >>"$log" 2>&1; then
        touch "$odir/.done"
        echo "[batch] done  $tid/speaker_$k"
    else
        echo "[batch] FAIL  $tid/speaker_$k — partial, will retry on rerun (see $log)"
    fi
}
export -f run_one
export OUT_DIR DATA_DIR WORK HERE PY

# xargs ‘-P N’ gives N concurrent workers; ‘-n 2’ feeds 2 args per child;
# the leading ‘_’ becomes $0 so $1/$2 are tid/k.
< "$QUEUE" tr '\t' ' ' | xargs -P "$MAX_PAR" -n 2 bash -c 'run_one "$1" "$2"' _

echo "[batch] all workers exited"
