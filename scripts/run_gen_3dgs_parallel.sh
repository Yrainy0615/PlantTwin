#!/bin/bash
# Launch 8 parallel gen_3dgs workers across 8 GPUs to generate 200 plant Gaussians.
# Each worker gets CUDA_VISIBLE_DEVICES=N and processes 25 plants (200/8).
#
# Usage:
#   bash scripts/run_gen_3dgs_parallel.sh [--dry-run]
#
# Logs: logs/gen_3dgs_gpu{0..7}.log

set -e

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

NUM_GPUS=8
TOTAL_PLANTS=200
PROMPT_FILE="configs/plant_prompts.txt"
OUTPUT_DIR="data/plants_3dgs"
LOG_DIR="logs"
CONDA_ENV="planttwin"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] Showing commands without executing"
fi

echo "=== gen_3dgs parallel launch ==="
echo "GPUs:          $NUM_GPUS"
echo "Total plants:  $TOTAL_PLANTS ($(python3 -c "print(($TOTAL_PLANTS + $NUM_GPUS - 1) // $NUM_GPUS)") per GPU)"
echo "Output:        $OUTPUT_DIR"
echo "Logs:          $LOG_DIR/gen_3dgs_gpu{0..7}.log"
echo ""

PIDS=()
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    CMD="conda run -n $CONDA_ENV --no-capture-output bash -c \
'CUDA_VISIBLE_DEVICES=$GPU_ID python data/generation/gen_3dgs.py \
  --prompt_file $PROMPT_FILE \
  --output_dir $OUTPUT_DIR \
  --total_plants $TOTAL_PLANTS \
  --shard_id $GPU_ID \
  --num_shards $NUM_GPUS'"

    LOG="$LOG_DIR/gen_3dgs_gpu${GPU_ID}.log"

    if $DRY_RUN; then
        echo "  GPU $GPU_ID: $CMD > $LOG 2>&1 &"
    else
        echo "  Launching GPU $GPU_ID → $LOG"
        eval "nohup $CMD > $LOG 2>&1 &"
        PIDS+=($!)
    fi
done

if $DRY_RUN; then
    echo ""
    echo "Remove --dry-run to execute."
    exit 0
fi

echo ""
echo "All $NUM_GPUS workers launched (PIDs: ${PIDS[*]})"
echo "Monitor with: tail -f logs/gen_3dgs_gpu0.log"
echo "Watch all:    tail -f logs/gen_3dgs_gpu*.log"
echo ""
echo "Waiting for all workers to finish..."
FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        echo "  WARNING: PID $PID exited with error"
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -eq 0 ]; then
    ACTUAL=$(find "$OUTPUT_DIR" -name "gaussian.ply" | wc -l)
    echo ""
    echo "=== gen_3dgs complete: $ACTUAL plants generated in $OUTPUT_DIR ==="
else
    echo ""
    echo "=== gen_3dgs finished with $FAILED failed workers — check logs ==="
    exit 1
fi
