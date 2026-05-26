#!/bin/bash
# Launch 8 parallel gen_video workers across 8 GPUs to generate 1000 videos.
# Each worker processes 25 plants × 5 force types = 125 videos.
#
# Usage:
#   bash scripts/run_gen_video_parallel.sh [--dry-run]
#
# Requires gen_3dgs to have completed first (data/plants_3dgs must be populated).
# Logs: logs/gen_video_gpu{0..7}.log

set -e

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

NUM_GPUS=8
INPUT_DIR="data/plants_3dgs"
OUTPUT_DIR="data/plants_video"
LOG_DIR="logs"
CONDA_ENV="planttwin"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] Showing commands without executing"
fi

# Count available plants
N_PLANTS=$(find "$INPUT_DIR" -maxdepth 1 -mindepth 1 -type d | wc -l)
N_FORCE_TYPES=5
TOTAL_VIDEOS=$((N_PLANTS * N_FORCE_TYPES))
PER_GPU=$((N_PLANTS / NUM_GPUS))

echo "=== gen_video parallel launch ==="
echo "GPUs:          $NUM_GPUS"
echo "Plants found:  $N_PLANTS"
echo "Force types:   $N_FORCE_TYPES"
echo "Total videos:  $TOTAL_VIDEOS ($PER_GPU plants × $N_FORCE_TYPES = $((PER_GPU * N_FORCE_TYPES)) per GPU)"
echo "Output:        $OUTPUT_DIR"
echo "Logs:          $LOG_DIR/gen_video_gpu{0..7}.log"
echo ""

if [ "$N_PLANTS" -eq 0 ]; then
    echo "ERROR: No plant directories found in $INPUT_DIR"
    echo "       Run scripts/run_gen_3dgs_parallel.sh first."
    exit 1
fi

PIDS=()
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    CMD="conda run -n $CONDA_ENV --no-capture-output bash -c \
'CUDA_VISIBLE_DEVICES=$GPU_ID python data/generation/gen_video.py \
  --input_dir $INPUT_DIR \
  --output_dir $OUTPUT_DIR \
  --shard_id $GPU_ID \
  --num_shards $NUM_GPUS'"

    LOG="$LOG_DIR/gen_video_gpu${GPU_ID}.log"

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
echo "Monitor with: tail -f logs/gen_video_gpu0.log"
echo "Watch all:    tail -f logs/gen_video_gpu*.log"
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
    ACTUAL=$(find "$OUTPUT_DIR" -name "motion.mp4" | wc -l)
    echo ""
    echo "=== gen_video complete: $ACTUAL videos generated in $OUTPUT_DIR ==="
else
    echo ""
    echo "=== gen_video finished with $FAILED failed workers — check logs ==="
    exit 1
fi
