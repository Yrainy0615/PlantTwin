#!/bin/bash
# Resilient video generation launcher.
# Each GPU runs in a restart loop: if the worker is killed (OOM, signal, etc.)
# it restarts automatically. The gen_video.py skip-existing logic means each
# restart resumes from where work left off.
#
# Usage:
#   bash scripts/run_gen_video_resilient.sh [--dry-run] [--num-gpus N]
#
# The loop exits when the GPU's shard has no remaining work (all motion.mp4
# files for that shard already exist). Overall completion is logged.

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
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --num-gpus=*) NUM_GPUS="${arg#*=}" ;;
        --num-gpus) shift; NUM_GPUS="$1" ;;
    esac
done

PYTHON="$(conda run -n $CONDA_ENV which python 2>/dev/null)"
if [[ -z "$PYTHON" ]]; then
    echo "ERROR: cannot find python in conda env '$CONDA_ENV'"
    exit 1
fi

N_PLANTS=$(find "$INPUT_DIR" -maxdepth 1 -mindepth 1 -type d | wc -l)
N_FORCE_TYPES=4
TOTAL_VIDEOS=$((N_PLANTS * N_FORCE_TYPES))

echo "=== resilient gen_video launch ==="
echo "GPUs:         $NUM_GPUS"
echo "Total videos: $TOTAL_VIDEOS"
echo "Python:       $PYTHON"
echo ""

if [ "$N_PLANTS" -eq 0 ]; then
    echo "ERROR: No plant directories in $INPUT_DIR"; exit 1
fi

# Worker loop for a single GPU shard. Runs until all videos for this shard exist.
run_gpu_loop() {
    local GPU_ID=$1
    local LOG="$LOG_DIR/gen_video_gpu${GPU_ID}.log"
    local attempt=0

    while true; do
        attempt=$((attempt + 1))
        echo "[GPU $GPU_ID] attempt $attempt — $(date '+%H:%M:%S')" >> "$LOG"

        CUDA_VISIBLE_DEVICES=$GPU_ID \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$PYTHON" data/generation/gen_video.py \
            --input_dir "$INPUT_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --shard_id "$GPU_ID" \
            --num_shards "$NUM_GPUS" \
            >> "$LOG" 2>&1
        EXIT=$?

        DONE=$(find "$OUTPUT_DIR" -name "motion.mp4" | wc -l)
        echo "[GPU $GPU_ID] attempt $attempt ended (exit=$EXIT). Total done: $DONE/$TOTAL_VIDEOS" >> "$LOG"

        if [ $EXIT -eq 0 ]; then
            echo "[GPU $GPU_ID] shard complete." >> "$LOG"
            break
        fi

        # Brief pause before restart to avoid rapid spin on persistent errors
        sleep 10
    done
}

export -f run_gpu_loop
export PYTHON INPUT_DIR OUTPUT_DIR LOG_DIR NUM_GPUS TOTAL_VIDEOS

if $DRY_RUN; then
    echo "Would launch GPU loops 0..$((NUM_GPUS-1)) in background."
    echo "Each loop restarts gen_video.py until its shard exits 0."
    exit 0
fi

PIDS=()
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    echo "  Launching GPU $GPU_ID restart-loop → logs/gen_video_gpu${GPU_ID}.log"
    bash -c "run_gpu_loop $GPU_ID" &
    PIDS+=($!)
done

echo ""
echo "All $NUM_GPUS loops launched (PIDs: ${PIDS[*]})"
echo "Monitor: tail -f logs/gen_video_gpu0.log"
echo "Watch all: tail -f logs/gen_video_gpu*.log"
echo ""
echo "Waiting for all loops to finish..."

FAILED=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || FAILED=$((FAILED + 1))
done

ACTUAL=$(find "$OUTPUT_DIR" -name "motion.mp4" | wc -l)
if [ $FAILED -eq 0 ]; then
    echo "=== All done: $ACTUAL / $TOTAL_VIDEOS videos ==="
else
    echo "=== Finished with $FAILED failed loops ($ACTUAL / $TOTAL_VIDEOS videos done) ==="
    exit 1
fi
