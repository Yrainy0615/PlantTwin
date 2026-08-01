#!/bin/bash
# Launch parallel gen_video workers, one per GPU.
# Workers are launched via direct `nohup python` (not conda run wrapper) to
# avoid the conda subprocess nesting that left inner Python processes unprotected
# from external SIGKILL.
#
# Usage:
#   bash scripts/run_gen_video_parallel.sh [--dry-run] [--num-gpus N]
#
# Defaults to all 8 GPUs. Pass --num-gpus 4 to use only GPUs 0-3.
# Logs: logs/gen_video_gpu{N}.log

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
        --num-gpus) shift; NUM_GPUS="$1" ;;
        --num-gpus=*) NUM_GPUS="${arg#*=}" ;;
    esac
done

if $DRY_RUN; then echo "[DRY RUN] Showing commands without executing"; fi

# Resolve conda python path once so workers don't need conda run
PYTHON="$(conda run -n $CONDA_ENV which python 2>/dev/null)"
if [[ -z "$PYTHON" ]]; then
    echo "ERROR: cannot find python in conda env '$CONDA_ENV'"
    exit 1
fi

N_PLANTS=$(find "$INPUT_DIR" -maxdepth 1 -mindepth 1 -type d | wc -l)
N_FORCE_TYPES=4
TOTAL_VIDEOS=$((N_PLANTS * N_FORCE_TYPES))
PER_GPU=$((N_PLANTS / NUM_GPUS))

echo "=== gen_video parallel launch ==="
echo "GPUs:          $NUM_GPUS (GPU IDs 0..$((NUM_GPUS-1)))"
echo "Plants found:  $N_PLANTS"
echo "Force types:   $N_FORCE_TYPES"
echo "Total videos:  $TOTAL_VIDEOS (~$((PER_GPU * N_FORCE_TYPES)) per GPU)"
echo "Python:        $PYTHON"
echo "Output:        $OUTPUT_DIR"
echo ""

if [ "$N_PLANTS" -eq 0 ]; then
    echo "ERROR: No plant directories found in $INPUT_DIR"
    exit 1
fi

PIDS=()
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    LOG="$LOG_DIR/gen_video_gpu${GPU_ID}.log"
    CMD="CUDA_VISIBLE_DEVICES=$GPU_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PYTHON data/generation/gen_video.py \
  --input_dir $INPUT_DIR \
  --output_dir $OUTPUT_DIR \
  --shard_id $GPU_ID \
  --num_shards $NUM_GPUS"

    if $DRY_RUN; then
        echo "  GPU $GPU_ID: nohup bash -c '$CMD' > $LOG 2>&1 &"
    else
        echo "  Launching GPU $GPU_ID → $LOG"
        nohup bash -c "$CMD" > "$LOG" 2>&1 &
        PIDS+=($!)
    fi
done

if $DRY_RUN; then echo ""; echo "Remove --dry-run to execute."; exit 0; fi

echo ""
echo "All $NUM_GPUS workers launched (PIDs: ${PIDS[*]})"
echo "Monitor: tail -f logs/gen_video_gpu0.log"
echo "Watch all: tail -f logs/gen_video_gpu*.log"
echo ""
echo "Waiting for all workers to finish..."
FAILED=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || { echo "  WARNING: PID $PID exited with error"; FAILED=$((FAILED + 1)); }
done

ACTUAL=$(find "$OUTPUT_DIR" -name "motion.mp4" | wc -l)
if [ $FAILED -eq 0 ]; then
    echo "=== gen_video complete: $ACTUAL / $TOTAL_VIDEOS videos in $OUTPUT_DIR ==="
else
    echo "=== gen_video finished with $FAILED failed workers — check logs ($ACTUAL videos done) ==="
    exit 1
fi
