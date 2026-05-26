#!/bin/bash
# Full data generation pipeline:
#   Stage 1: Generate 200 plant 3DGS across 8 GPUs in parallel
#   Stage 2: Generate 1000 videos (200 plants × 5 force types) across 8 GPUs in parallel
#
# Usage:
#   bash scripts/run_full_generation.sh
#   bash scripts/run_full_generation.sh --skip-3dgs      # only run video stage
#   bash scripts/run_full_generation.sh --dry-run        # preview commands only

set -e

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

SKIP_3DGS=false
DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --skip-3dgs) SKIP_3DGS=true ;;
        --dry-run)   DRY_RUN=true ;;
    esac
done

DR_FLAG=""
$DRY_RUN && DR_FLAG="--dry-run"

echo "╔══════════════════════════════════════════╗"
echo "║   PlantTwin Full Data Generation          ║"
echo "║   Target: 200 plants, 1000 videos         ║"
echo "║   Hardware: 8× RTX A6000 (49 GB each)     ║"
echo "╚══════════════════════════════════════════╝"
echo ""
date

# ── Stage 1: Generate plant 3DGS ─────────────────────────────────────────────
if ! $SKIP_3DGS; then
    echo ""
    echo "━━━ Stage 1: Generate 200 plant Gaussians ━━━"
    echo "  8 GPUs × 25 plants each"
    echo "  TRELLIS text-to-3DGS (microsoft/TRELLIS-text-xlarge)"
    echo ""
    START_3DGS=$(date +%s)
    bash scripts/run_gen_3dgs_parallel.sh $DR_FLAG
    END_3DGS=$(date +%s)
    echo "Stage 1 elapsed: $(( (END_3DGS - START_3DGS) / 60 )) min"
else
    echo "[--skip-3dgs] Skipping Stage 1"
fi

# ── Stage 2: Generate motion videos ──────────────────────────────────────────
echo ""
echo "━━━ Stage 2: Generate 1000 motion videos ━━━"
echo "  8 GPUs × 25 plants × 5 force types = 125 videos per GPU"
echo "  Wan2.1 I2V (Wan-AI/Wan2.1-I2V-14B-480P-Diffusers)"
echo ""
START_VID=$(date +%s)
bash scripts/run_gen_video_parallel.sh $DR_FLAG
END_VID=$(date +%s)
echo "Stage 2 elapsed: $(( (END_VID - START_VID) / 60 )) min"

# ── Summary ───────────────────────────────────────────────────────────────────
if ! $DRY_RUN; then
    echo ""
    echo "━━━ Final Summary ━━━"
    N_GS=$(find data/plants_3dgs -name "gaussian.ply" 2>/dev/null | wc -l)
    N_VID=$(find data/plants_video -name "motion.mp4" 2>/dev/null | wc -l)
    echo "  Plant Gaussians: $N_GS"
    echo "  Motion videos:   $N_VID"
    echo ""
    date
fi
