#!/bin/bash
# Full environment setup for PlantTwin on A6000/Ampere (sm_86), CUDA 12.4
set -e

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

echo "=== PlantTwin Environment Setup ==="
echo "Workspace: $WORKSPACE"
echo "CUDA: $(nvcc --version | grep release)"

# ── 1. Conda environment ──────────────────────────────────────────────────────
if conda env list | grep -q "^planttwin "; then
    echo "[1/5] planttwin env already exists, skipping create"
else
    echo "[1/5] Creating conda env: planttwin (python=3.10)"
    conda create -n planttwin python=3.10 -y
fi

CONDA_RUN="conda run -n planttwin --no-capture-output"

# ── 2. PyTorch (cu124 for A6000/Ampere) ──────────────────────────────────────
echo "[2/5] Installing PyTorch 2.6.0+cu124"
$CONDA_RUN pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ── 3. Core dependencies ──────────────────────────────────────────────────────
echo "[3/5] Installing Python dependencies"
$CONDA_RUN pip install \
    "xformers==0.0.29.post3" \
    diffusers==0.32.2 \
    accelerate \
    taichi \
    spconv-cu120 \
    pillow \
    imageio \
    imageio-ffmpeg \
    tqdm \
    easydict \
    "opencv-python-headless" \
    scipy \
    ninja \
    rembg \
    onnxruntime-gpu \
    trimesh \
    open3d \
    xatlas \
    pyvista \
    pymeshfix \
    igraph \
    "transformers>=4.40.0" \
    safetensors \
    sentencepiece \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# ── 4. Third-party repos ──────────────────────────────────────────────────────
echo "[4/5] Cloning third-party repos"

if [ ! -d "$WORKSPACE/third_party/TRELLIS/.git" ]; then
    echo "  Cloning TRELLIS..."
    git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git \
        "$WORKSPACE/third_party/TRELLIS"
else
    echo "  TRELLIS already cloned"
fi

# Patch TRELLIS to remove kaolin dependency
FLEXICUBES="$WORKSPACE/third_party/TRELLIS/trellis/representations/mesh/flexicubes/flexicubes.py"
if [ -f "$FLEXICUBES" ] && grep -q "from kaolin" "$FLEXICUBES"; then
    sed -i 's/from kaolin.utils.testing import check_tensor/def check_tensor(*a, **kw): pass/' \
        "$FLEXICUBES"
    echo "  Patched kaolin dependency"
fi

# Install TRELLIS requirements (excluding kaolin)
if [ -f "$WORKSPACE/third_party/TRELLIS/requirements.txt" ]; then
    grep -v kaolin "$WORKSPACE/third_party/TRELLIS/requirements.txt" > /tmp/trellis_reqs.txt
    $CONDA_RUN pip install -r /tmp/trellis_reqs.txt --no-deps 2>/dev/null || true
fi

if [ ! -d "$WORKSPACE/third_party/ReconPhys/.git" ]; then
    git clone -b Code https://github.com/chuanshuogushi/ReconPhys.git \
        "$WORKSPACE/third_party/ReconPhys" || echo "  WARNING: ReconPhys clone failed (non-critical)"
fi

if [ ! -d "$WORKSPACE/third_party/OmniPhysGS/.git" ]; then
    git clone https://github.com/wgsxm/OmniPhysGS.git \
        "$WORKSPACE/third_party/OmniPhysGS" || echo "  WARNING: OmniPhysGS clone failed (non-critical)"
fi

# ── 5. Compile CUDA extensions ────────────────────────────────────────────────
echo "[5/5] Compiling CUDA extensions (CUDA_HOME=/usr/local/cuda)"
export CUDA_HOME=/usr/local/cuda

# diff-gaussian-rasterization
if ! $CONDA_RUN python -c "import diff_gaussian_rasterization" 2>/dev/null; then
    echo "  Compiling diff-gaussian-rasterization..."
    if [ ! -d "/tmp/mip-splatting" ]; then
        git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting
    fi
    $CONDA_RUN pip install \
        /tmp/mip-splatting/submodules/diff-gaussian-rasterization/ \
        --no-build-isolation
else
    echo "  diff-gaussian-rasterization already installed"
fi

# diffoctreerast
if ! $CONDA_RUN python -c "import diffoctreerast" 2>/dev/null; then
    echo "  Compiling diffoctreerast..."
    if [ ! -d "/tmp/diffoctreerast" ]; then
        git clone --recurse-submodules \
            https://github.com/JeffreyXiang/diffoctreerast.git /tmp/diffoctreerast
    fi
    $CONDA_RUN pip install /tmp/diffoctreerast --no-build-isolation
else
    echo "  diffoctreerast already installed"
fi

# nvdiffrast
if ! $CONDA_RUN python -c "import nvdiffrast" 2>/dev/null; then
    echo "  Compiling nvdiffrast..."
    if [ ! -d "/tmp/nvdiffrast" ]; then
        git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast
    fi
    $CONDA_RUN pip install /tmp/nvdiffrast --no-build-isolation
else
    echo "  nvdiffrast already installed"
fi

echo ""
echo "=== Setup complete! ==="
echo "Activate with: conda activate planttwin"
echo ""
echo "NOTE: Run 'huggingface-cli login' inside the env before generation"
echo "      (needed for TRELLIS and Wan2.1 model downloads)"
