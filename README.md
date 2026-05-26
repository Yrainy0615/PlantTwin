# PlantTwin

Generate large-scale realistic 4D plant data by combining video diffusion priors with differentiable physics simulation.

## Overview

PlantTwin uses a three-stage pipeline:

1. **Stage 1 (Pretrain)**: Use video diffusion SDS to pretrain a plant physics decoder — learn plausible spring-mass parameters without real dynamic data
2. **Stage 2 (Dataset)**: Prepare training data from synthetic generation, existing datasets, and new captures
3. **Stage 3 (Finetune)**: Integrate structure priors and structure-based regularization for fine-grained physics estimation

### Stage 1 Pipeline

```
Text prompts → TRELLIS (text-to-3DGS) → Static plant Gaussians
                                              ↓
              Learnable physics params → Spring-mass simulation → Diff. rendering → SDS loss (ModelScope T2V)
                                              ↓
              Optimized params: stiffness (k), damping, velocity per particle
```

## Project Structure

```
PlantTwin/
├── configs/                  # Plant prompts and training configs
├── data/
│   └── generation/           # Data generation scripts (TRELLIS + Wan2.1)
├── models/
│   ├── physics_decoder/      # Feed-forward video → physics params (InternViT + Transformer)
│   └── renderer/             # Differentiable Gaussian splatting renderer
├── optimization/             # SDS guidance (ModelScope text-to-video)
├── simulation/               # Differentiable spring-mass simulator
├── scripts/                  # Training scripts
├── third_party/              # ReconPhys, OmniPhysGS, TRELLIS (gitignored)
└── docs/                     # Design specs and baseline comparison
```

## Installation

Tested on: RTX 5070 Ti (Blackwell sm_120), Ubuntu (WSL2), CUDA Toolkit 12.4 system-level.

### 1. Create conda environment

```bash
conda create -n planttwin python=3.10 -y
conda activate planttwin
```

### 2. Install PyTorch (Blackwell GPU support)

```bash
# PyTorch 2.12.0 with CUDA 13.0 — native sm_120 support
pip install torch==2.12.0 torchvision
```

For non-Blackwell GPUs (A100, 4090, etc.), use `pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124`.

### 3. Install CUDA Toolkit 13.0 (for compiling extensions)

```bash
# Required to match PyTorch's bundled CUDA for custom CUDA extension compilation
conda install -c nvidia cuda-toolkit=13.0 -y
```

### 4. Install dependencies

```bash
# Core
pip install xformers diffusers accelerate taichi spconv-cu120
pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja
pip install rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
```

### 5. Compile CUDA extensions

```bash
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

# diff-gaussian-rasterization (from mip-splatting)
git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting
pip install /tmp/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation

# diffoctreerast
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/diffoctreerast
pip install /tmp/diffoctreerast --no-build-isolation

# nvdiffrast
git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast
pip install /tmp/nvdiffrast --no-build-isolation
```

### 6. Clone third-party repos

```bash
cd third_party
git clone -b Code https://github.com/chuanshuogushi/ReconPhys.git
git clone https://github.com/wgsxm/OmniPhysGS.git
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
```

Patch TRELLIS to remove kaolin dependency (only needed for mesh, not Gaussians):
```bash
sed -i 's/from kaolin.utils.testing import check_tensor/def check_tensor(*a, **kw): pass/' \
  third_party/TRELLIS/trellis/representations/mesh/flexicubes/flexicubes.py
```

### 7. HuggingFace login (for TRELLIS model download)

```bash
huggingface-cli login
```

## Quick Start

### Generate plant 3DGS

```bash
python data/generation/gen_3dgs.py --prompt_file configs/plant_prompts.txt --output_dir data/plants_3dgs --smoke_test
```

### Generate motion videos (Wan2.1)

```bash
python data/generation/gen_video.py --input_dir data/plants_3dgs --output_dir data/plants_video --smoke_test
```

### Train with SDS

```bash
python scripts/train_sds_e2e.py \
  --ply data/plants_3dgs/<plant>/gaussian.ply \
  --prompt "a single plant swaying in the wind" \
  --epochs 200 --n_frames 16
```
