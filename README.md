# PlantTwin

Build **re-simulatable, sim-ready digital plants** by combining a static
[GaussianPlant](https://github.com/Yrainy0615/GaussianPlant) reconstruction with
**dynamic (video) information** that refines the plant's *structure* and estimates its
*physics*.

The core idea: a single static reconstruction gives you appearance and a rough branch
skeleton, but it is blind to **depth** and **connectivity** — two branches can look
adjacent in one view yet be unconnected, and a node's depth is unobservable from one
image. **Motion disambiguates both.** A monocular video of the plant swaying (wind) or
being pulled (hand) provides the residual signal that tells us *where the skeleton is
wrong* and *how stiff each part is*.

```
GaussianPlant static reconstruction            Monocular dynamic video
  (AppGas + branch graph + leaf clusters)         (wind / hand-pull)
                    │                                     │
                    ▼                                     ▼
        rooted branch tree + LBS skinning     ── motion residual ──►  refine structure
        (graph_cleanup, leaf_attachment,          (depth via video RGB,
         edge_binding, skinning)                    topology via 3D rigidity)
                    │                                     │
                    └──────────────► joint optimization ◄─┘
                        (articulated dynamics + per-part stiffness/damping
                         + contact force)  →  sim-ready re-drivable plant
```

---

## Three tracks

| Track | What | Entry points |
|---|---|---|
| **Per-scene refinement** (main) | Fit physics + refine structure of one real plant from static recon + video. | `scripts/pipeline/` |
| **Structure-from-motion PoC** | Synthetic-GT validation that motion recovers depth & topology. | `scripts/poc/` |
| **Feed-forward pretrain** | Learn a plant→physics decoder via video-diffusion SDS (no dynamic GT needed). | `scripts/pretrain/` |

The **per-scene refinement** track is the mature one. See
[`docs/per_scene_v11_method.md`](docs/per_scene_v11_method.md) (physics, frozen skeleton)
and [`docs/per_scene_struct_opt_method.md`](docs/per_scene_struct_opt_method.md)
(structure-in-the-loop: §8 "Motion-in-the-Loop" is the depth+topology refinement).

---

## Per-scene pipeline (main workflow)

Input: a GaussianPlant reconstruction bundle (default `/mnt/data/gaussianplant_data/<scene>`,
e.g. `newplant9`), COLMAP cameras, and a monocular motion video.

```bash
# 0. (offline) structure is built deterministically inside the scripts:
#    branch_mst → rooted tree → per-edge radii → leaf attachment → LBS skinning

# 1. Precompute video targets (SAM masks + RAFT flow) and contact bundle
python scripts/pipeline/precompute_video_targets.py --source <scene> --output-dir <out>
python scripts/pipeline/track_video_cotracker.py     --video <motion.mp4> --out <tracks.pt>
python scripts/pipeline/precompute_contact.py        --source <scene> --output-dir <out>

# 2a. Per-scene physics optimization (frozen skeleton — v11)
python scripts/pipeline/optimize_per_scene.py --source <scene> --output-dir <out> \
    --target-video <motion.mp4>

# 2b. Joint motion + structure refinement (depth + soft topology — motion-in-the-loop)
python scripts/pipeline/fuse_motion_structure.py --source <scene> --output-dir <out>

# 3. Export a sim-ready re-drivable plant
python scripts/pipeline/export_sim_ready.py --source <scene> --output-dir <out>
```

Validate the underlying claims on synthetic ground truth (no real video needed):

```bash
python scripts/poc/synth_poc.py            # motion recovers node depth
python scripts/poc/synth_poc_topology.py   # motion recovers connectivity (3D rigidity)
```

---

## Repository layout

```
PlantTwin/
├── data/
│   ├── gaussian_plant_loader.py   # load GaussianPlant bundle (AppGas + branch graph + leaves)
│   ├── colmap_loader.py           # COLMAP sparse → renderer camera
│   └── generation/                # text→3DGS (TRELLIS) + text→video (Wan2.1) data gen
├── models/
│   ├── structure/                 # branch tree, edge binding, skinning, densify/decimate, motion residual
│   ├── physics_decoder/           # feed-forward plant→physics network (pretrain track)
│   └── renderer/                  # differentiable Gaussian rasterizer wrapper (mip-splatting)
├── simulation/                    # articulated-chain dynamics, leaf dynamics, contact force
├── optimization/                  # struct_constraints, structure_loss, video_loss
├── scripts/
│   ├── pipeline/                  # per-scene motion→structure refinement (main)
│   ├── poc/                       # synthetic-GT structure-from-motion validation
│   ├── pretrain/                  # SDS / part-aware feed-forward pretrain
│   ├── viz/                       # renders, overlays, comparisons, HTML reports
│   ├── datagen/                   # 3DGS + video generation launchers
│   ├── debug/                     # COLMAP alignment, hand detection, root/anchor debugging
│   ├── demo/                      # interactive / scripted re-drive demos
│   └── legacy/                    # superseded PhysX-VLM / DINO exploration
├── tests/                         # unit / integration tests (run from repo root, PYTHONPATH=.)
├── docs/                          # method write-ups (see docs/README.md)
└── third_party/
    ├── GaussianPlant/             # static reconstruction repo (code only)
    ├── TRELLIS/ OmniPhysGS/ ReconPhys/ PhysX-Omni/
```

All scripts use absolute package imports (`from models.structure import ...`) and expect to
be run from the repo root with `PYTHONPATH=.` (or `python -m scripts.pipeline.<name>`).

---

## Installation

Verified on **8× RTX A6000** (Ampere sm_86, driver 535.146.02).

```bash
conda env create -f environment.yml      # planttwin: python 3.10, torch 2.6.0+cu124, cuda-toolkit 12.4
conda activate planttwin
```

> **GPU note.** `environment.yml` targets Ampere/Ada GPUs (CUDA 12.4). For a Blackwell
> GPU (RTX 50xx, sm_120) switch to `torch==2.12.0`/`torchvision==0.27.0` (cu130) and
> `cuda-toolkit=13.0` — see the comments in `environment.yml`.

### Custom CUDA extensions (compile from source)

The differentiable renderer needs the **mip-splatting** rasterizer; TRELLIS data
generation additionally needs `diffoctreerast` and `nvdiffrast`.

```bash
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="8.6"          # A6000; use 12.0 for Blackwell

# renderer (required for optimize_per_scene / fuse_motion_structure)
git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting
pip install /tmp/mip-splatting/submodules/diff-gaussian-rasterization/ --no-build-isolation

# data generation only (optional)
git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/diffoctreerast
pip install /tmp/diffoctreerast --no-build-isolation
git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast
pip install /tmp/nvdiffrast --no-build-isolation
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15
```

### Third-party repos

`GaussianPlant` (static reconstruction) is vendored in `third_party/GaussianPlant`.
Data-generation / baseline repos:

```bash
cd third_party
git clone -b Code https://github.com/chuanshuogushi/ReconPhys.git
git clone https://github.com/wgsxm/OmniPhysGS.git
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
# TRELLIS: patch out the kaolin dependency (Gaussians don't need it)
sed -i 's/from kaolin.utils.testing import check_tensor/def check_tensor(*a, **kw): pass/' \
  TRELLIS/trellis/representations/mesh/flexicubes/flexicubes.py
```

---

## Data generation (pretrain track)

```bash
# text → static 3DGS (TRELLIS)
python data/generation/gen_3dgs.py --prompt_file configs/plant_prompts.txt \
  --output_dir data/plants_3dgs --smoke_test
# static 3DGS → motion video (Wan2.1)
python data/generation/gen_video.py --input_dir data/plants_3dgs \
  --output_dir data/plants_video --smoke_test
# SDS pretrain of the physics decoder
python scripts/pretrain/train_sds_e2e.py --ply <plant>/gaussian.ply \
  --prompt "a single plant swaying in the wind" --epochs 200 --n_frames 16
```

---

## Docs

See [`docs/README.md`](docs/README.md) for the full index. Start with the per-scene method
write-ups above; `docs/baseline.md` compares against dense KNN spring-mass approaches.
