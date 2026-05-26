# Stage 1: Video Diffusion Prior for Plant Physics Decoder Pretrain

## Goal

Pretrain a feed-forward plant physics decoder that predicts spring-mass simulation parameters from video input, using video diffusion priors (SDS) as the primary supervision signal. No real captured dynamic plant data is needed at this stage.

## Architecture Overview

```
┌─────────── Data Generation ───────────┐
│ Text → TRELLIS → Static 3DGS          │
│ Static 3DGS → Render → Wan2.1 → Video │
└────────────────────────────────────────┘
         ↓ (generated videos + 3DGS)
┌─────────── SDS Optimization Path ─────┐
│ Static 3DGS + spring-mass graph       │
│ → optimize params via Wan2.1 SDS loss │
│ (based on OmniPhysGS, MPM→spring-mass)│
└────────────────────────────────────────┘
         ↓ (pseudo-GT params / self-supervised)
┌─────────── Feed-forward Decoder ──────┐
│ Video → Physics Decoder → params      │
│ → DiffTaichi spring-mass sim → render │
│ → reconstruction loss vs input video  │
│ (based on ReconPhys video branch)     │
└────────────────────────────────────────┘
```

## Components

### 1. Data Generation Pipeline

**3DGS Generation:**
- Model: TRELLIS (microsoft/TRELLIS, CVPR'25 Spotlight)
- Input: text prompts describing diverse plants
- Output: static 3D Gaussians per plant instance
- Scale: 100-500 plant instances initially

**Video Generation:**
- Model: Wan2.1 image-to-video
- Input: rendered static frame from 3DGS
- Output: 14-25 frame deformation videos
- Goal: realistic-looking plant motion (wind, touch, etc.)

### 2. SDS Optimization (per-scene, from OmniPhysGS)

- Code base: OmniPhysGS (github.com/wgsxm/OmniPhysGS)
- Key modification: replace MPM solver with spring-mass system
- Graph topology: KNN-based (no structure prior in Stage 1)
- Optimized parameters: per-edge stiffness, per-node mass/damping/drag
- SDS supervision: Wan2.1 provides gradient signal for plausible motion
- Purpose: validate spring-mass + SDS works; generate pseudo-GT parameters

### 3. Feed-forward Physics Decoder (from ReconPhys)

- Code base: ReconPhys (github.com/chuanshuogushi/ReconPhys, branch: Code)
- Architecture: dual-branch (appearance + physics) with video branch
- Input: plant deformation video
- Output: spring-mass parameters (stiffness, damping, mass per particle)
- Simulator: DiffTaichi differentiable spring-mass
- Training: self-supervised (predict → simulate → render → compare with input)
- Optional: use SDS-optimized params as additional supervision signal

## Code Organization

```
PlantTwin/
├── third_party/
│   ├── ReconPhys/          # Full clone (branch: Code)
│   ├── OmniPhysGS/        # Full clone
│   └── TRELLIS/            # Full clone
├── models/                 # Extracted model architectures
│   ├── physics_decoder/    # Feed-forward decoder (from ReconPhys)
│   └── renderer/           # 3DGS renderer
├── data/
│   ├── generation/         # Data generation scripts
│   │   ├── gen_3dgs.py     # TRELLIS text-to-3DGS
│   │   └── gen_video.py    # Wan2.1 motion video generation
│   ├── plants_3dgs/        # Generated static 3DGS
│   └── plants_video/       # Generated motion videos
├── simulation/             # Spring-mass diff simulator (from ReconPhys)
├── optimization/           # SDS optimization pipeline (from OmniPhysGS)
├── configs/                # Training and generation configs
├── scripts/                # Training/eval scripts
└── environment.yml         # Conda environment
```

## Environment Requirements

- CUDA: 12.4
- PyTorch: >= 2.3 (cu124 build)
- Taichi: >= 1.7 (for DiffTaichi, CUDA 12.x support)
- diffusers: latest (for Wan2.1)
- Key TRELLIS deps: spconv, flash-attn, xformers
- Key OmniPhysGS deps: diff-gaussian-rasterization, warp/taichi

## Implementation Phases

### Phase 1a: Environment + Data Generation
1. Clone all repos to third_party/
2. Set up conda env based on TRELLIS requirements
3. Get TRELLIS running for text-to-3DGS generation
4. Generate initial batch of plant 3DGS
5. Set up Wan2.1 pipeline for motion video generation
6. Generate plant deformation videos

### Phase 1b: SDS Optimization Validation
1. Extract OmniPhysGS SDS pipeline
2. Replace MPM with spring-mass from ReconPhys
3. Build KNN graph on 3DGS particles
4. Run SDS optimization on generated plants
5. Validate spring-mass produces plausible motion under SDS

### Phase 1c: Feed-forward Decoder Training
1. Extract ReconPhys video branch architecture
2. Adapt to plant data format
3. Train on generated videos (self-supervised)
4. Optionally add SDS-optimized params as supervision
5. Evaluate decoder quality on held-out plants

## Key Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Wan2.1-generated videos not physically plausible | Use SDS to constrain simulation, not raw Wan2.1 output as GT |
| Spring-mass insufficient for plant deformation | Start simple, can upgrade to more expressive springs (bending, torsion) |
| TRELLIS plant quality insufficient | Supplement with other 3D gen models or real reconstructions |
| Domain gap between generated and real plants | This is pretraining only; Stage 3 finetunes on real data |
| DiffTaichi gradient instability | Use gradient clipping, short simulation horizons |

## Success Criteria (Stage 1)

1. TRELLIS generates diverse, recognizable plant 3DGS from text
2. Wan2.1 produces motion videos that look like plant deformation
3. SDS optimization converges and produces visually plausible spring-mass dynamics
4. Feed-forward decoder trained on generated data can predict reasonable parameters
5. Predicted parameters, when simulated, produce motion that roughly matches input video

## Dependencies on Later Stages

- Stage 2 provides real captured data for finetuning
- Stage 3 introduces structure prior (plant graph topology replaces KNN)
- Stage 3 adds structure-based regularization on the decoder
