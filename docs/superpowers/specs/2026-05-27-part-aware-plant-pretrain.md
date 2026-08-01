# Part-Aware Plant Physics Pretrain

## Pipeline

```
Static 3DGS (canonical frame, from TRELLIS)
    │
    ├──→ Render static image ──→ DINO / PhysX-Omni VLM ──→ Structure Info
    │                                                          │
    │    ┌─────────────────────────────────────────────────────┘
    │    │   part labels: trunk / branch / leaf
    │    │   material priors: stiff / semi-rigid / flexible
    │    │   deformation constraints per part
    │    │
    ├──→ Gaussian features (pos + SH + cov + opacity)
    │    + Structure Info (concatenated or as condition)
    │    │
    │    ▼
    │  PhysicsNetwork (adapted from OmniPhysGS KNNTransformer)
    │    │  FPS grouping → group encoding → transformer → output heads
    │    │  Output: per-Gaussian {k, damp, drag, mass}
    │    │
    │    ▼
    │  Spring-Mass Simulation (simulation/spring_mass.py, extended)
    │    │  intra-part springs + cross-part attachment springs
    │    │  trunk base anchored, wind drag per-part
    │    │  → trajectory [T, N, 3]
    │    │
    │    ▼
    │  Differentiable Rendering (models/renderer/gaussian_renderer.py)
    │    │  → rendered video [T, 3, H, W]
    │    │
    │    ▼
    └──→ Losses:
           L_sds      = Video diffusion SDS (Wan2.1 / ModelScope)
           L_struct   = Structure-based constraints on deformation
```

No VideoPhysicsDecoder. No video input. Everything starts from static 3DGS + canonical frame.

---

## Component Mapping

### What we reuse from OmniPhysGS (and what changes)

**PhysicsNetwork + KNNTransformer** (`third_party/OmniPhysGS/src/physics_guided_network/`)

OmniPhysGS flow:
```
Gaussian features = cat(pos, norm_shs, norm_cov, norm_opacity)  # [N, D]
                          ↓
              KNNTransformer(x=pos, features=features)
              FPS → groups → group_encoder → (transformer blocks) → heads
                          ↓
              e_cat [N, E], p_cat [N, P]   (constitutive model category weights)
                          ↓
              GumbelElasticity(F, e_cat) → stress tensor
              GumbelPlasticity(F, p_cat) → deformation gradient update
                          ↓
              MPM p2g2p solver
```

Our adaptation:
```
Gaussian features = cat(pos, norm_shs, norm_cov, norm_opacity, structure_info)  # [N, D+S]
                          ↓
              KNNTransformer(x=pos, features=features)   ← same architecture
              FPS → groups → group_encoder → (transformer blocks)
                          ↓
              NEW output heads:
                head_k    → per-Gaussian spring stiffness
                head_damp → per-Gaussian damping
                head_drag → per-Gaussian wind drag coefficient
                head_mass → per-Gaussian mass (or per-part constant)
                          ↓
              Spring-mass simulator (not MPM)
```

**Changes to KNNTransformer:**
1. Replace `to_group_e_cat` / `to_group_p_cat` heads with `head_k`, `head_damp`, `head_drag`, `head_mass`
2. `in_channels` increases by structure feature dimension (part embedding or DINO feature)
3. Everything else (Grouper, GroupEncoder, Block, Attention) stays identical

**What we drop from OmniPhysGS:**
- `GumbelElasticity` / `GumbelPlasticity` (constitutive model ensemble — MPM-specific)
- `MPMModel` (replaced by spring-mass)
- `physical_constitutive_models.py` (CorotatedElasticity, StVK, etc. — continuum mechanics)
- Warp/Taichi dependencies
- Particle filling preprocessing

**What we keep as-is:**
- `KNNTransformer` architecture: FPS grouping, GroupEncoder, Attention blocks
- `PhysicsNetwork` wrapper (change output heads only)
- Feature construction: `cat(pos, normalized_shs, normalized_cov, normalized_opacity)`
- Gaussian loading + rotation/scaling + sim area selection from `render_utils.py`
- Camera view utilities
- SDS guidance loop structure (but may use our `VideoSDSGuidance` instead of their `ModelscopeGuidance`)

---

## Structure Info: Two Paths

### Path A: DINO Features (simpler, no external API)

```python
# Render canonical 3DGS from multiple views
# Extract DINOv2 features per view
# Project features back to Gaussians via alpha-compositing or splatting
# Result: per-Gaussian DINO feature [N, D_dino]

dino_features = extract_dino_per_gaussian(static_3dgs, views)  # [N, 384] for DINOv2-S

# Option 1: concatenate directly to Gaussian features as input to KNNTransformer
features = cat(pos, norm_shs, norm_cov, norm_opacity, dino_features)

# Option 2: cluster first, use cluster ID as part label → embedding
part_labels = spectral_clustering(dino_features, n_clusters=3)  # trunk/branch/leaf
part_embed = embedding_layer(part_labels)  # [N, D_part]
features = cat(pos, norm_shs, norm_cov, norm_opacity, part_embed)
```

DINO clustering for plants works because:
- Bark/wood and leaf surfaces have very different texture features
- DINOv2 features are semantically meaningful even without finetuning
- 3 clusters (trunk/branch/leaf) aligns well with natural plant structure

### Path B: PhysX-Omni VLM (richer, needs API)

```python
# Render canonical view → send to VLM
# Get: part list, material priors, RLE voxel per part
# Decode RLE → voxel occupancy per part
# Query Gaussian centers against voxel grids → part labels

vlm_output = physx_omni_vlm(rendered_image)
# vlm_output.parts = [
#   {name: "trunk", type: "rigid_wood", stiffness: "high", rle: "..."},
#   {name: "branch_0", type: "semi_rigid_wood", stiffness: "high", rle: "..."},
#   {name: "leaf_0", type: "flexible_leaf", stiffness: "low", rle: "..."},
# ]

for part in vlm_output.parts:
    voxel_grid = decode_rle(part.rle)
    mask = query_voxel(gaussian_centers, voxel_grid)
    part_labels[mask] = part.id
```

### Path C: Geometric Heuristic (fallback, zero dependency)

```python
# Height + color + local density → trunk/branch/leaf
# Works for typical upright TRELLIS plants
height_norm = normalize(xyz[:, 2])  # Z-up
green_ratio = colors[:, 1] / colors.sum(dim=1)

trunk_mask = (height_norm < 0.2) & (density > threshold)
leaf_mask = green_ratio > 0.35
branch_mask = ~trunk_mask & ~leaf_mask
```

**Recommendation:** Start with Path A (DINO) or Path C (heuristic) for MVP. Upgrade to Path B later.

---

## Structure Loss (L_struct)

Structure info doesn't just feed into the network — it also constrains the simulation output.

```python
# 1. Part-aware deformation amplitude constraint
#    Trunk should barely move, leaves can move a lot
#    Measured as displacement from canonical position across trajectory
displacement = (trajectory - trajectory[0:1]).norm(dim=-1)  # [T, N]

# Trunk: penalize any significant motion
L_trunk_anchor = displacement[:, trunk_mask].mean()

# Branch: penalize excessive motion (soft upper bound)
L_branch_limit = relu(displacement[:, branch_mask] - branch_threshold).mean()

# Leaf: no penalty on amplitude — let SDS drive the motion freely
# (optionally: penalize leaf breaking away from branch)

# 2. Part material consistency
#    Predicted k/damp should be similar within same part instance
for part_id in unique_parts:
    mask = (part_labels == part_id)
    L_consistency += predicted_k[mask].var() + predicted_damp[mask].var()

# 3. Material prior ordering
#    k_trunk > k_branch > k_leaf (stiffness ordering)
L_ordering = relu(k_leaf.mean() - k_branch.mean()) + relu(k_branch.mean() - k_trunk.mean())

# 4. Attachment preservation
#    Gaussians at part boundaries should maintain connectivity
#    (branch-leaf boundary Gaussians shouldn't drift apart)
for (part_i, part_j) in adjacent_part_pairs:
    boundary_dist = pairwise_distance(boundary_gaussians_i, boundary_gaussians_j)
    L_attach += (boundary_dist - initial_boundary_dist).pow(2).mean()

# Combined
L_struct = λ_anchor * L_trunk_anchor
         + λ_branch * L_branch_limit
         + λ_consist * L_consistency
         + λ_order * L_ordering
         + λ_attach * L_attach
```

---

## Modifications to Existing Code

### 1. `simulation/spring_mass.py` — extend for part-awareness

Current `forward()` signature:
```python
def forward(self, physics_params, n_frames=10, xyz_all=None)
```

New signature:
```python
def forward(self, physics_params, n_frames=10, xyz_all=None,
            part_labels=None, wind_velocity=None)
```

New behaviors:
- **Trunk anchoring**: if `part_labels` provided, mask velocity updates for trunk Gaussians
- **Wind drag**: `wind_force[k] = drag[k] * wind_velocity` added to force computation
- **Part-typed stiffness**: K matrix already per-Gaussian, no structural change needed — the network just needs to predict different values per part

### 2. `models/physics_decoder/` — new module adapted from OmniPhysGS

New file: `models/physics_decoder/plant_material_network.py`

Wraps OmniPhysGS's `KNNTransformer` with:
- Modified `in_channels` to accept structure features
- New output heads for spring-mass params instead of constitutive model categories
- Structure-aware feature construction

```python
class PlantMaterialNetwork(nn.Module):
    """
    Predicts per-Gaussian spring-mass params from static Gaussian features + structure info.
    Backbone: OmniPhysGS KNNTransformer (FPS → GroupEncoder → Transformer → heads).
    """
    def __init__(self, gaussian_feat_dim, structure_feat_dim, 
                 num_groups=2048, group_size=32, hidden_size=768, depth=0, ...):
        # KNNTransformer with modified in_channels and output heads
        self.knn_backbone = KNNTransformer(
            elasticity_dim=1,    # placeholder, we replace heads
            plasticity_dim=1,    # placeholder
            in_channels=gaussian_feat_dim + structure_feat_dim,
            num_groups=num_groups,
            group_size=group_size,
            hidden_size=hidden_size,
            depth=depth,
        )
        # Replace OmniPhysGS heads
        self.head_k = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.head_damp = nn.Sequential(...)  # same structure
        self.head_drag = nn.Sequential(...)  # same structure

    def forward(self, pos, gaussian_features, structure_features):
        features = torch.cat([gaussian_features, structure_features], dim=-1)
        # Use KNNTransformer backbone but intercept before original heads
        # Get group-level features from FPS+GroupEncoder+Transformer
        group_feat = self.knn_backbone.get_group_features(pos, features)  # need to expose this
        # Apply our heads
        k = softplus(self.head_k(group_feat))
        damp = softplus(self.head_damp(group_feat))
        drag = sigmoid(self.head_drag(group_feat))
        return {'k': k, 'damp': damp, 'drag': drag}
```

Note: OmniPhysGS config uses `depth=0` for simpler objects (no attention blocks, just GroupEncoder). For plants this may be fine initially; deeper transformer can be tried later.

### 3. Training script — `scripts/pretrain/train_part_aware_pretrain.py`

Combines:
- Gaussian loading + feature construction (from OmniPhysGS `render_utils.load_params`)
- Structure extraction (DINO / heuristic / VLM)
- PlantMaterialNetwork forward → spring-mass params
- Spring-mass simulation → trajectory
- Differentiable rendering → video
- SDS loss + structure loss → backprop through network

---

## OmniPhysGS Config Adapted for Plants

```yaml
train:
  seed: 42
  gpu: 0
  model_path: 'data/plants_3dgs/rose_s42'
  epochs: 10
  internal_epochs: 30
  learning_rate: 5e-5
  prompt: "a rose plant swaying gently in the wind"

model:
  network: 'knn'
  normalize_features: True
  hidden_size: 768
  depth: 0              # start without attention (same as OmniPhysGS bear config)
  num_heads: 8
  mlp_ratio: 2
  num_groups: 4096       # may need fewer groups than OmniPhysGS for smaller plant GS
  group_size: 32

structure:
  method: 'dino'         # 'dino' | 'heuristic' | 'vlm'
  n_parts: 3             # trunk / branch / leaf
  dino_model: 'dinov2_vits14'
  part_embed_dim: 32

sim:
  dt: 0.03
  n_step: 100
  n_frames: 16
  k_neighbors: 256
  gravity: [0.0, 0.0, -9.8]
  wind_velocity: [0.5, 0.0, 0.0]  # default gentle wind

loss:
  lambda_sds: 1.0
  lambda_anchor: 10.0     # trunk must stay fixed
  lambda_branch: 1.0      # branch deformation softly bounded
  lambda_consistency: 0.1  # within-part material uniformity
  lambda_ordering: 0.5     # k_trunk > k_branch > k_leaf
  lambda_attach: 1.0       # part boundaries stay connected
  lambda_smooth: 0.01      # spatial smoothness on kNN
```

---

## Implementation Status

| Step | What | Status | File |
|------|------|--------|------|
| 1 | `PlantMaterialNetwork` wrapping KNNTransformer with new heads | **Done** | `models/physics_decoder/plant_material_network.py` |
| 2 | Heuristic part labeling (height + color) | **Done** | `models/structure/structure_extractor.py` |
| 3 | Extend `SpringMassSimulator` with trunk anchoring + wind drag | **Done** | `simulation/spring_mass.py` |
| 4 | Gaussian feature construction (reuse OmniPhysGS `load_params` logic) | **Done** | `scripts/pretrain/train_part_aware_pretrain.py` |
| 5 | Structure loss functions | **Done** | `optimization/structure_loss.py` |
| 6 | Training script combining all above | **Done** | `scripts/pretrain/train_part_aware_pretrain.py` |
| 7 | DINO feature extraction + projection to Gaussians | Stub | `models/structure/structure_extractor.py` (needs rasterizer alpha-weight projection) |
| 8 | VLM-RLE part labeling (upgrade path) | Not started | — |

End-to-end gradient flow verified: PlantMaterialNetwork → SpringMassSimulator → StructureLoss ✓
