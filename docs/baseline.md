# Project Overview: Dynamic Plant 4D Data Generation

## Goal

The goal of this project is to generate large-scale realistic 4D plant data from real videos. The target pipeline is:

```text
shape reconstruction -> physics system integration -> physics parameter estimation -> simulation / generation
```

The input data will be videos of plants under human interaction. Depending on capture setting, the videos may be multi-view or single-view. The final system should recover a simulation-ready plant representation and use it to generate diverse physically plausible 4D plant motion under new interactions.

The long-term target is feed-forward physical parameter estimation. We want to move beyond per-scene inverse optimization and learn transferable plant material and structure priors, so that large-scale 4D plant data can be generated efficiently.

## Related Work Comparison

| Work | Target | Input data | Geometry representation | Physics representation | Material modeling | Parameter estimation | Generalization | Main limitation for our goal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MatPhys | General deformable objects | Single-view RGB interaction video | 3D Gaussians reconstructed from a keyframe | Spring-mass graph | Part-level material prior from DINO/MLLM + learnable material codebook | Feed-forward prediction of spring stiffness, damping, contact parameters | Stronger cross-scene and unseen-interaction transfer than per-scene fitting | Not plant-specific; topology is semantic/part-aware but does not explicitly encode botanical structure |
| ReconPhys | General deformable objects in synthetic dynamics | Single monocular video, mainly synthetic free-fall/collision/rebound data | Feed-forward 3DGS reconstruction | KNN spring-mass system bound to 3D Gaussians | Continuous physical attributes: mass, stiffness, damping, global friction | Feed-forward prediction trained self-supervised through differentiable simulation and rendering | Zero-shot transfer to unseen synthetic objects; fast inference without per-scene optimization | Synthetic-data-driven; dynamics are simpler than human-plant interaction; KNN topology lacks material/semantic/plant-structure priors |
| M-PhyGs | Real multi-material flowers | Dense multi-view static capture + sparse-view dynamic interaction video | Surface 3D Gaussians + internal regular particles | MLS-MPM continuum simulator | Joint material segmentation; estimates Young's modulus and density per particle; accounts for gravity | Per-scene inverse optimization with cascaded 3D/2D losses and temporal mini-batching | Good physical fitting for observed flower objects; can predict unseen frames/interactions after optimization | Optimization-heavy; requires stronger capture; no amortized feed-forward estimator; structure prior mainly from appearance/DINO rather than explicit plant graph |
| OmniPhysGS | General 4D physics-based generation | Multi-view static images + text prompt / video diffusion supervision | Constitutive 3D Gaussians | MPM with learnable constitutive models | Ensemble of expert constitutive models for elastic, viscoelastic, plastic, fluid-like materials | Per-scene SDS optimization using pretrained video diffusion | Broad material coverage and prompt-driven dynamics | Generation-oriented, not real plant reconstruction; time-intensive optimization; physical priors are material-general, not plant-structure-aware |

## Comparison Axes

### Physical Expressiveness

MPM-based methods such as M-PhyGs and OmniPhysGS are more physically expressive for continuum deformation, gravity, density, and material heterogeneity. They are better suited when we need accurate stress-strain behavior or volumetric material modeling.

Spring-mass systems, as used in MatPhys and ReconPhys, are less expressive but much lighter. They are easier to integrate with feed-forward prediction, interactive editing, and large-scale generation. For plants, this tradeoff is attractive because much visible plant motion is dominated by hierarchical bending, attachment constraints, damping, and contact, which can be represented effectively by a structured graph.

### Optimization vs Feed-Forward

M-PhyGs and OmniPhysGS optimize physical parameters per scene. This improves fitting quality but makes large-scale data generation expensive and can produce inconsistent parameters across sequences.

ReconPhys and MatPhys directly predict simulator parameters from video. ReconPhys demonstrates that spring-mass attributes can be amortized from synthetic monocular videos, while MatPhys further explores material-aware transfer from real interaction videos. This is closer to our final goal: use optimization only as a bootstrap or supervision source, then train an amortized model that estimates plant physics in one forward pass.

### Generalization

ReconPhys shows that feed-forward physical attribute prediction can transfer across synthetic objects when trained on large-scale simulated data. MatPhys explicitly targets cross-scene material consistency through a material codebook. M-PhyGs shows strong per-object fitting on real flowers but does not solve transferable parameter prediction. OmniPhysGS generalizes across material categories through constitutive-model mixtures, but its supervision comes from diffusion priors rather than observed plant interaction data.

Our project should combine these ideas: use plant-specific structure priors and learned material priors to make parameter prediction transferable across plant species, viewpoints, and interaction patterns.

### Input Assumptions

Multi-view video gives stronger geometry, contact localization, and motion supervision. It is the safer setting for building the first high-quality dataset and optimization baseline.

Single-view video is more scalable and is necessary for large-scale collection. The final feed-forward model should support single-view input, possibly using multi-view-trained models or optimized results as supervision.

## Proposed Method Direction

### 1. Physical Representation

A practical starting point is a structure-aware spring-mass plant model:

- Use 3D Gaussians or mesh/point primitives for appearance and rendering.
- Use a plant structure graph as the physical backbone.
- Attach visible surface primitives to graph nodes, organ nodes, or local deformation frames.
- Simulate dynamics with spring stiffness, bending stiffness, damping, drag, contact, and gravity.

This keeps the simulator lightweight and compatible with feed-forward inference. MPM can be used as an optimization baseline or upper-bound physical model, especially for evaluating whether spring-mass is sufficient for stems, leaves, petals, and branches.

### 2. Plant Structure Prior

The key innovation should be to introduce plant structure into dynamic modeling instead of relying only on KNN particle topology or appearance clustering.

Represent the plant as a 3D graph:

- Nodes: root/base anchors, stem joints, branch points, petiole endpoints, leaf anchors, leaf patches, flower/petal anchors.
- Edges: stem segments, branch segments, petioles, leaf veins, leaf surface connections, organ attachment links.
- Attributes: organ type, hierarchy depth, length, radius/thickness, orientation, local density, semantic/material embedding, visibility confidence.

This graph provides a structural prior for dynamics:

- Connectivity follows biological topology, not only Euclidean proximity.
- Stiffness can be parameterized by organ type and geometry, e.g. stem/branch stiffness depends on radius and length.
- Boundary conditions can respect attachment points, e.g. leaves move through petiole constraints rather than arbitrary nearest-neighbor springs.
- Damping and drag can be organ-aware, e.g. broad leaves and petals should have stronger air resistance than stems.
- Material sharing can be hierarchical, e.g. same species or same organ type should reuse similar latent material codes.

### 3. Parameter Estimation Strategy

Use a two-stage path from optimization to feed-forward learning:

1. Build an optimization baseline from multi-view videos.
2. Recover shape, plant graph, contact trajectories, and per-sequence physical parameters.
3. Use optimized parameters and simulated rollouts as pseudo-labels.
4. Train a feed-forward estimator conditioned on video motion, reconstructed geometry, plant graph, and organ/material embeddings.
5. Distill to single-view input by training with multi-view supervision but testing on sparse or monocular observations.

The feed-forward estimator should predict graph-level and edge-level parameters:

- Edge stiffness and bending stiffness.
- Rest length or correction terms.
- Local damping.
- Global/object-level drag and gravity response.
- Contact/controller coupling strength.
- Optional material code per organ or per graph substructure.

### 4. Structure Decoder from 4D Data

Large-scale generated 4D plant data can also be used to train a structure decoder. The decoder should recover a 3D plant graph from image/video observations.

Possible decoder outputs:

- A 3D graph with nodes, edges, and organ labels.
- Node positions and hierarchy.
- Attachment relations between stems, branches, leaves, and petals.
- Surface-to-structure skinning weights or Gaussian-to-graph assignments.
- Structure-conditioned latent codes for dynamics.

This creates a closed loop:

```text
real videos -> optimized / predicted plant physics -> 4D plant simulation data
4D plant data -> train structure decoder -> better graph reconstruction
better graph reconstruction -> better physical parameter estimation and generation
```

## Baselines and Experiments

### Baseline A: M-PhyGs-style Optimization

Use multi-view static reconstruction and sparse-view dynamic videos. Fit material and physical parameters through analysis-by-synthesis. This baseline is useful for high-quality pseudo-label generation and for measuring the upper-bound performance of per-scene optimization.

### Baseline B: MatPhys-style Feed-Forward Spring-Mass

Use a spring-mass graph and predict parameters from visual motion, geometry, and material cues. This is the closest baseline for the final system. The main modification is replacing generic part decomposition with plant-structure-aware graph construction.

### Baseline C: ReconPhys-style Feed-Forward Spring-Mass

Train a feed-forward model on synthetic plant-like or generic deformable data to predict mass, stiffness, damping, and friction from monocular videos. Use this baseline to test whether synthetic dynamics alone can transfer to real human-plant interaction videos, and whether KNN topology is sufficient.

### Baseline D: Structure-Agnostic Graph

Construct topology with KNN or local radius connections only. Compare against the plant-structure graph to isolate the effect of structural priors.

### Baseline E: Material-Only Prior

Use organ/material embeddings but no explicit plant hierarchy. This tests whether plant graph topology provides additional value beyond semantic material labels.

## Evaluation Plan

Core metrics:

- Reconstruction quality: RGB PSNR/SSIM/LPIPS and mask/geometry consistency.
- Motion prediction: 2D tracking error, 3D tracking error when available, Chamfer distance over time.
- Future prediction: rollout error on held-out frames.
- Unseen interaction: simulate new human contact trajectories and compare with held-out videos.
- Parameter consistency: variance of predicted parameters across similar species, organ types, and interactions.
- Generation quality: diversity and physical plausibility of generated 4D plant motion.

Key ablations:

- With vs without plant graph.
- KNN graph vs skeleton/organ graph.
- Object-level material code vs organ-level material code.
- Optimization-only vs feed-forward prediction.
- Multi-view input vs single-view input.
- With vs without gravity/contact modeling.

## Expected Contribution

The project can be positioned as structure-aware physical reconstruction and generation of dynamic plants from interaction videos. Compared with prior work:

- Unlike M-PhyGs, the final target is feed-forward and scalable rather than per-scene optimization.
- Unlike ReconPhys, the target is real plant interaction data with explicit plant structure priors rather than synthetic free-fall/collision dynamics with KNN topology.
- Unlike MatPhys, the physical graph is not only semantic/material-aware but explicitly plant-structure-aware.
- Unlike OmniPhysGS, the focus is real plant video reconstruction and data generation, not prompt-driven general 4D synthesis.

The central hypothesis is that plant dynamics are strongly constrained by botanical structure. If the model recovers or predicts a 3D plant graph, the physical parameter estimation problem becomes better posed, more transferable, and more useful for generating large-scale realistic 4D plant data.
