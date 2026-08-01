# docs/

Method write-ups for PlantTwin. Read in this order.

| Doc | What |
|---|---|
| [`per_scene_v11_method.md`](per_scene_v11_method.md) | **Per-scene physics (v11).** GaussianPlant static recon + monocular video + COLMAP → sim-ready plant. Offline structure build (rooted tree, per-edge radii, leaf attachment, LBS skinning), articulated-chain + petiole dynamics, contact-force trajectory, losses, and convergence on `newplant9`. Skeleton is **frozen**. |
| [`per_scene_struct_opt_method.md`](per_scene_struct_opt_method.md) | **Structure-in-the-loop.** Makes the branch tree itself optimizable. §1–7: per-node rest-position correction δ + densification driven by motion residual, with geometric-containment and motion-direction constraints. **§8 "Motion-in-the-Loop"** is the depth+topology refinement — the headline "dynamics refine structure" result (edge binding, 3D rigidity, joint optimizer, synthetic-GT verdicts). |
| [`superpowers/specs/2026-05-27-part-aware-plant-pretrain.md`](superpowers/specs/2026-05-27-part-aware-plant-pretrain.md) | **Feed-forward pretrain spec.** Part-aware plant→physics network (adapted from OmniPhysGS KNNTransformer) trained with video-diffusion SDS + structure losses. Structure info via DINO / VLM / heuristic. |
| [`baseline.md`](baseline.md) | Comparison against dense KNN spring-mass approaches (Spring-Gaus / OmniPhysGS) and why a discrete, plant-semantic branch tree is preferred. |
| [`data_generation_tips.md`](data_generation_tips.md) | Practical notes for the TRELLIS + Wan2.1 data-generation pipeline. |

## Where the code lives

- Structure build / refinement: `models/structure/`, `optimization/struct_constraints.py`
- Dynamics: `simulation/`
- Per-scene entry points: `scripts/pipeline/`; synthetic validation: `scripts/poc/`
- See [`../scripts/README.md`](../scripts/README.md) for the per-script guide.
