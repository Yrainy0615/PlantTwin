# Survey — Foundation Models for Generating a 4D Plant Dataset

**Goal.** Generalize our dynamic-plant work into a **4D plant dataset (multi-view + swaying/wind motion)
built without manual capture**, and identify where a foundation / world model can replace or augment our
current TRELLIS + Wan + articulated-physics pipeline. Focus: reconstruction, video generation, world
models (Cosmos-like), and 4D/physics generation.

**TL;DR (the three questions).**
1. **Directly usable to synthesize a 4D plant dataset?** Yes — a practical stack exists today:
   *diverse static plants* (TRELLIS / 3DBonsai) → *controllable swaying* (**Force Prompting**, Wan, PhysDreamer)
   → *multi-view* (SynCamMaster / ReCamMaster / CVD) → *lift to 4D* (CAT4D / SV4D / GVFDiffusion / DreamGaussian4D).
2. **Is a Cosmos-like world model a viable engine?** Yes, and it is the highest-leverage option:
   **Cosmos-Transfer1/2.5** conditions on depth/segmentation/edges, so a **rendered GaussianPlant skeleton
   can drive photoreal, controllable plant video (Sim2Real)**; **Lyra** shows a Cosmos-family video model can
   be self-distilled into feed-forward 3D/4D Gaussians. Caveat: no plant-specific training — needs our
   structural/physical conditioning to respect botanical motion.
3. **Biggest extension point.** **Force Prompting** — a video model fine-tuned to obey *point forces and
   wind fields* (it literally pokes a plant), trained on Blender-synthesized **force-labeled** clips. It
   maps 1:1 onto our contact-force / wind optimization and can synthesize the exact (video, force) pairs our
   per-scene physics fit needs — removing manual hand-pull capture.

---

## Area 1 — Plant / tree reconstruction & neural-procedural structure

**GaussianPlant** (arXiv 2512.14087) — *our base system.* Jointly recovers plant **appearance and internal
structure** from multi-view images with 3DGS. Hierarchical, structure-aligned representation with explicit
botanical primitives: **branches = cylinders, leaves = disks**, with 3D Gaussians bound to each primitive so
conventional 3DGS rendering still works. Input: multi-view **images** (static). Output: rendered
reconstruction + extracted **branch skeleton + leaf instances**. → This is exactly the StPr our pipeline
consumes; everything below is about (a) improving that structure and (b) adding motion.

**Masks-to-Skeleton** (Sensors 2025, doi 10.3390/s25144354) — reconstructs a 3D tree skeleton **directly
from multi-view RGB segmentation masks**, using 3DGS to render silhouettes and **differentiably optimizing
node positions + adjacency** to fit multi-view masks — bypassing point-cloud quality. Nodes store
position+radius; adjacency encodes branch connectivity. → A cleaner, mask-driven alternative to our
branch-MST for **topology correction** (our "prune false edges / add missed branches"). Same differentiable-
silhouette idea we could fold into StPr correction.

**Smart-Tree** (arXiv 2303.11560, open code+data) — sparse-voxel CNN predicts per-point **radius + direction
to the medial axis**, then greedy skeletonization → robust branch skeleton from **point clouds**; robust to
self-occlusion and touching branches, trained on multi-species synthetic trees. → A learned skeleton prior we
could run on GaussianPlant's dense cloud to seed/repair the StPr graph (open weights = usable now).

**Autoregressive Generation of Static and Growing Trees** (SIGGRAPH Asia 2025, arXiv 2502.04762) — a
transformer/autoregressive **neural-L-system** over branch tokens that generates static 3D trees and supports
**image-to-tree, point-cloud-to-tree, and 4D growth** (growth stages as chronologically concatenated token
sequences). → Two uses: (i) a **generative structural prior** to hallucinate plausible missing branches
(our "漏检的枝"); (ii) a route to **procedurally diverse plant topologies** for dataset scale. Note: its "4D"
is *growth*, not wind/swaying.

**CropCraft** (arXiv 2411.09693) — recovers **complete organ-level structure** of crop canopies from field
images via **inverse procedural modeling**, explicitly to beat the occlusion that defeats generic 3D recon.
→ Species-aware, structure-first reconstruction analogous to StP recovery; a parametric-prior alternative for
occluded plants.

**3DBonsai** (arXiv 2504.01619) — **text→3D** structured woody plants via 3DGS **conditioned on an explicit
branch structure** (trainable space-colonization → structure → 3DGS priors). → Demonstrates the exact
"explicit skeleton drives Gaussian appearance" coupling we rely on; a text-to-structured-plant generator for
diversity.

## Area 2 — Image/Text → 3D & 3D-Gaussian generation

**TRELLIS** (arXiv 2412.01506, open) — **Structured LATent (SLAT)**: one latent decodes to **Radiance Fields,
3D Gaussians, or meshes**; conditioned on **text or image**. → Our current text→3DGS engine. Strength:
one-shot diverse assets. Weakness for us: thin stems / fine foliage are where feed-forward 3D generators
blur — a known failure mode motivating structure-aware priors (Area 1).

**LGM** (2402.05054), **GRM** (2403.14621), **Hunyuan3D 2.0** (2501.12202) — open feed-forward image/text→3D
(Gaussians or mesh) in seconds. → Faster/alternative generators to TRELLIS for the *static* stage; all share
the thin-structure/foliage fidelity limitation, so none removes the need for a plant structural prior.
*(These three were curtailed by a session-usage limit during verification; arXiv ids given for confirmation.)*

## Area 3 — Video generation & WORLD FOUNDATION MODELS  *(the priority)*

**Cosmos WFM Platform** (NVIDIA, arXiv 2501.03575, open weights, permissive license) — diffusion + autoregressive
**world foundation models** + video tokenizers + a data-curation pipeline, positioned as a customizable base
for **Physical AI**. → The open base to judge as a plant video/3D/4D engine.

**Cosmos-Transfer1** (arXiv 2503.14492, open) — **ControlNet-style conditional world generation** on Cosmos,
conditioned on **depth, segmentation, edges, LiDAR, and simulator renders (Sim2Real)**, with spatially adaptive
multimodal control. → *The single most direct answer to "condition video on 3D/structure."* We can render a
GaussianPlant / physics-simulated plant to depth+segmentation and get **photoreal, controllable, structure-
faithful plant video** — turning our low-fidelity physics renders into training-grade video.

**Cosmos-Predict2.5 / Transfer2.5** (arXiv 2511.00062, open) — latest generation: flow-based **Predict2.5**
unifying Text2World / Image2World / Video2World (2B and 14B, RL post-trained on 200M clips) + **Transfer2.5**
for Sim2Real/Real2Real from 3D spatial inputs. → Current SOTA open world model; the upgrade path for the above.

**Wan 2.1 / 2.2** (arXiv 2503.20314, open) — Alibaba's fully open video suite (T2V/I2V up to 14B; 2.2 adds MoE),
matching commercial systems. → Our current I2V for swaying clips; **strong natural-motion prior** and the base
many downstream controllers fine-tune. Good for *plausible* wind, but **not physically labeled/controllable**.

**Force Prompting** (arXiv 2505.19386, open) — ⭐⭐ fine-tunes a pretrained video diffusion model
(**CogVideoX-5B**) to obey **physics-based control signals: localized point forces and global wind force
fields**, with training data **synthesized in Blender** (force is a known label). The paper **explicitly pokes
a plant**. → The most on-target model in this survey. It (i) generates **swaying/poke plant video conditioned
on an explicit force**, exactly our hand-pull/wind setting; (ii) provides **force-labeled** synthetic data — a
template for generating our (video, contact-force) supervision without manual capture; (iii) shows a video
model *can* generalize a physical control signal.

**PhysCtrl** (arXiv 2509.20358) — a generative physics network that outputs **3D point trajectories conditioned
on explicit physical parameters** (material type, stiffness, forces), then drives video generation. → Answers
"condition motion on physics" at the trajectory level; a bridge between our physics params and video.

**Controllability / multi-view primitives.** **CameraCtrl** (2404.02101, open) — precise camera trajectories
for T2V; **CVD** (2405.17414) — epipolar cross-video sync → multi-camera-consistent motion; **SynCamMaster**
(2412.07760, open) — text→**synchronized multi-view** dynamic video (build a capture rig from scratch);
**ReCamMaster** (2503.11647, open) — re-render a **single** (monocular plant) video at **new camera
trajectories**. → Together these solve the *multi-view* half: either synthesize a multi-camera swaying rig, or
lift one monocular clip to many views — directly attacking our monocular depth ambiguity.

## Area 4 — 4D generation & physics-of-Gaussians

**PhysDreamer** (arXiv 2404.13026, ECCV'24, open) — ⭐ **plants are the headline use case.** Represents an
object as 3D Gaussians, learns a **spatially-varying Young's-modulus (stiffness) field** by **distilling motion
priors from a video-generation model**, then simulates elastic response to wind/forces. → The closest sibling
to our per-part stiffness fit — but it obtains material **without any force capture**, purely from a video-gen
prior. Strong candidate to **replace/augment our contact-force optimization** with distilled physics.

**PhysGaussian** (arXiv 2311.12198, CVPR'24, open) — treats 3DGS kernels as **MPM particles** with
deformation+stress → simulate-and-render simultaneously (elastic/plastic/granular/fluid). Foundational
physics-of-Gaussians. **Spring-Gaus** (arXiv 2403.09434, ECCV'24, open) — embeds a **spring-mass** system in
3DGS to **learn stiffness from multi-view video** of a moving object, then re-simulate — the direct ancestor of
our spring-mass approach. **OmniPhysGS** (arXiv 2501.18982, ICLR'25, open) — makes the constitutive law
learnable ("Constitutive Gaussians" blend expert material models in a differentiable MPM); **our repo already
builds on it**. **Physics3D** (arXiv 2406.04338, open) — viscoelastic material via video-diffusion distillation
(PhysDreamer cousin). → This cluster is our physics lineage; PhysDreamer/Physics3D add the "material from video
prior, no capture" trick worth importing.

**Video → 4D (dynamic Gaussians).** **DreamGaussian4D** (2312.17142, open) — image/video→4D (static 3DGS +
deformation field, video-diffusion supervised). **4D Gaussian Splatting** (hustvl/4DGaussians, CVPR'24, open) —
real-time dynamic-scene 4DGS. **CAT4D** (2411.18613, Google DeepMind) — ⭐ archetypal *no-manual-capture*:
monocular video → **multi-view synchronized video** → **deformable-3DGS 4D**; works on **generated** videos.
**4Diffusion** (2405.20674, NeurIPS'24, open) — motion module on a frozen 3D-aware diffusion model → 4D from a
single video. **SV4D** (2407.17470, Stability, on SVD) — multi-frame×multi-view image matrix → dynamic 3D;
trained on the curated **ObjaverseDy**. **4Real** (openreview SO1aRpwVLk) — drops multi-view-model dependence,
distills **real-video realism** → deformable GS. **GVFDiffusion** (2507.23785) — ⭐ diffuses a **canonical 3DGS
+ per-Gaussian Variation Field** (deformation over time) from one video — structurally ≈ our "static plant +
motion field." **Lyra** (2509.19296, NVIDIA, open) — ⭐ **self-distills a Cosmos-family video diffusion model
into feed-forward 3D/4D Gaussians**, no multi-view training data. → These convert generated swaying videos into
**4D Gaussian ground truth (multi-view + time)** — the supervision our monocular pipeline lacks.

> ⚠️ Domain caveat: 4Diffusion, SV4D, GVFDiffusion, CAT4D are trained on **Objaverse-style general objects**,
> not plants. They lift *motion→4D* well but do not themselves know botanical/foliage dynamics — they need a
> plant-aware video source (Force Prompting / Wan / Cosmos-Transfer conditioned on our structure).

---

## Synthesis 1 — A concrete pipeline to synthesize a 4D plant dataset (no manual capture)

```
 (A) Diverse static plants           (B) Controllable swaying          (C) Multi-view            (D) Lift to 4D
 ─────────────────────────           ──────────────────────────        ────────────────          ──────────────
 TRELLIS / 3DBonsai  ──▶  3DGS  ──▶  render RGB+depth+seg  ──▶  Cosmos-Transfer1  ──▶  photoreal   ──▶  CAT4D /
 (text→plant)              │          (our GaussianPlant / physics)     OR Force Prompting          swaying video    SV4D /
 + Autoregressive-Tree /   │                                          (force/wind-conditioned)         │            GVFDiffusion /
   Smart-Tree structure ───┘                                                                           │            Lyra
   prior (thin branches)          ├─ Wan 2.2 I2V for quick natural sway (no labels)                    │              │
                                  └─ Force Prompting for FORCE-LABELED sway (our supervision)          └──────────────┴──▶ 4D GS
        ReCamMaster / SynCamMaster / CVD  ──▶ multi-view of the swaying clip (kills monocular depth ambiguity)
```

- **Static diversity:** TRELLIS is in hand; add a **structural prior** (Autoregressive-Tree / Smart-Tree /
  3DBonsai) to fix thin-branch blur — the known failure mode for a *plant* dataset.
- **Motion, two tiers:** *fast & plausible* = Wan 2.2 I2V; *controllable & labeled* = **Force Prompting**
  (point-force / wind field) — this is what gives us (video ↔ force) pairs matching our physics fit.
- **Multi-view:** **ReCamMaster** (lift one clip to new cameras) or **SynCamMaster/CVD** (synthesize the rig)
  → provides the multi-view motion our monocular pipeline can't see (depth).
- **4D ground truth:** **CAT4D / SV4D / GVFDiffusion / Lyra** turn the (multi-view) swaying video into
  **dynamic Gaussians** = the 4D supervision label.

## Synthesis 2 — Is a Cosmos-like world model a viable engine? (verdict: yes, highest-leverage)

- **For video:** **Cosmos-Transfer1/2.5** is the key — it is **conditioned on depth/segmentation/edges/sim-
  renders**, so our **GaussianPlant + physics simulation renders directly become the control signal** →
  photoreal, structure-faithful, controllable plant video (Sim2Real). This is strictly more controllable than
  Wan (which only takes an image + text).
- **For 3D/4D:** **Lyra** proves a Cosmos-family model self-distills into **feed-forward 3D/4D Gaussians** with
  **no multi-view training data** — a route to turn Cosmos video into 4D plant assets.
- **Open & licensable:** Cosmos weights are open under a permissive license — usable and fine-tunable.
- **Caveat:** Cosmos has **no plant-specific training**; left unconditioned it will not respect fine botanical
  motion. The win is *conditioning it on our structure/physics*, not using it blind.

## Synthesis 3 — Extension points for OUR dynamic-plant pipeline

1. **Remove manual hand-pull capture → Force Prompting.** Generate force-labeled poke/wind plant videos; feed
   them to our per-scene physics fit as (video, contact-force) supervision. Closest single-paper match to our
   contact-force optimization; also a data-generation template (Blender force labels).
2. **Upgrade the video engine → Cosmos-Transfer1** (or keep Wan for speed). Condition on our GaussianPlant
   depth/segmentation render for controllable, photoreal, *structure-consistent* motion — and it naturally
   supports multi-view via camera control, attacking our monocular depth ambiguity.
3. **Material without capture → PhysDreamer / Physics3D.** Distill a per-part stiffness field from a video-gen
   prior instead of (or to initialize) our contact-force optimization; compare against our learned k/c.
4. **Structure prior for "missed / wrong branches" → Autoregressive-Tree / Smart-Tree / Masks-to-Skeleton.**
   Learned skeleton priors to seed StPr correction: hallucinate plausible missing thin branches; mask-based
   differentiable topology to prune false edges. Directly serves the StPr add/prune loop we scoped.
5. **Get 4D ground truth → CAT4D / SV4D / GVFDiffusion / Lyra.** Lift generated (multi-view) swaying videos to
   dynamic Gaussians → gives us **4D supervision** (multi-view + time) our monocular pipeline lacks, and a
   benchmark for our articulated-physics reconstruction.
6. **Multi-view supervision → ReCamMaster / SynCamMaster / CVD.** Synthesize multi-view of a single swaying
   clip → resolve the depth null-space that forced our synthetic-GT validation.

## Recommended next experiments (cheap → ambitious)

1. **Tonight/near-term (open, runnable):** TRELLIS text→3DGS diverse plants + Wan 2.2 I2V swaying = a first
   generated 4D-ish sample; render our newplant9 multi-view orbit as a control.
2. **High-value, medium effort:** stand up **Force Prompting** (CogVideoX-5B + released weights) → generate
   force-labeled plant sway; check whether our physics fit recovers the prescribed force.
3. **Strategic:** **Cosmos-Transfer1** conditioned on GaussianPlant depth/seg → controllable photoreal plant
   video; then **Lyra/CAT4D** to lift to 4D Gaussians → a scalable synthetic 4D plant dataset.

---
*Method: parallel multi-source web search → source fetch → 3-vote adversarial verification → synthesis
(2026-07-17). 36 sources, 78 verified findings. A session-usage limit truncated verification of a few Area-2/3
items (LGM/GRM/Hunyuan3D and some Cosmos votes); those are flagged and given by arXiv id for confirmation.*
