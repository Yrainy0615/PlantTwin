# Paper List — Generative / Reconstructive Models for a 4D Plant Dataset

Research target: build a **4D plant dataset (multi-view + swaying/wind motion) without manual video capture**,
and find extension points for our dynamic-plant pipeline (GaussianPlant static reconstruction →
monocular motion → articulated physics + StPr structure correction).

Legend: 🔓 open weights/code · 🌿 plant/vegetation shown · ⭐ most relevant to us

See [`survey.md`](survey.md) for per-paper abstracts + relevance + the synthesis (usable models,
Cosmos verdict, extension points, and a concrete generation plan).

---

## Area 1 — Plant / tree reconstruction & neural-procedural structure

| Paper | id / link | open | one-line relevance |
|---|---|---|---|
| **GaussianPlant** — Structure-aligned 3DGS for plants | [2512.14087](https://arxiv.org/abs/2512.14087) | — | 🌿⭐ our base: StPr (branch=cylinder, leaf=disk) + bound ApPs from multi-view images |
| **Masks-to-Skeleton** — multi-view mask → tree skeleton (Sensors'25) | [10.3390/s25144354](https://doi.org/10.3390/s25144354) | — | 🌿 differentiable node+adjacency fit to multi-view silhouettes (better skeleton/topology) |
| **Smart-Tree** — neural medial-axis skeleton from point cloud | [2303.11560](https://arxiv.org/abs/2303.11560) | 🔓 | 🌿 sparse-voxel CNN → per-point radius+direction → branch skeleton; open code+data |
| **Autoregressive Static & Growing Trees** (SIGGRAPH Asia'25) | [2502.04762](https://arxiv.org/abs/2502.04762) | — | 🌿⭐ transformer neural-L-system; image/pointcloud→tree + **4D growth** |
| **CropCraft** — inverse procedural crop reconstruction | [2411.09693](https://arxiv.org/abs/2411.09693) | — | 🌿 organ-level parametric plant fit under heavy occlusion |
| **3DBonsai** — structure-aware bonsai via conditioned 3DGS | [2504.01619](https://arxiv.org/abs/2504.01619) | — | 🌿 text→structured woody plant; couples explicit skeleton + 3DGS (parallel to StPr) |

## Area 2 — Image/Text → 3D & 3D-Gaussian generation

| Paper | id / link | open | one-line relevance |
|---|---|---|---|
| **TRELLIS** — Structured LATent (SLAT), text/img→3D | [2412.01506](https://arxiv.org/abs/2412.01506) | 🔓 | ⭐ our current text→3DGS engine; one latent → RF / 3DGS / mesh |
| **LGM** — Large Gaussian Model (feed-forward multiview→3DGS) | [2402.05054](https://arxiv.org/abs/2402.05054) | 🔓 | fast img/text→3DGS; thin-structure fidelity limited |
| **GRM** — feed-forward transformer, sparse-view→3DGS | [2403.14621](https://arxiv.org/abs/2403.14621) | 🔓 | ~seconds sparse-view→Gaussians; text/image-to-3D |
| **Hunyuan3D 2.0** — shape + texture two-stage | [2501.12202](https://arxiv.org/abs/2501.12202) | 🔓 | open img→3D; mesh-centric, foliage detail limited |

## Area 3 — Video generation & WORLD FOUNDATION MODELS (priority)

| Paper | id / link | open | one-line relevance |
|---|---|---|---|
| **Cosmos WFM Platform** (NVIDIA) | [2501.03575](https://arxiv.org/abs/2501.03575) | 🔓 | ⭐ open world-foundation base (diffusion+AR), tokenizers, Physical-AI focus |
| **Cosmos-Transfer1** — conditional world gen | [2503.14492](https://arxiv.org/abs/2503.14492) | 🔓 | 🌿⭐ **depth/seg/edge/LiDAR-conditioned** → render GaussianPlant → controllable video (Sim2Real) |
| **Cosmos-Predict2.5 / Transfer2.5** | [2511.00062](https://arxiv.org/abs/2511.00062) | 🔓 | latest Cosmos: unified T2/I2/V2World (2B/14B) + Sim2Real from 3D inputs |
| **Wan 2.1 / 2.2** — open large video models | [2503.20314](https://arxiv.org/abs/2503.20314) | 🔓 | ⭐ strong natural-motion prior (swaying/wind); our current I2V; Wan2.2 MoE |
| **Force Prompting** — physics control signals in video gen | [2505.19386](https://arxiv.org/abs/2505.19386) | 🔓 | 🌿⭐⭐ **point force + wind field control** on CogVideoX-5B; **pokes a plant**; force-labeled synthetic data |
| **PhysCtrl** — generative physics for controllable video | [2509.20358](https://arxiv.org/abs/2509.20358) | — | ⭐ physics params (material/stiffness/force) → 3D trajectories → video |
| **CameraCtrl** — camera control for T2V | [2404.02101](https://arxiv.org/abs/2404.02101) | 🔓 | controllability primitive for orbit / multi-view plant clips |
| **CVD** — Collaborative Video Diffusion (multi-view consistent) | [2405.17414](https://arxiv.org/abs/2405.17414) | — | epipolar cross-video sync → multi-camera consistent motion |
| **SynCamMaster** — synchronized multi-camera T2V | [2412.07760](https://arxiv.org/abs/2412.07760) | 🔓 | text→multi-view synchronized dynamic scene (build capture rigs from scratch) |
| **ReCamMaster** — re-render single video at new cameras | [2503.11647](https://arxiv.org/abs/2503.11647) | 🔓 | monocular plant clip → novel camera trajectories (mono→multi-view) |

## Area 4 — 4D generation & physics-of-Gaussians

| Paper | id / link | open | one-line relevance |
|---|---|---|---|
| **PhysDreamer** — physics interaction via video prior (ECCV'24) | [2404.13026](https://arxiv.org/abs/2404.13026) | 🔓 | 🌿⭐ **plants headline**; stiffness field distilled from video-gen prior, no force capture |
| **PhysGaussian** — MPM on 3D Gaussians (CVPR'24) | [2311.12198](https://arxiv.org/abs/2311.12198) | 🔓 | foundational sim-of-Gaussians (elastic/plastic/granular/fluid) |
| **Spring-Gaus** — spring-mass 3D Gaussians (ECCV'24) | [2403.09434](https://arxiv.org/abs/2403.09434) | 🔓 | learn stiffness from multi-view video of moving object → re-simulate (our lineage) |
| **OmniPhysGS** — learnable constitutive Gaussians (ICLR'25) | [2501.18982](https://arxiv.org/abs/2501.18982) | 🔓 | 🌿 general material law in diff-MPM; our repo builds on it |
| **Physics3D** — viscoelastic material via video diffusion | [2406.04338](https://arxiv.org/abs/2406.04338) | 🔓 | material inference cousin of PhysDreamer |
| **DreamGaussian4D** — generative 4D Gaussians | [2312.17142](https://arxiv.org/abs/2312.17142) | 🔓 | image/video→4D (static 3DGS + deformation field) |
| **4D Gaussian Splatting (4DGS)** (CVPR'24) | [github hustvl/4DGaussians](https://github.com/hustvl/4DGaussians) | 🔓 | real-time dynamic-scene 4DGS representation |
| **CAT4D** — monocular→multi-view→4D (Google DeepMind) | [2411.18613](https://arxiv.org/abs/2411.18613) | — | ⭐ archetypal "no manual capture": mono video → multi-view → deformable-3DGS 4D |
| **4Diffusion** — multi-view video diffusion for 4D (NeurIPS'24) | [2405.20674](https://arxiv.org/abs/2405.20674) | 🔓 | motion module on frozen 3D-aware diffusion → 4D from one video |
| **SV4D** — video-to-4D on SVD (Stability) | [2407.17470](https://arxiv.org/abs/2407.17470) | 🔓 | multi-frame×multi-view image matrix → dynamic 3D |
| **4Real** — photorealistic 4D via video diffusion | [openreview SO1aRpwVLk](https://openreview.net/forum?id=SO1aRpwVLk) | — | drops multi-view-model dependence; distills real-video realism → deformable GS |
| **GVFDiffusion** — Gaussian Variation Field, video→4D | [2507.23785](https://arxiv.org/abs/2507.23785) | — | ⭐ canonical 3DGS + per-Gaussian variation field (≈ our static-plant + motion-field) |
| **Lyra** — Cosmos self-distilled → 3D/4D Gaussians (NVIDIA) | [2509.19296](https://arxiv.org/abs/2509.19296) | 🔓 | ⭐ turns a Cosmos-family video model into feed-forward 3D/4D GS, no multi-view training |

---

*Verified via multi-source fan-out + adversarial verification (2026-07-17). Some Area-2 rows
(LGM/GRM/Hunyuan3D) hit a session-usage limit during verification and are listed from known metadata;
arXiv ids provided for confirmation. Full analysis in [`survey.md`](survey.md).*
