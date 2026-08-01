# scripts/

Run everything from the repo root with `PYTHONPATH=.` (or `python -m scripts.<pkg>.<name>`).
Most scripts default `--source /mnt/data/gaussianplant_data/newplant9` and
`--output-dir outputs/gsplant_output/newplant9`.

## `pipeline/` — per-scene motion → structure refinement (main workflow)

| Script | Role |
|---|---|
| `precompute_video_targets.py` | Build video-loss targets: SAM masks + RAFT flow (mask path falls back to a temporal-variance pseudo-mask if SAM isn't wired). |
| `track_video_cotracker.py` | CoTracker3 2D keypoint tracks (grid or explicit queries) for stem/leaf-tip supervision. |
| `precompute_contact.py` | Contact bundle: GroundedSAM hand mask + CoTracker3 + filter → per-frame anchor node, pin pixel, stem tracks. |
| `optimize_per_scene.py` | **v11** physics optimizer: articulated-chain rollout → leaf frames → LBS → render, fit per-part stiffness/damping + contact force against video. Frozen skeleton. |
| `fuse_motion_structure.py` | **Motion-in-the-loop**: jointly optimize node depth (δ) and soft topology (edge logits); depth from video RGB, connectivity from 3D rigidity strain. |
| `export_sim_ready.py` | Export a re-drivable sim-ready plant (cleaned tree + per-part params + contact trajectory). |

## `poc/` — structure-from-motion validation (synthetic GT)

| Script | Role |
|---|---|
| `synth_poc.py` (+`_viz`) | Perturb node depths, recover under static-only vs static+motion; reports depth RMSE. Shows **motion supplies depth**. |
| `synth_poc_topology.py` (+`_viz`) | Candidate edges = tree ∪ kNN; score by 3D rigidity under motion; ROC-AUC / prec@|E*|. Shows **motion supplies connectivity**. |

## `pretrain/` — feed-forward physics decoder (SDS)

`train_pretrain.py`, `train_part_aware_pretrain.py`, `train_sds.py`, `train_sds_e2e.py`,
`train_decoder.py`, `train_from_video.py` — the Stage-1 track that learns a plant→physics
network with video-diffusion SDS. See `docs/superpowers/specs/2026-05-27-part-aware-plant-pretrain.md`.

## `viz/` — diagnostics, renders, comparisons

Overlay/compare renders (`render_*`, `compare_struct_before_after.py`,
`strpr_overlay_chamfer.py`), progress viz (`densify_progress_viz.py`,
`joint_verdict_viz.py`, `visualize_*`, `viz_geom_containment.py`), and HTML report
generation (`make_struct_opt_html.py`).

## `datagen/` — text → 3DGS → video generation launchers

`run_full_generation.sh`, `run_gen_3dgs_parallel.sh`, `run_gen_video_parallel.sh`,
`run_gen_video_resilient.sh`, `check_video_quality.py`.

## `debug/` — alignment & detection

COLMAP alignment (`verify_colmap_alignment.py`, `sweep_colmap_alignment.py`), hand
detection (`detect_hand_grounded_sam.py`), root/anchor debugging (`debug_root_and_anchor.py`).

## `demo/` — re-drive demos

`interactive_demo.py`, `scripted_interactive_video.py`.

## `legacy/` — superseded exploration

PhysX-VLM (`test_physx_vlm*.py`, `vis_physx_voxel.py`) and DINO (`test_dino_plant.py`)
experiments, kept for reference. The DINO part-extraction path in
`models/structure/structure_extractor.py` is still a stub.

---

Tests live in `../tests/` (`test_struct_constraints.py` is a real 8-assertion unit test;
run `PYTHONPATH=. python tests/test_struct_constraints.py`).
