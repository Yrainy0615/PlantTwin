"""Headless replay of the interactive demo: scripts a sequence of "drag" forces
at random branch nodes, runs the StreamingPlantSim, and writes a side-by-side
video (rendered rollout + force-anchor overlay).

Used to record an offline preview of the interactive UX without needing a
browser session.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter

from data.colmap_loader import (
    read_cameras_bin, read_images_bin, find_image_by_name, colmap_camera_to_renderer,
)
from data.gaussian_plant_loader import load_gaussian_plant_scene
from models.structure.graph_cleanup import root_branch_graph
from models.structure.leaf_attachment import infer_leaf_attachments
from models.structure.skinning import compute_skinning
from simulation.streaming_sim import StreamingPlantSim


def _pick_drag_nodes(tree, n_nodes: int, rng: random.Random) -> list[int]:
    """Pick a handful of branch nodes (degree-1 terminals are most fun)."""
    # Use mid-to-deep nodes; avoid leaves themselves and the root.
    depths = tree.depth.tolist()
    max_d = max(depths)
    candidates = [i for i, d in enumerate(depths) if d >= max_d * 0.35 and d <= max_d * 0.9]
    if len(candidates) < n_nodes:
        candidates = list(range(tree.nodes.shape[0]))
        candidates.remove(tree.root_idx)
    return rng.sample(candidates, k=min(n_nodes, len(candidates)))


def _drag_direction(rng: random.Random) -> np.ndarray:
    """Random in-plane direction (avoid pure-vertical so the motion reads on screen)."""
    azimuth = rng.uniform(0.0, 2 * math.pi)
    # mostly horizontal with a slight tilt
    horiz = np.array([math.cos(azimuth), 0.0, math.sin(azimuth)])
    tilt = np.array([0.0, rng.uniform(-0.25, 0.25), 0.0])
    v = horiz + tilt
    return v / (np.linalg.norm(v) + 1e-8)


def project_world_to_pixels(points_world: torch.Tensor, camera: dict, H: int, W: int) -> torch.Tensor:
    P = camera['proj_matrix'].T
    homog = torch.cat([points_world, torch.ones(points_world.shape[0], 1, device=points_world.device)], dim=-1)
    clip = (P @ homog.T).T
    w = clip[:, 3:4].clamp_min(1e-6)
    ndc = clip[:, :3] / w
    u = (ndc[:, 0] * 0.5 + 0.5) * W
    v = (ndc[:, 1] * 0.5 + 0.5) * H
    return torch.stack([u, v], dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    parser.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    parser.add_argument('--params', required=True)
    parser.add_argument('--colmap-image', default='IMG_1388.JPG')
    parser.add_argument('--dt', type=float, default=0.02)
    parser.add_argument('--substeps', type=int, default=2)
    parser.add_argument('--n-drags', type=int, default=4)
    parser.add_argument('--drag-frames', type=int, default=18,
                        help='Per-drag: hold force for this many display frames.')
    parser.add_argument('--release-frames', type=int, default=42,
                        help='Per-drag: settle (no force) for this many frames before next drag.')
    parser.add_argument('--force-mag', type=float, default=3.5,
                        help='Force magnitude in N at the chosen node during a drag.')
    parser.add_argument('--ramp-frames', type=int, default=4,
                        help='Linear ramp-up frames at start of each drag (avoids impulse spikes).')
    parser.add_argument('--H', type=int, default=1088)
    parser.add_argument('--W', type=int, default=736)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skin-max-dist', type=float, default=0.3)
    parser.add_argument('--out', default='outputs/per_scene_optim/scripted_interactive.mp4')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    rng = random.Random(args.seed)

    print('Loading scene...')
    scene = load_gaussian_plant_scene(args.source, args.output_dir)
    tree = root_branch_graph(scene.branch, scene.tube)
    attachments = infer_leaf_attachments(scene.leaves, tree)
    skinning = compute_skinning(scene.app.xyz, tree, attachments, max_dist=args.skin_max_dist)

    params = torch.load(args.params, map_location='cpu', weights_only=False)
    sim = StreamingPlantSim(tree, attachments, skinning, params, scene.app.xyz, device)

    # COLMAP camera
    sparse = Path(args.source) / 'sparse' / '0'
    cams = read_cameras_bin(sparse / 'cameras.bin')
    imgs = read_images_bin(sparse / 'images.bin')
    rec = find_image_by_name(imgs, args.colmap_image)
    if rec is None:
        raise SystemExit(f'{args.colmap_image} not in COLMAP images.bin')
    cam = colmap_camera_to_renderer(rec, cams[rec['cam_id']], args.H, args.W)
    cam = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cam.items()}

    # Renderer
    from models.renderer.gaussian_renderer import GaussianRenderer
    renderer = GaussianRenderer(image_height=args.H, image_width=args.W, sh_degree=0).to(device)

    scales = scene.app.scales.to(device)
    rotations = scene.app.rots.to(device)
    opacities = scene.app.opacities.to(device)
    colors = scene.app.colors.to(device)

    # Build drag schedule
    drag_nodes = _pick_drag_nodes(tree, args.n_drags, rng)
    print(f'drag schedule: nodes={drag_nodes}')
    schedule = []  # list of (node_id, dir_unit_world) one entry per drag
    for nid in drag_nodes:
        d = _drag_direction(rng)
        schedule.append((nid, d))

    n_per = args.drag_frames + args.release_frames
    T = len(schedule) * n_per

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(args.W / 80.0, args.H / 80.0), dpi=80)
    writer = FFMpegWriter(fps=args.fps)

    substep_dt = args.dt / args.substeps

    print(f'Recording {T} frames @ {args.fps} fps, sim substeps={args.substeps}...')
    with writer.saving(fig, str(out_path), dpi=80):
        for t in range(T):
            drag_idx = t // n_per
            local_t = t % n_per
            node_id, direction = schedule[drag_idx]

            if local_t < args.drag_frames:
                ramp = min(1.0, local_t / max(1, args.ramp_frames))
                mag = args.force_mag * ramp
            else:
                mag = 0.0

            f = torch.zeros(sim.N, 3)
            f[node_id] = torch.as_tensor(direction * mag, dtype=torch.float32)

            for _ in range(args.substeps):
                ap_xyz, pos, rot = sim.step(substep_dt, f)

            # Render this display frame
            frame = renderer.render_frame(ap_xyz, scales, rotations, opacities, colors, cam, shs=None)
            img = frame.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()

            ax.clear()
            ax.imshow(img)
            ax.set_axis_off()

            # Overlay: project the active anchor; mark with circle + arrow showing drag direction
            if mag > 0:
                pos_gpu = pos[node_id:node_id + 1].to(device)
                uv = project_world_to_pixels(pos_gpu, cam, args.H, args.W).cpu().numpy()[0]
                ax.scatter([uv[0]], [uv[1]], s=200, facecolors='none', edgecolors='#ff3030', linewidths=2.5)
                # Project endpoint of arrow (small offset in world along drag direction)
                arrow_len_world = 0.04
                end_world = pos[node_id:node_id + 1] + torch.as_tensor(direction * arrow_len_world,
                                                                        dtype=torch.float32)
                uv_end = project_world_to_pixels(end_world.to(device), cam, args.H, args.W).cpu().numpy()[0]
                ax.annotate(
                    '', xy=(uv_end[0], uv_end[1]), xytext=(uv[0], uv[1]),
                    arrowprops=dict(arrowstyle='->', color='#ff3030', lw=2.5),
                )
                ax.text(
                    uv[0] + 12, uv[1] - 12, f'node {node_id}',
                    color='#ff3030', fontsize=9, weight='bold',
                )

            ax.text(
                12, 24, f'drag {drag_idx + 1}/{len(schedule)}    frame {t + 1}/{T}',
                color='white', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.5),
            )
            writer.grab_frame()
    plt.close(fig)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
