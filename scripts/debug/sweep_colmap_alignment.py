"""Project branch nodes under several world->camera conventions onto
IMG_1388.JPG to figure out which one (if any) lines up with the photo."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image

from data.colmap_loader import (
    read_cameras_bin,
    read_images_bin,
    find_image_by_name,
    colmap_camera_to_renderer,
)
from data.gaussian_plant_loader import load_gaussian_plant_scene
from models.structure.graph_cleanup import root_branch_graph, STEM


def project(points_world: torch.Tensor, cam: dict, H: int, W: int) -> torch.Tensor:
    P = cam['proj_matrix'].T
    homog = torch.cat([points_world, torch.ones(points_world.shape[0], 1)], dim=-1)
    clip = (P @ homog.T).T
    w = clip[:, 3:4].clamp_min(1e-6)
    ndc = clip[:, :3] / w
    u = (ndc[:, 0] * 0.5 + 0.5) * W
    v = (ndc[:, 1] * 0.5 + 0.5) * H
    return torch.stack([u, v], dim=-1)


import math


def _Rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# Each variant: a camera-frame roll applied on TOP of the COLMAP (R, t).
# Effective transform: P_cam = (roll @ R) @ P_world + (roll @ t)
ROLLS = {
    'as-is (0°)':      torch.eye(3, dtype=torch.float32),
    'roll +90° (CCW)': _Rz(90),
    'roll -90° (CW)':  _Rz(-90),
    'roll 180°':       _Rz(180),
}
# Keep a few world-axis variants too for completeness
VARIANTS = ROLLS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    parser.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    parser.add_argument('--image-name', default='IMG_1388.JPG')
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=348)
    parser.add_argument('--out', default='outputs/colmap_overlay_sweep.png')
    args = parser.parse_args()

    sparse_dir = Path(args.source) / 'sparse' / '0'
    cams = read_cameras_bin(sparse_dir / 'cameras.bin')
    imgs = read_images_bin(sparse_dir / 'images.bin')
    rec = find_image_by_name(imgs, args.image_name)
    cam_rec = cams[rec['cam_id']]
    base_cam = colmap_camera_to_renderer(rec, cam_rec, args.H, args.W)

    scene = load_gaussian_plant_scene(args.source, args.output_dir)
    tree = root_branch_graph(scene.branch, scene.tube)

    img = Image.open(Path(args.source) / 'images' / args.image_name).convert('RGB').resize((args.W, args.H), Image.LANCZOS)

    n = len(VARIANTS)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(args.W * cols / 80.0, args.H * rows / 80.0), dpi=80)
    axes = axes.flatten()
    is_stem = (tree.edge_type == STEM).numpy()
    for k, (name, roll) in enumerate(VARIANTS.items()):
        ax = axes[k]
        ax.imshow(img)
        ax.set_xlim(0, args.W); ax.set_ylim(args.H, 0); ax.set_axis_off()
        # Build a rolled camera by applying `roll` on top of the COLMAP (R, t)
        from data.colmap_loader import quat_to_R
        R_cm = quat_to_R(rec['q'])
        t_cm = torch.tensor(rec['t'], dtype=torch.float32)
        R = roll @ R_cm
        t = roll @ t_cm
        # Re-pack into the renderer camera using the same intrinsics as base_cam
        view = torch.eye(4); view[:3, :3] = R; view[:3, 3] = t
        proj_no_view = base_cam['proj_matrix'].T @ torch.inverse(torch.eye(4).T @ torch.eye(4))  # noop, just for clarity
        # The cleanest: reconstruct proj as (perspective @ view) directly using base_cam's perspective.
        # Recover perspective from base_cam by view_inv * proj_T.T
        view_base_T = base_cam['view_matrix']
        view_base = view_base_T.T
        proj_base = base_cam['proj_matrix'].T
        perspective = proj_base @ torch.inverse(view_base)
        proj = perspective @ view
        cam = {
            'view_matrix': view.T.contiguous(),
            'proj_matrix': proj.T.contiguous(),
        }
        pw = tree.nodes
        nodes_px = project(pw, cam, args.H, args.W)
        p = project(tree.nodes[tree.edges_oriented[:, 0]], cam, args.H, args.W)
        q = project(tree.nodes[tree.edges_oriented[:, 1]], cam, args.H, args.W)
        for i in range(p.shape[0]):
            c = '#ff7f00' if is_stem[i] else '#3aa757'
            lw = 1.4 if is_stem[i] else 0.7
            ax.plot([p[i, 0], q[i, 0]], [p[i, 1], q[i, 1]], color=c, linewidth=lw, alpha=0.85)
        ax.scatter(nodes_px[:, 0], nodes_px[:, 1], c='yellow', s=3, alpha=0.5)
        ax.set_title(name, fontsize=10)
    for k in range(n, rows * cols):
        axes[k].set_axis_off()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=80, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
