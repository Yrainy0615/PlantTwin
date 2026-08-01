"""Overlay the projected branch tree on the original COLMAP image to verify
that the COLMAP camera + the (StP + ApP) frame are in the same coordinate
system. If the branches sit on top of the woody parts of the plant in the
photo, we're aligned and can drop the default render camera in favor of this.

Saves a side-by-side PNG: left = original image (downsized) with projected
branch nodes + edges; right = the same image without the overlay.
"""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    parser.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    parser.add_argument('--image-name', default='IMG_1388.JPG')
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=512)
    parser.add_argument('--out', default='outputs/colmap_overlay.png')
    args = parser.parse_args()

    sparse_dir = Path(args.source) / 'sparse' / '0'
    cams = read_cameras_bin(sparse_dir / 'cameras.bin')
    imgs = read_images_bin(sparse_dir / 'images.bin')
    rec = find_image_by_name(imgs, args.image_name)
    if rec is None:
        raise SystemExit(f'{args.image_name} not in COLMAP images.bin')
    cam_rec = cams[rec['cam_id']]
    print(f'COLMAP cam: {cam_rec}')
    print(f'COLMAP image: q={rec["q"]}, t={rec["t"]}')

    scene = load_gaussian_plant_scene(args.source, args.output_dir)
    tree = root_branch_graph(scene.branch, scene.tube)

    src_img = Image.open(Path(args.source) / 'images' / args.image_name).convert('RGB')
    src_W0, src_H0 = src_img.size
    img_small = src_img.resize((args.W, args.H), Image.LANCZOS)
    cam = colmap_camera_to_renderer(rec, cam_rec, args.H, args.W)

    nodes_px = project(tree.nodes, cam, args.H, args.W)
    p = project(tree.nodes[tree.edges_oriented[:, 0]], cam, args.H, args.W)
    q = project(tree.nodes[tree.edges_oriented[:, 1]], cam, args.H, args.W)
    is_stem = (tree.edge_type == STEM).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(args.W * 2 / 80.0, args.H / 80.0), dpi=80)
    for ax in axes:
        ax.imshow(img_small)
        ax.set_xlim(0, args.W); ax.set_ylim(args.H, 0); ax.set_axis_off()
    for i in range(p.shape[0]):
        color = '#ff7f00' if is_stem[i] else '#3aa757'
        lw = 1.6 if is_stem[i] else 0.8
        axes[0].plot([p[i, 0], q[i, 0]], [p[i, 1], q[i, 1]],
                     color=color, linewidth=lw, alpha=0.9)
    axes[0].scatter(nodes_px[:, 0], nodes_px[:, 1], c='yellow', s=4, alpha=0.6)
    axes[0].set_title(f'projected branches over {args.image_name}', fontsize=8)
    axes[1].set_title('original', fontsize=8)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=80, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
