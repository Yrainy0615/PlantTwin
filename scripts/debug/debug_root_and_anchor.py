"""Overlay key indices (root, anchor, all branch nodes color-coded by depth)
on the canonical render through the COLMAP camera. Use this to confirm the
root is at the plant base and the anchor is where the hand grips."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image

from data.colmap_loader import (
    read_cameras_bin, read_images_bin, find_image_by_name, colmap_camera_to_renderer,
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
    parser.add_argument('--contact-bundle', default=None)
    parser.add_argument('--colmap-image', default='IMG_1388.JPG')
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=348)
    parser.add_argument('--out', default='outputs/debug_root_anchor.png')
    args = parser.parse_args()

    sparse = Path(args.source) / 'sparse' / '0'
    cams = read_cameras_bin(sparse / 'cameras.bin')
    imgs = read_images_bin(sparse / 'images.bin')
    rec = find_image_by_name(imgs, args.colmap_image)
    cam = colmap_camera_to_renderer(rec, cams[rec['cam_id']], args.H, args.W)

    scene = load_gaussian_plant_scene(args.source, args.output_dir)
    tree = root_branch_graph(scene.branch, scene.tube)

    print('Tree info:')
    print(f'  root_idx={tree.root_idx}, root_pos={tree.nodes[tree.root_idx].tolist()}')
    print(f'  branch nodes bbox: '
          f'x={[tree.nodes[:, 0].min().item(), tree.nodes[:, 0].max().item()]}, '
          f'y={[tree.nodes[:, 1].min().item(), tree.nodes[:, 1].max().item()]}, '
          f'z={[tree.nodes[:, 2].min().item(), tree.nodes[:, 2].max().item()]}')
    print(f'  max depth = {int(tree.depth.max().item())}')

    nodes_px = project(tree.nodes, cam, args.H, args.W)
    img = Image.open(Path(args.source) / 'images' / args.colmap_image).convert('RGB').resize((args.W, args.H), Image.LANCZOS)
    is_stem = (tree.edge_type == STEM).numpy()
    p_edge = project(tree.nodes[tree.edges_oriented[:, 0]], cam, args.H, args.W)
    q_edge = project(tree.nodes[tree.edges_oriented[:, 1]], cam, args.H, args.W)

    fig, ax = plt.subplots(1, 1, figsize=(args.W / 80.0, args.H / 80.0), dpi=80)
    ax.imshow(img)
    ax.set_xlim(0, args.W); ax.set_ylim(args.H, 0); ax.set_axis_off()

    for i in range(p_edge.shape[0]):
        c = '#ff7f00' if is_stem[i] else '#3aa757'
        lw = 1.4 if is_stem[i] else 0.7
        ax.plot([p_edge[i, 0], q_edge[i, 0]], [p_edge[i, 1], q_edge[i, 1]], color=c, linewidth=lw, alpha=0.85)

    # Color all branch nodes by depth (yellow=root, magenta=deep terminals)
    depths = tree.depth.float().numpy()
    sc = ax.scatter(nodes_px[:, 0], nodes_px[:, 1], c=depths, cmap='plasma', s=8, alpha=0.85)
    plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label='depth')

    # Root (big yellow star)
    root_px = nodes_px[tree.root_idx]
    ax.scatter([root_px[0]], [root_px[1]], c='yellow', marker='*', s=320, edgecolors='black', linewidths=1.0, label=f'root ({tree.root_idx})')

    # Anchor from contact bundle
    if args.contact_bundle is not None:
        bundle = torch.load(args.contact_bundle, map_location='cpu', weights_only=False)
        ac = bundle['anchor_node_id']
        valid = ac[ac >= 0]
        anchor_id = int(torch.bincount(valid).argmax().item())
        anchor_px = nodes_px[anchor_id]
        ax.scatter([anchor_px[0]], [anchor_px[1]], c='lime', marker='X', s=200, edgecolors='black', linewidths=0.8, label=f'anchor ({anchor_id})')

    ax.legend(loc='upper left', fontsize=8)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=80, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
