"""Visualize the Stage 0 output: rooted branch tree + leaf clusters + petiole attachments.

Renders three matplotlib views (front / top / side) to a single PNG so attachment
quality can be eyeballed before moving on to dynamics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb

from data.gaussian_plant_loader import load_gaussian_plant_scene
from models.structure.graph_cleanup import root_branch_graph, STEM
from models.structure.leaf_attachment import infer_leaf_attachments


def _leaf_palette(n: int) -> np.ndarray:
    h = np.linspace(0.0, 1.0, n, endpoint=False)
    s = np.full(n, 0.85)
    v = np.full(n, 0.85)
    return hsv_to_rgb(np.stack([h, s, v], -1))


def _draw_one_view(ax, scene, tree, attachments, leaf_colors, elev: float, azim: float, title: str):
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    ax.set_axis_off()

    nodes = tree.nodes.numpy()
    # Branch edges
    p = tree.nodes[tree.edges_oriented[:, 0]].numpy()
    q = tree.nodes[tree.edges_oriented[:, 1]].numpy()
    is_stem = (tree.edge_type == STEM).numpy()
    for i in range(len(p)):
        color = '#7a4a1f' if is_stem[i] else '#a07b4a'
        lw = 2.0 if is_stem[i] else 1.0
        ax.plot([p[i, 0], q[i, 0]], [p[i, 1], q[i, 1]], [p[i, 2], q[i, 2]],
                color=color, linewidth=lw, alpha=0.9)

    # Leaf clusters
    for k, leaf in enumerate(scene.leaves):
        pts = leaf.points.numpy()
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.0, c=[leaf_colors[k]], alpha=0.5)

    # Petiole edges (surface_point → disk_center) and disk centers
    for k, att in enumerate(attachments):
        sp = att.surface_point.numpy()
        dc = att.disk_center.numpy()
        ax.plot([sp[0], dc[0]], [sp[1], dc[1]], [sp[2], dc[2]],
                color=leaf_colors[k], linewidth=1.3, alpha=0.9)
        ax.scatter([sp[0]], [sp[1]], [sp[2]], color=leaf_colors[k], s=14, edgecolors='black', linewidths=0.4)

    # Root marker
    root = nodes[tree.root_idx]
    ax.scatter([root[0]], [root[1]], [root[2]], color='red', s=40, marker='^', label='root')


def visualize(scene_dir: str, out_dir: str, output_path: Path, root_idx: int | None):
    scene = load_gaussian_plant_scene(scene_dir, out_dir)
    tree = root_branch_graph(scene.branch, scene.tube, root_idx=root_idx)
    attachments = infer_leaf_attachments(scene.leaves, tree)
    leaf_colors = _leaf_palette(len(scene.leaves))

    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    _draw_one_view(ax1, scene, tree, attachments, leaf_colors, elev=20, azim=-60, title='perspective')
    _draw_one_view(ax2, scene, tree, attachments, leaf_colors, elev=90, azim=-90, title='top-down')
    _draw_one_view(ax3, scene, tree, attachments, leaf_colors, elev=0, azim=-90, title='front')

    fig.suptitle(f'Stage 0 topology — {scene.name}    '
                 f'(branch nodes={tree.nodes.shape[0]}, leaves={len(attachments)}, '
                 f'unique parent edges={len(set(a.parent_edge_idx for a in attachments))})')
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    print(f'wrote {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    parser.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    parser.add_argument('--out', default='outputs/viz_topology_newplant9.png')
    parser.add_argument('--root', type=int, default=None)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    visualize(args.source, args.output_dir, out_path, args.root)
