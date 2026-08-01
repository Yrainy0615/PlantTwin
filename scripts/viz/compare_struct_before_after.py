"""Before/after structure comparison: original branch tree vs densified + delta tree.

Left  : original RootedBranchTree (N nodes) projected over the canonical render.
Right : densified tree with learned delta_rest_pos applied (nodes_eff = nodes + delta);
        newly inserted midpoint nodes and the edges that were split are highlighted.

Both panels share the same rendered canonical frame (the static-background ApPs),
so the only visual difference is the structure overlay.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.colmap_loader import (
    read_cameras_bin, read_images_bin, find_image_by_name, colmap_camera_to_renderer,
)
from data.gaussian_plant_loader import load_gaussian_plant_scene
from models.structure.graph_cleanup import root_branch_graph, RootedBranchTree


def project(points: torch.Tensor, camera: dict, H: int, W: int) -> np.ndarray:
    P = camera['proj_matrix'].T
    homog = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=-1)
    clip = (P @ homog.T).T
    w = clip[:, 3:4].clamp_min(1e-6)
    ndc = clip[:, :3] / w
    u = (ndc[:, 0] * 0.5 + 0.5) * W
    v = (ndc[:, 1] * 0.5 + 0.5) * H
    return torch.stack([u, v], dim=-1).detach().cpu().numpy()


def tree_from_densified(dt: dict) -> RootedBranchTree:
    return RootedBranchTree(
        nodes=dt['nodes'].cpu(), root_idx=int(dt['root_idx']),
        parent=dt['parent'].cpu(), depth=dt['depth'].cpu(),
        edges_oriented=dt['edges_oriented'].cpu(), edge_length=dt['edge_length'].cpu(),
        edge_radius=dt['edge_radius'].cpu(), edge_type=dt['edge_type'].cpu(),
        subtree_size=dt['subtree_size'].cpu(),
    )


def draw_tree(ax, uv, edges, *, node_color, edge_color, node_size, edge_lw,
              highlight_nodes=None, highlight_uv=None, hl_color='#ff3030', hl_size=70):
    for e in edges:
        a, b = int(e[0]), int(e[1])
        ax.plot([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]],
                color=edge_color, lw=edge_lw, alpha=0.7, zorder=1)
    ax.scatter(uv[:, 0], uv[:, 1], s=node_size, c=node_color, alpha=0.9,
               edgecolors='white', linewidths=0.3, zorder=2)
    if highlight_nodes is not None and len(highlight_nodes) > 0:
        hu = uv[highlight_nodes]
        ax.scatter(hu[:, 0], hu[:, 1], s=hl_size, c=hl_color, marker='*',
                   edgecolors='white', linewidths=0.6, zorder=3, label='new node')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    parser.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    parser.add_argument('--params', default='outputs/per_scene_optim/newplant9_v13_dens/final_params.pt',
                        help='Densified + delta checkpoint (after structure-in-the-loop).')
    parser.add_argument('--colmap-image', default='IMG_1388.JPG')
    parser.add_argument('--H', type=int, default=1088)
    parser.add_argument('--W', type=int, default=736)
    parser.add_argument('--skin-max-dist', type=float, default=0.3)
    parser.add_argument('--out', default='outputs/per_scene_optim/struct_before_after.png')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    scene = load_gaussian_plant_scene(args.source, args.output_dir)

    # Original tree
    tree0 = root_branch_graph(scene.branch, scene.tube)
    N0 = tree0.nodes.shape[0]

    p = torch.load(args.params, map_location='cpu', weights_only=False)
    if 'densified_tree' not in p:
        raise SystemExit('params has no densified_tree — pass a v13-style densified checkpoint.')
    tree1 = tree_from_densified(p['densified_tree'])
    delta = p.get('delta_rest_pos', torch.zeros_like(tree1.nodes))
    new_node_idx = np.arange(N0, tree1.nodes.shape[0])  # midpoints appended after originals
    nodes1_eff = tree1.nodes + delta

    # Camera
    sparse = Path(args.source) / 'sparse' / '0'
    cams = read_cameras_bin(sparse / 'cameras.bin')
    imgs = read_images_bin(sparse / 'images.bin')
    rec = find_image_by_name(imgs, args.colmap_image)
    cam = colmap_camera_to_renderer(rec, cams[rec['cam_id']], args.H, args.W)
    cam = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cam.items()}

    # Render canonical background (static ApPs) once — same image for both panels.
    from models.renderer.gaussian_renderer import GaussianRenderer
    renderer = GaussianRenderer(image_height=args.H, image_width=args.W, sh_degree=0).to(device)
    frame = renderer.render_frame(
        scene.app.xyz.to(device), scene.app.scales.to(device), scene.app.rots.to(device),
        scene.app.opacities.to(device), scene.app.colors.to(device), cam, shs=None,
    ).clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()

    uv0 = project(tree0.nodes.to(device), cam, args.H, args.W)
    uv1 = project(nodes1_eff.to(device), cam, args.H, args.W)

    delta_norm = delta.norm(dim=-1).cpu().numpy()
    p95 = float(np.percentile(delta_norm, 95)) + 1e-8

    fig, axes = plt.subplots(1, 2, figsize=(2 * args.W / 90.0, args.H / 90.0), dpi=90)

    # Left: original tree
    axes[0].imshow(frame)
    axes[0].set_axis_off()
    draw_tree(axes[0], uv0, tree0.edges_oriented.cpu().numpy(),
              node_color='#33c0ff', edge_color='#1f7fb0', node_size=14, edge_lw=1.0)
    axes[0].set_title(f'Before — original tree (N={N0})', color='black', fontsize=14)

    # Right: densified + delta tree, nodes sized by |delta|, new nodes starred
    axes[1].imshow(frame)
    axes[1].set_axis_off()
    sizes = 14 + (delta_norm / p95).clip(0, 2.5) * 60
    draw_tree(axes[1], uv1, tree1.edges_oriented.cpu().numpy(),
              node_color='#ff9b30', edge_color='#b06010', node_size=sizes, edge_lw=1.0,
              highlight_nodes=new_node_idx, highlight_uv=uv1)
    # delta arrows: only the top-K largest, at true scale, capped in pixel length.
    amp = 3.0
    top_k_arrows = 10
    max_px = 90.0
    uv1_tip = project((nodes1_eff + delta * amp).to(device), cam, args.H, args.W)
    order = np.argsort(delta_norm[:N0])[::-1][:top_k_arrows]
    for i in order:
        v = uv1_tip[i] - uv1[i]
        L = float(np.hypot(*v))
        if L > max_px:
            v = v * (max_px / L)
        axes[1].annotate('', xy=(uv1[i, 0] + v[0], uv1[i, 1] + v[1]),
                         xytext=(uv1[i, 0], uv1[i, 1]),
                         arrowprops=dict(arrowstyle='->', color='#ff3030', lw=1.6, alpha=0.9))
    axes[1].set_title(
        f'After — densified + δ (N={tree1.nodes.shape[0]}, +{len(new_node_idx)} nodes, '
        f'max|δ|={delta_norm.max()*100:.1f}cm; top-{top_k_arrows} δ arrows x{amp:.0f})',
        color='black', fontsize=13)
    axes[1].scatter([], [], s=80, c='#ff3030', marker='*', label='inserted midpoint node')
    axes[1].plot([], [], color='#ff3030', lw=1.6, label='top-δ direction')
    axes[1].legend(loc='lower right', fontsize=10, framealpha=0.6)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1, dpi=110)
    plt.close(fig)
    print(f'wrote {out_path}  ({N0} -> {tree1.nodes.shape[0]} nodes, '
          f'{len(new_node_idx)} inserted)')


if __name__ == '__main__':
    main()
