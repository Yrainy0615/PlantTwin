"""Side-by-side containment check: unconstrained vs geom-constrained structure.

For each checkpoint, project the densified+delta nodes over the canonical render and
color each node by whether it sits inside its branch band (green) or has drifted out
(red, sized by how far). Title reports the out-of-branch count. Makes the effect of
the geometric containment constraint directly visible.
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
from optimization.struct_constraints import (
    subsample_points, estimate_geom_margin, rest_relative_tolerance, nearest_branch_distance,
)


def project(points, camera, H, W):
    P = camera['proj_matrix'].T
    homog = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=-1)
    clip = (P @ homog.T).T
    w = clip[:, 3:4].clamp_min(1e-6)
    ndc = clip[:, :3] / w
    u = (ndc[:, 0] * 0.5 + 0.5) * W
    v = (ndc[:, 1] * 0.5 + 0.5) * H
    return torch.stack([u, v], dim=-1).detach().cpu().numpy()


def tree_from_densified(dt):
    return RootedBranchTree(
        nodes=dt['nodes'].cpu(), root_idx=int(dt['root_idx']), parent=dt['parent'].cpu(),
        depth=dt['depth'].cpu(), edges_oriented=dt['edges_oriented'].cpu(),
        edge_length=dt['edge_length'].cpu(), edge_radius=dt['edge_radius'].cpu(),
        edge_type=dt['edge_type'].cpu(), subtree_size=dt['subtree_size'].cpu(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    ap.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    ap.add_argument('--params', nargs=2, action='append', metavar=('LABEL', 'PATH'), required=True,
                    help='Repeatable: a label and a densified checkpoint path (2 expected).')
    ap.add_argument('--colmap-image', default='IMG_1388.JPG')
    ap.add_argument('--H', type=int, default=1088)
    ap.add_argument('--W', type=int, default=736)
    ap.add_argument('--out', default='outputs/per_scene_optim/geom_containment.png')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    scene = load_gaussian_plant_scene(args.source, args.output_dir)
    tree0 = root_branch_graph(scene.branch, scene.tube)
    N0 = tree0.nodes.shape[0]

    gen = torch.Generator().manual_seed(0)
    bp = subsample_points(scene.tube.vertices, 8000, generator=gen)
    g = estimate_geom_margin(tree0.nodes, bp, 0.95, 2.0)

    sparse = Path(args.source) / 'sparse' / '0'
    cams = read_cameras_bin(sparse / 'cameras.bin')
    imgs = read_images_bin(sparse / 'images.bin')
    rec = find_image_by_name(imgs, args.colmap_image)
    cam = colmap_camera_to_renderer(rec, cams[rec['cam_id']], args.H, args.W)
    cam = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cam.items()}

    from models.renderer.gaussian_renderer import GaussianRenderer
    renderer = GaussianRenderer(image_height=args.H, image_width=args.W, sh_degree=0).to(device)
    frame = renderer.render_frame(
        scene.app.xyz.to(device), scene.app.scales.to(device), scene.app.rots.to(device),
        scene.app.opacities.to(device), scene.app.colors.to(device), cam, shs=None,
    ).clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()

    panels = args.params
    fig, axes = plt.subplots(1, len(panels), figsize=(len(panels) * args.W / 90.0, args.H / 90.0), dpi=90)
    if len(panels) == 1:
        axes = [axes]

    for ax, (label, path) in zip(axes, panels):
        p = torch.load(path, map_location='cpu', weights_only=False)
        tree = tree_from_densified(p['densified_tree'])
        delta = p['delta_rest_pos']
        eff = tree.nodes + delta
        tol = rest_relative_tolerance(tree.nodes, bp, g, margin=0.02)
        d = nearest_branch_distance(eff, bp)
        exc = (d - tol).clamp_min(0)
        inside = exc <= 0
        uv = project(eff.to(device), cam, args.H, args.W)

        ax.imshow(frame); ax.set_axis_off()
        for e in tree.edges_oriented.cpu().numpy():
            a, b = int(e[0]), int(e[1])
            ax.plot([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]], color='#6a7080', lw=0.7, alpha=0.6, zorder=1)
        ins = inside.cpu().numpy()
        ax.scatter(uv[ins, 0], uv[ins, 1], s=10, c='#33d17a', alpha=0.85,
                   edgecolors='white', linewidths=0.2, zorder=2, label='inside branch')
        out = ~ins
        exc_np = exc.cpu().numpy()
        sizes = 30 + (exc_np[out] / (exc_np.max() + 1e-8)).clip(0, 1) * 120
        ax.scatter(uv[out, 0], uv[out, 1], s=sizes, c='#ff2b2b', alpha=0.95,
                   edgecolors='white', linewidths=0.4, zorder=3, marker='X', label='out of branch')
        ax.set_title(f'{label}\nout-of-branch nodes: {int(out.sum())}  '
                     f'(max excess {float(exc.max())*100:.1f}cm, max|δ| {float(delta.norm(dim=-1).max())*100:.1f}cm)',
                     fontsize=12)
        ax.legend(loc='lower right', fontsize=9, framealpha=0.6)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1, dpi=110)
    plt.close(fig)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
