"""Highlight WHERE the motion-densified StPr nodes are.

Unlike compare_struct_before_after.py (which colours the whole tree by |delta| and
buries the 12 inserted nodes among hundreds of markers), this figure de-emphasises the
existing skeleton to faint grey and draws only the NEW midpoint nodes big, haloed and
numbered — so "where are the new points" is answered at a glance.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch

from data.colmap_loader import read_cameras_bin, read_images_bin, find_image_by_name, colmap_camera_to_renderer
from data.gaussian_plant_loader import load_gaussian_plant_scene
from models.structure.graph_cleanup import root_branch_graph, RootedBranchTree

FORCE = '#E4362A'; HALO = '#FFD24A'; GREY = '#7c8088'


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
        edge_type=dt['edge_type'].cpu(), subtree_size=dt['subtree_size'].cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    ap.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    ap.add_argument('--params', default='outputs/per_scene_optim/newplant9_v14_constrained/final_params.pt')
    ap.add_argument('--colmap-image', default='IMG_1388.JPG')
    ap.add_argument('--H', type=int, default=1088)
    ap.add_argument('--W', type=int, default=736)
    ap.add_argument('--out', default='outputs/rerun_2026-07/densify_newnodes.png')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    scene = load_gaussian_plant_scene(args.source, args.output_dir)
    tree0 = root_branch_graph(scene.branch, scene.tube)
    N0 = tree0.nodes.shape[0]

    p = torch.load(args.params, map_location='cpu', weights_only=False)
    tree1 = tree_from_densified(p['densified_tree'])
    delta = p.get('delta_rest_pos', torch.zeros_like(tree1.nodes))
    new_idx = np.arange(N0, tree1.nodes.shape[0])
    nodes1_eff = tree1.nodes + delta

    sparse = Path(args.source) / 'sparse' / '0'
    cams = read_cameras_bin(sparse / 'cameras.bin'); imgs = read_images_bin(sparse / 'images.bin')
    rec = find_image_by_name(imgs, args.colmap_image)
    cam = colmap_camera_to_renderer(rec, cams[rec['cam_id']], args.H, args.W)
    cam = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cam.items()}

    from models.renderer.gaussian_renderer import GaussianRenderer
    renderer = GaussianRenderer(image_height=args.H, image_width=args.W, sh_degree=0).to(device)
    frame = renderer.render_frame(
        scene.app.xyz.to(device), scene.app.scales.to(device), scene.app.rots.to(device),
        scene.app.opacities.to(device), scene.app.colors.to(device), cam, shs=None,
    ).clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()

    uv1 = project(nodes1_eff.to(device), cam, args.H, args.W)
    e1 = tree1.edges_oriented.cpu().numpy()
    # parent edge of each new node -> for a short "split" tick
    parent = tree1.parent.cpu().numpy()

    fig, ax = plt.subplots(figsize=(args.W / 96.0, args.H / 96.0), dpi=96)
    ax.imshow(frame); ax.set_axis_off()

    # faint existing skeleton
    for e in e1:
        a, b = int(e[0]), int(e[1])
        ax.plot([uv1[a, 0], uv1[b, 0]], [uv1[a, 1], uv1[b, 1]], color=GREY, lw=1.0, alpha=0.5, zorder=1)
    ax.scatter(uv1[:, 0], uv1[:, 1], s=5, c=GREY, alpha=0.45, zorder=2)

    # NEW nodes: halo + big star (no per-node numbers; they cluster too tightly)
    nx, ny = uv1[new_idx, 0], uv1[new_idx, 1]
    ax.scatter(nx, ny, s=540, c=HALO, alpha=0.32, zorder=3, edgecolors='none')
    ax.scatter(nx, ny, s=200, c=FORCE, marker='*', edgecolors='white', linewidths=1.4, zorder=5)

    # group callout around the lower-stem cluster (most new nodes)
    from matplotlib.patches import FancyBboxPatch
    med_y = np.median(ny)
    cluster = new_idx[uv1[new_idx, 1] >= med_y - 1]
    if len(cluster) >= 3:
        cx0, cx1 = uv1[cluster, 0].min() - 26, uv1[cluster, 0].max() + 26
        cy0, cy1 = uv1[cluster, 1].min() - 26, uv1[cluster, 1].max() + 26
        ax.add_patch(FancyBboxPatch((cx0, cy0), cx1 - cx0, cy1 - cy0,
                     boxstyle='round,pad=6', fill=False, ec=FORCE, lw=2.0, ls=(0, (5, 3)), zorder=4))
        ax.annotate(f'{len(cluster)} of {len(new_idx)} new nodes here:\nlower main stem — largest hand-pull\nbend, one cylinder too rigid to fit',
                    xy=(cx0, (cy0 + cy1) / 2), xytext=(cx0 - 250, (cy0 + cy1) / 2 - 30),
                    fontsize=10.5, color=FORCE, weight='bold', va='center', ha='left', zorder=7,
                    arrowprops=dict(arrowstyle='->', color=FORCE, lw=1.8))

    ax.set_title(f'Where the motion-densified StPr nodes are  ·  +{len(new_idx)} new (★), N={N0}→{tree1.nodes.shape[0]}',
                 color='black', fontsize=13)
    ax.scatter([], [], s=170, c=FORCE, marker='*', edgecolors='white', label='new midpoint node (split)')
    ax.plot([], [], color=GREY, lw=1.0, label='existing StPr skeleton')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.7)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out, bbox_inches='tight', pad_inches=0.08, dpi=120); plt.close()
    print(f'wrote {args.out}  ({len(new_idx)} new nodes highlighted)')


if __name__ == '__main__':
    main()
