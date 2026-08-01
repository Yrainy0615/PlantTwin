"""Before/after StPr skeleton comparison.

Left : original StPr skeleton (N0 nodes).
Right: motion-densified skeleton, where the split edges are chosen by the DIFFERENTIAL
       rest correction ||delta_c - delta_p|| (local articulation deficit) rather than the
       absolute ||delta_c|| (which is dominated by the global pivot at the anchored base
       and wrongly piles every new node at the root). New nodes are drawn in red.

The plant photo is rendered as a faded, semi-transparent backdrop so the skeleton reads
clearly on top.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.colmap_loader import read_cameras_bin, read_images_bin, find_image_by_name, colmap_camera_to_renderer
from data.gaussian_plant_loader import load_gaussian_plant_scene
from models.structure.graph_cleanup import root_branch_graph
from models.structure.tree_densify import split_edges

BLUE = '#2b7fb8'; EDGE = '#4a5568'; RED = '#E4362A'; HALO = '#FFD24A'


def project(points, camera, H, W):
    P = camera['proj_matrix'].T
    homog = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=-1)
    clip = (P @ homog.T).T
    w = clip[:, 3:4].clamp_min(1e-6)
    ndc = clip[:, :3] / w
    u = (ndc[:, 0] * 0.5 + 0.5) * W
    v = (ndc[:, 1] * 0.5 + 0.5) * H
    return torch.stack([u, v], dim=-1).detach().cpu().numpy()


def draw_skeleton(ax, uv, edges, *, node_c, edge_c, node_s=12, edge_lw=1.4, node_alpha=0.9):
    for e in edges:
        a, b = int(e[0]), int(e[1])
        ax.plot([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]], color=edge_c, lw=edge_lw, alpha=0.85, zorder=2)
    ax.scatter(uv[:, 0], uv[:, 1], s=node_s, c=node_c, alpha=node_alpha,
               edgecolors='white', linewidths=0.25, zorder=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    ap.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    ap.add_argument('--params', default='outputs/per_scene_optim/newplant9_v14_constrained/final_params.pt')
    ap.add_argument('--colmap-image', default='IMG_1388.JPG')
    ap.add_argument('--H', type=int, default=1088)
    ap.add_argument('--W', type=int, default=736)
    ap.add_argument('--k-split', type=int, default=12)
    ap.add_argument('--bg-alpha', type=float, default=0.32, help='photo opacity (0=white, 1=full)')
    ap.add_argument('--out', default='outputs/rerun_2026-07/strpr_skeleton_compare.png')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    scene = load_gaussian_plant_scene(args.source, args.output_dir)
    tree0 = root_branch_graph(scene.branch, scene.tube)
    N0 = tree0.nodes.shape[0]
    edges0 = tree0.edges_oriented

    p = torch.load(args.params, map_location='cpu', weights_only=False)
    delta = p['delta_rest_pos'][:N0]                      # learned rest correction on original nodes

    # ---- corrected split selection: differential delta along each edge ----
    elen = tree0.edge_length.numpy(); med = float(np.median(elen))
    dc = delta[edges0[:, 1]]; dp = delta[edges0[:, 0]]
    diff = (dc - dp).norm(dim=-1).numpy()                 # |delta_c - delta_p| per edge
    ok = elen >= 0.3 * med                                # don't split already-short edges
    score = np.where(ok, diff, -1.0)
    split_idx = np.argsort(score)[::-1][:args.k_split].tolist()
    tree1, info = split_edges(tree0, split_idx)
    new_idx = info['new_node_indices'].numpy()

    # ---- camera + faded background render ----
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
    faded = frame * args.bg_alpha + (1.0 - args.bg_alpha)     # blend toward white

    uv0 = project(tree0.nodes.to(device), cam, args.H, args.W)
    uv1 = project(tree1.nodes.to(device), cam, args.H, args.W)

    fig, axes = plt.subplots(1, 2, figsize=(2 * args.W / 96.0, args.H / 96.0), dpi=96)
    for ax in axes:
        ax.imshow(faded); ax.set_axis_off()

    draw_skeleton(axes[0], uv0, edges0.numpy(), node_c=BLUE, edge_c=EDGE)
    axes[0].set_title(f'Before  ·  StPr skeleton (N={N0})', color='black', fontsize=14)

    draw_skeleton(axes[1], uv1, tree1.edges_oriented.numpy(), node_c=BLUE, edge_c=EDGE)
    # highlight new nodes in red (halo + marker)
    nu = uv1[new_idx]
    axes[1].scatter(nu[:, 0], nu[:, 1], s=430, c=HALO, alpha=0.35, zorder=4, edgecolors='none')
    axes[1].scatter(nu[:, 0], nu[:, 1], s=150, c=RED, marker='o', edgecolors='white',
                    linewidths=1.3, zorder=5, label=f'new node (+{len(new_idx)})')
    axes[1].set_title(f'After  ·  motion-densified (N={tree1.nodes.shape[0]}, +{len(new_idx)})',
                      color='black', fontsize=14)
    axes[1].legend(loc='lower right', fontsize=11, framealpha=0.75)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out, bbox_inches='tight', pad_inches=0.08, dpi=120); plt.close()
    Y = tree1.nodes[new_idx, 1]; ymin, ymax = tree0.nodes[:, 1].min(), tree0.nodes[:, 1].max()
    hn = ((Y - ymin) / (ymax - ymin)).numpy()
    print(f'wrote {args.out}  | new-node heights (0=root,1=top): {np.round(np.sort(hn),2)}')


if __name__ == '__main__':
    main()
