"""Before/after skeleton for MOTION-NON-RIGIDITY densification.

The branch tree is the motion basis. Edges whose bound AppGas move in a way a single
rigid bone cannot reproduce (high non-rigidity residual, models/structure/motion_residual.py)
are split — so new nodes land where the observed motion is un-fittable, i.e. the
under-articulated thin branches, NOT the pivot at the root.

New nodes are filtered to lie within the image (a reasonable-range guard; the optimizer's
geometric-containment loss L_geom enforces the same in the real pipeline).
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

BLUE = '#2b7fb8'; EDGE = '#4a5568'; RED = '#E4362A'; HALO = '#FFD24A'; OUT = '#9aa0a6'


def project(points, camera, H, W):
    P = camera['proj_matrix'].T
    homog = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=-1)
    clip = (P @ homog.T).T
    w = clip[:, 3:4].clamp_min(1e-6)
    ndc = clip[:, :3] / w
    u = (ndc[:, 0] * 0.5 + 0.5) * W
    v = (ndc[:, 1] * 0.5 + 0.5) * H
    return torch.stack([u, v], dim=-1).detach().cpu().numpy()


def draw(ax, uv, edges, *, node_c, node_s=11, edge_lw=1.3):
    for e in edges:
        a, b = int(e[0]), int(e[1])
        ax.plot([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]], color=EDGE, lw=edge_lw, alpha=0.8, zorder=2)
    ax.scatter(uv[:, 0], uv[:, 1], s=node_s, c=node_c, alpha=0.85, edgecolors='white', linewidths=0.2, zorder=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    ap.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    ap.add_argument('--densify-pt', default='outputs/per_scene_optim/fuse/densify_G_detected/densify.pt')
    ap.add_argument('--colmap-image', default='IMG_1388.JPG')
    ap.add_argument('--H', type=int, default=1088)
    ap.add_argument('--W', type=int, default=736)
    ap.add_argument('--bg-alpha', type=float, default=0.32)
    ap.add_argument('--margin', type=float, default=8.0, help='px margin for the in-image range guard')
    ap.add_argument('--out', default='outputs/rerun_2026-07/densify_nonrigid_compare.png')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    scene = load_gaussian_plant_scene(args.source, args.output_dir)

    d = torch.load(args.densify_pt, map_location='cpu', weights_only=False)
    coarse_P = d['coarse_nodes']; coarse_E = d['coarse_edges']
    final_P = d['final_P']; final_E = d['final_edges']
    Nc, Nf = coarse_P.shape[0], final_P.shape[0]
    new_idx = np.arange(Nc, Nf)

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
    faded = frame * args.bg_alpha + (1.0 - args.bg_alpha)

    uvc = project(coarse_P.to(device), cam, args.H, args.W)
    uvf = project(final_P.to(device), cam, args.H, args.W)

    # in-image range guard for the new nodes
    m = args.margin
    unew = uvf[new_idx]
    inrange = (unew[:, 0] > m) & (unew[:, 0] < args.W - m) & (unew[:, 1] > m) & (unew[:, 1] < args.H - m)
    kept = new_idx[inrange]; dropped = new_idx[~inrange]

    fig, axes = plt.subplots(1, 2, figsize=(2 * args.W / 96.0, args.H / 96.0), dpi=96)
    for ax in axes:
        ax.imshow(faded); ax.set_axis_off()

    draw(axes[0], uvc, coarse_E.numpy(), node_c=BLUE)
    axes[0].set_title(f'Before  ·  under-articulated basis (N={Nc})', color='black', fontsize=14)

    draw(axes[1], uvf, final_E.numpy(), node_c=BLUE)
    ku = uvf[kept]
    axes[1].scatter(ku[:, 0], ku[:, 1], s=150, c=HALO, alpha=0.30, zorder=4, edgecolors='none')
    axes[1].scatter(ku[:, 0], ku[:, 1], s=60, c=RED, marker='o', edgecolors='white', linewidths=0.8,
                    zorder=5, label=f'new node — motion un-fittable (+{len(kept)})')
    if len(dropped):
        du = uvf[dropped]
        axes[1].scatter(du[:, 0], du[:, 1], s=45, c=OUT, marker='x', zorder=5,
                        label=f'dropped — out of image range ({len(dropped)})')
    axes[1].set_title(f'After  ·  split where the branch basis can’t fit the motion (N={Nf})',
                      color='black', fontsize=13)
    axes[1].legend(loc='lower right', fontsize=10.5, framealpha=0.8)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out, bbox_inches='tight', pad_inches=0.08, dpi=120); plt.close()
    print(f'wrote {args.out} | new={len(new_idx)} kept(in-range)={len(kept)} dropped={len(dropped)}')


if __name__ == '__main__':
    main()
