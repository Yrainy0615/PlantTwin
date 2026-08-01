"""Visualize motion-driven densification: per-round metrics + before/after overlay.

Panel 1: non-rigidity residual and chamfer-to-dense vs densify round (both should drop),
         with node count and round-0 recovery precision annotated.
Panel 2: BEFORE — brown StPr cylinders of the decimated coarse tree over the faded input
         photo; the removed GT joints (decimated) marked red x.
Panel 3: AFTER  — brown cylinders of the densified tree; nodes added by densification
         starred cyan, removed GT joints still marked red x (visual recovery check).

Input: outputs/per_scene_optim/fuse/densify.pt (from fuse_motion_structure --decimate-frac>0).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from data.colmap_loader import (
    read_cameras_bin, read_images_bin, find_image_by_name, colmap_camera_to_renderer,
)


def project_uv_z(P, cam, H, W):
    proj = cam['proj_matrix'].T
    homog = torch.cat([P, torch.ones(P.shape[0], 1, device=P.device)], -1)
    clip = (proj @ homog.T).T
    w = clip[:, 3:4].clamp_min(1e-6)
    ndc = clip[:, :3] / w
    u = (ndc[:, 0] * 0.5 + 0.5) * W
    v = (ndc[:, 1] * 0.5 + 0.5) * H
    zc = (cam['view_matrix'].T @ homog.T).T[:, 2]
    return torch.stack([u, v], -1).cpu().numpy(), zc.cpu().numpy()


def draw_cylinders(ax, img, P, edges, er, cam, fx, H, W, crop, title,
                   star_uv=None, mark_uv=None):
    x0, y0, x1, y1 = crop
    ax.imshow(img[y0:y1, x0:x1]); ax.set_axis_off()
    uv, zc = project_uv_z(P, cam, H, W)
    uvc = uv - np.array([x0, y0])
    e = edges.numpy() if torch.is_tensor(edges) else np.asarray(edges)
    zmid = 0.5 * (zc[e[:, 0]] + zc[e[:, 1]])
    rpx = fx * (er.numpy() if torch.is_tensor(er) else er) / np.clip(zmid, 1e-3, None)
    order = np.argsort(-zmid)
    zr = (zmid - zmid.min()) / (np.ptp(zmid) + 1e-6)
    for k in order:
        a, b = int(e[k, 0]), int(e[k, 1])
        sh = 0.55 + 0.45 * (1 - zr[k])
        col = (0.40 * sh + 0.18, 0.26 * sh + 0.10, 0.11 * sh + 0.04)
        ax.plot([uvc[a, 0], uvc[b, 0]], [uvc[a, 1], uvc[b, 1]], color=col,
                lw=max(0.8, 2 * rpx[k]), alpha=0.95, solid_capstyle='round', zorder=2)
    if mark_uv is not None:
        mu = mark_uv - np.array([x0, y0])
        ax.scatter(mu[:, 0], mu[:, 1], s=55, marker='x', c='#e11d48', linewidths=1.6,
                   zorder=4, label='removed GT joint')
    if star_uv is not None and len(star_uv) > 0:
        su = star_uv - np.array([x0, y0])
        ax.scatter(su[:, 0], su[:, 1], s=70, marker='*', c='#06b6d4',
                   edgecolors='white', linewidths=0.5, zorder=5, label='densified node')
    ax.set_title(title, fontsize=12)
    if mark_uv is not None or star_uv is not None:
        ax.legend(loc='lower right', fontsize=9, framealpha=0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pt', default='outputs/per_scene_optim/fuse/densify.pt')
    ap.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    ap.add_argument('--image', default='IMG_1388.JPG')
    ap.add_argument('--out', default='outputs/per_scene_optim/fuse/densify_progress.png')
    ap.add_argument('--img-alpha', type=float, default=0.4)
    ap.add_argument('--H', type=int, default=1106)
    ap.add_argument('--W', type=int, default=736)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    device = torch.device(args.device)

    d = torch.load(args.pt, map_location='cpu', weights_only=False)
    rounds = d['rounds']; rec = d['recovery']; N_full = d['N_full']
    coarse_nodes, coarse_edges, coarse_er = d['coarse_nodes'], d['coarse_edges'], d['coarse_edge_radius']
    final_P, final_edges, final_er = d['final_P'], d['final_edges'], d['final_edge_radius']
    Nc0 = coarse_nodes.shape[0]
    collapsed_gt = d['collapsed_gt_idx']; Pstar = d['Pstar']
    er_cap = float(coarse_er.quantile(0.95))
    coarse_er = coarse_er.clamp(max=er_cap); final_er = final_er.clamp(max=er_cap)

    # camera + faded image
    sparse = Path(args.source) / 'sparse' / '0'
    cams = read_cameras_bin(sparse / 'cameras.bin'); imgs = read_images_bin(sparse / 'images.bin')
    rec_im = find_image_by_name(imgs, args.image)
    cam = colmap_camera_to_renderer(rec_im, cams[rec_im['cam_id']], args.H, args.W)
    cam = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cam.items()}
    fx = float(cam['fx'])
    img = np.asarray(Image.open(Path(args.source) / 'images' / args.image)
                     .convert('RGB').resize((args.W, args.H))).astype(np.float32) / 255.0
    img = img * args.img_alpha + (1 - args.img_alpha)

    # crop to plant bbox from final nodes
    uv_all, _ = project_uv_z(final_P.to(device), cam, args.H, args.W)
    m = 45
    x0 = max(0, int(uv_all[:, 0].min()) - m); x1 = min(args.W, int(uv_all[:, 0].max()) + m)
    y0 = max(0, int(uv_all[:, 1].min()) - m); y1 = min(args.H, int(uv_all[:, 1].max()) + m)
    crop = (x0, y0, x1, y1)

    mark_uv, _ = project_uv_z(Pstar[collapsed_gt].to(device), cam, args.H, args.W)
    added = final_P[Nc0:]
    star_uv = (project_uv_z(added.to(device), cam, args.H, args.W)[0] if added.shape[0] > 0 else None)

    fig = plt.figure(figsize=(17, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1])

    # panel 1: metrics
    axm = fig.add_subplot(gs[0, 0])
    rr = [h['round'] for h in rounds]
    nonrig = [h['nonrig_cm'] for h in rounds]; cham = [h['chamfer_cm'] for h in rounds]
    Ns = [h['N'] for h in rounds]
    axm.plot(rr, nonrig, '-o', color='#2563eb', label='non-rigidity (cm)')
    axm.plot(rr, cham, '-s', color='#16a34a', label='chamfer to dense (cm)')
    axm.set_xlabel('densify round'); axm.set_ylabel('cm'); axm.grid(alpha=0.3)
    axm.set_xticks(rr)
    ax2 = axm.twinx()
    ax2.plot(rr, Ns, '--^', color='#9333ea', alpha=0.7, label='node count')
    ax2.axhline(N_full, ls=':', color='#9333ea', alpha=0.5)
    ax2.set_ylabel('node count', color='#9333ea')
    lines = axm.get_lines() + ax2.get_lines()
    axm.legend(lines, [l.get_label() for l in lines], fontsize=9, loc='center right')
    title = f'densify: nonrig {nonrig[0]:.3f}->{nonrig[-1]:.3f}, chamfer {cham[0]:.3f}->{cham[-1]:.3f}, N {Ns[0]}->{Ns[-1]} (GT {N_full})'
    if rec:
        title += f'\nround-0 recovery precision = {rec["precision"]:.2f} ({rec["hit"]}/{rec["split"]} splits on removed joints)'
    axm.set_title(title, fontsize=10)

    # panel 2: coarse (before)
    axb = fig.add_subplot(gs[0, 1])
    draw_cylinders(axb, img, coarse_nodes.to(device), coarse_edges, coarse_er, cam, fx,
                   args.H, args.W, crop, f'BEFORE — coarse (N={Nc0})', mark_uv=mark_uv)
    # panel 3: final (after)
    axa = fig.add_subplot(gs[0, 2])
    draw_cylinders(axa, img, final_P.to(device), final_edges, final_er, cam, fx,
                   args.H, args.W, crop, f'AFTER — densified (N={final_P.shape[0]})',
                   star_uv=star_uv, mark_uv=mark_uv)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
