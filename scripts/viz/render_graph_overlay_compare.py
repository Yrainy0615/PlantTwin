"""Branch-graph topology before/after, projected onto the real input image.

Topology = which candidate edges are real branch connections. The fused optimizer
assigns each candidate edge a soft weight w = sigmoid(logit):

  before = motion_out  (geometry-only: short edges look connected)
  after  = motion_in   (+ 3D rigidity: true segments stay rigid, spurious edges stretch)

Both panels project the SAME GT node layout (Pstar) onto IMG_1388 so the only visual
difference is the per-edge weight. True edges are drawn green, spurious candidate edges
red; per-edge opacity + width scale with the learned weight. After cleanup the red
(spurious) edges fade out and the green (true) tree stays bright.

Input: outputs/per_scene_optim/fuse/joint.pt (cand, is_true, results[*]['edge_weight'],
Pstar).
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


def project(points, camera, H, W):
    P = camera['proj_matrix'].T
    homog = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=-1)
    clip = (P @ homog.T).T
    w = clip[:, 3:4].clamp_min(1e-6)
    ndc = clip[:, :3] / w
    u = (ndc[:, 0] * 0.5 + 0.5) * W
    v = (ndc[:, 1] * 0.5 + 0.5) * H
    return torch.stack([u, v], dim=-1).cpu().numpy()


def draw_panel(ax, img, uv, cand, is_true, w, title, crop):
    x0, y0, x1, y1 = crop
    ax.imshow(img[y0:y1, x0:x1])
    ax.set_axis_off()
    uvc = uv - np.array([x0, y0])
    is_true = is_true.numpy() if torch.is_tensor(is_true) else is_true
    w = w.numpy() if torch.is_tensor(w) else np.asarray(w)
    order = np.argsort(w)  # draw weak first so strong (confident) edges land on top
    for e in order:
        a, b = int(cand[e, 0]), int(cand[e, 1])
        col = '#00e5ff' if is_true[e] else '#ff2d2d'   # cyan true vs red spurious (contrast vs green foliage)
        ax.plot([uvc[a, 0], uvc[b, 0]], [uvc[a, 1], uvc[b, 1]],
                color=col, lw=0.8 + 3.5 * float(w[e]),
                alpha=float(np.clip(0.12 + 0.85 * w[e], 0, 1)), zorder=2,
                solid_capstyle='round')
    ax.scatter(uvc[:, 0], uvc[:, 1], s=5, c='#1e293b', alpha=0.6,
               edgecolors='white', linewidths=0.2, zorder=3)
    ax.set_title(title, fontsize=13)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pt', default='outputs/per_scene_optim/fuse/joint.pt')
    ap.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    ap.add_argument('--image', default='IMG_1388.JPG')
    ap.add_argument('--out', default='outputs/per_scene_optim/fuse/graph_before_after.png')
    ap.add_argument('--H', type=int, default=1106)
    ap.add_argument('--W', type=int, default=736)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    d = torch.load(args.pt, map_location='cpu', weights_only=False)
    cand = d['cand']; is_true = d['is_true']; Pstar = d['Pstar'].to(device)
    w_before = d['results']['motion_out']['edge_weight']
    w_after = d['results']['motion_in']['edge_weight']
    auc_b, prec_b = d['results']['motion_out']['auc'], d['results']['motion_out']['prec']
    auc_a, prec_a = d['results']['motion_in']['auc'], d['results']['motion_in']['prec']
    kE = int(is_true.sum())

    sparse = Path(args.source) / 'sparse' / '0'
    cams = read_cameras_bin(sparse / 'cameras.bin')
    imgs = read_images_bin(sparse / 'images.bin')
    rec = find_image_by_name(imgs, args.image)
    cam = colmap_camera_to_renderer(rec, cams[rec['cam_id']], args.H, args.W)
    cam = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cam.items()}

    img = np.asarray(Image.open(Path(args.source) / 'images' / args.image)
                     .convert('RGB').resize((args.W, args.H)))
    uv = project(Pstar, cam, args.H, args.W)

    # crop to the plant region (graph bbox + margin) so the thin branch graph is legible
    m = 45
    x0 = max(0, int(uv[:, 0].min()) - m); x1 = min(args.W, int(uv[:, 0].max()) + m)
    y0 = max(0, int(uv[:, 1].min()) - m); y1 = min(args.H, int(uv[:, 1].max()) + m)
    crop = (x0, y0, x1, y1)
    cw, ch = x1 - x0, y1 - y0

    fig, axes = plt.subplots(1, 2, figsize=(2 * cw / 70.0, ch / 70.0), dpi=70)
    draw_panel(axes[0], img, uv, cand, is_true, w_before,
               f'BEFORE (motion_out)\nAUC={auc_b:.3f}  prec@{kE}={prec_b:.3f}', crop)
    draw_panel(axes[1], img, uv, cand, is_true, w_after,
               f'AFTER (motion_in)\nAUC={auc_a:.3f}  prec@{kE}={prec_a:.3f}', crop)
    # shared legend
    axes[1].plot([], [], color='#00e5ff', lw=3, label='true branch edge')
    axes[1].plot([], [], color='#ff2d2d', lw=3, label='spurious candidate edge')
    axes[1].legend(loc='lower right', fontsize=10, framealpha=0.6)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'wrote {args.out}  (AUC {auc_b:.3f}->{auc_a:.3f}, prec {prec_b:.3f}->{prec_a:.3f})')


if __name__ == '__main__':
    main()
