"""StPr cylinder overlay (GaussianPlant paper style) + chamfer-to-dense-branch.

Two deliverables for the motion before/after comparison:

  1. CHAMFER: symmetric chamfer between the StPr cylinder surface (sampled from the
     branch tree: nodes P + edges + edge_radius) and GaussianPlant's DENSE branch
     point cloud (branch.ply, ~54k pts). Reported for the GT skeleton (floor), the
     degraded init, and before/after motion optimization. This is the quantitative
     3D metric the input-view overlay cannot show (perturbation is along that view's
     forward axis).

  2. OVERLAY: brown StPr cylinders drawn over the semi-transparent input photo, in
     the GaussianPlant figure style. Cylinders are projected with the verified
     proj_matrix (same one that lands the branch graph on the plant), depth-sorted,
     each edge drawn at its projected pixel radius. before = motion_out, after =
     motion_in.

Input: outputs/per_scene_optim/fuse/joint.pt (P per mode, edges_true, edge_radius,
Pstar, P_init).
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
from data.gaussian_plant_loader import load_gaussian_plant_scene


# --------------------------------------------------------------------------- #
# chamfer
# --------------------------------------------------------------------------- #

def sample_cylinders(P, edges, er, device, around=12, per_len=300, cap=40):
    P = P.to(device); a = P[edges[:, 0]]; b = P[edges[:, 1]]
    axis = b - a; L = axis.norm(dim=-1).clamp_min(1e-6); dax = axis / L.unsqueeze(-1)
    up = torch.tensor([0., 0., 1.], device=device).expand_as(dax)
    e1 = torch.cross(dax, up, dim=-1); n = e1.norm(dim=-1, keepdim=True)
    e1 = torch.where(n > 1e-4, e1 / n.clamp_min(1e-6),
                     torch.tensor([1., 0., 0.], device=device).expand_as(dax))
    e2 = torch.cross(dax, e1, dim=-1)
    ang = torch.linspace(0, 2 * np.pi, around + 1, device=device)[:-1]
    nalong = (L * per_len).clamp(2, cap).long()
    pts = []
    for i in range(edges.shape[0]):
        na = int(nalong[i]); ts = torch.linspace(0, 1, na, device=device)
        cen = a[i].unsqueeze(0) + ts.unsqueeze(-1) * axis[i].unsqueeze(0)
        ring = er[i] * (torch.cos(ang).unsqueeze(-1) * e1[i] + torch.sin(ang).unsqueeze(-1) * e2[i])
        pts.append((cen.unsqueeze(1) + ring.unsqueeze(0)).reshape(-1, 3))
    return torch.cat(pts, 0)


def chamfer(X, Y, chunk=2048):
    def nn(A, B):
        out = torch.empty(A.shape[0], device=A.device)
        for i in range(0, A.shape[0], chunk):
            out[i:i + chunk] = torch.cdist(A[i:i + chunk], B).min(1).values
        return out
    return float(0.5 * (nn(X, Y).mean() + nn(Y, X).mean()))


# --------------------------------------------------------------------------- #
# overlay
# --------------------------------------------------------------------------- #

def project_uv_z(P, cam, H, W):
    proj = cam['proj_matrix'].T
    homog = torch.cat([P, torch.ones(P.shape[0], 1, device=P.device)], -1)
    clip = (proj @ homog.T).T
    w = clip[:, 3:4].clamp_min(1e-6)
    ndc = clip[:, :3] / w
    u = (ndc[:, 0] * 0.5 + 0.5) * W
    v = (ndc[:, 1] * 0.5 + 0.5) * H
    view = cam['view_matrix'].T
    zc = (view @ homog.T).T[:, 2]                       # camera-space depth (COLMAP Z fwd)
    return torch.stack([u, v], -1).cpu().numpy(), zc.cpu().numpy()


def draw_overlay(ax, img, uv, zc, edges, er_px, title, crop):
    from matplotlib.collections import LineCollection
    x0, y0, x1, y1 = crop
    ax.imshow(img[y0:y1, x0:x1])
    ax.set_axis_off()
    uvc = uv - np.array([x0, y0])
    e = edges.numpy() if torch.is_tensor(edges) else np.asarray(edges)
    zmid = 0.5 * (zc[e[:, 0]] + zc[e[:, 1]])
    order = np.argsort(-zmid)                           # far first (painter)
    zr = (zmid - zmid.min()) / (np.ptp(zmid) + 1e-6)
    for k in order:
        a, b = int(e[k, 0]), int(e[k, 1])
        shade = 0.55 + 0.45 * (1 - zr[k])              # nearer = brighter brown
        col = (0.40 * shade + 0.18, 0.26 * shade + 0.10, 0.11 * shade + 0.04)
        ax.plot([uvc[a, 0], uvc[b, 0]], [uvc[a, 1], uvc[b, 1]],
                color=col, lw=max(0.8, 2 * er_px[k]), alpha=0.95,
                solid_capstyle='round', zorder=2)
    ax.set_title(title, fontsize=13)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pt', default='outputs/per_scene_optim/fuse/joint.pt')
    ap.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    ap.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    ap.add_argument('--image', default='IMG_1388.JPG')
    ap.add_argument('--out', default='outputs/per_scene_optim/fuse/strpr_overlay_before_after.png')
    ap.add_argument('--img-alpha', type=float, default=0.45, help='input image opacity (rest faded to white)')
    ap.add_argument('--H', type=int, default=1106)
    ap.add_argument('--W', type=int, default=736)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    d = torch.load(args.pt, map_location='cpu', weights_only=False)
    edges = d['edges_true']
    er_cap = float(d['edge_radius'].quantile(0.95))
    er = d['edge_radius'].clamp(max=er_cap)
    Pstar, P_init = d['Pstar'], d['P_init']
    P_before, P_after = d['results']['motion_out']['P'], d['results']['motion_in']['P']

    # ---- chamfer vs dense branch ----
    sc = load_gaussian_plant_scene(args.source, args.output_dir, load_raw_branch=True)
    Dpts = sc.raw_branch_surface.to(device)
    cd = {}
    for name, P in [('GT (Pstar)', Pstar), ('init (degraded)', P_init),
                    ('before (motion_out)', P_before), ('after (motion_in)', P_after)]:
        S = sample_cylinders(P, edges.to(device), er.to(device), device)
        cd[name] = chamfer(S, Dpts) * 100
    floor = cd['GT (Pstar)']
    gap_b = cd['before (motion_out)'] - floor
    gap_a = cd['after (motion_in)'] - floor
    print('chamfer StPr-cylinders -> dense branch (cm):')
    for k, v in cd.items():
        print(f'  {k:22s} {v:.3f}')
    print(f'  gap to GT floor: before {gap_b:.3f} -> after {gap_a:.3f}  '
          f'({(1 - gap_a / max(gap_b,1e-9)) * 100:.0f}% of recoverable gap closed)')

    # ---- overlay render ----
    sparse = Path(args.source) / 'sparse' / '0'
    cams = read_cameras_bin(sparse / 'cameras.bin'); imgs = read_images_bin(sparse / 'images.bin')
    rec = find_image_by_name(imgs, args.image)
    cam = colmap_camera_to_renderer(rec, cams[rec['cam_id']], args.H, args.W)
    cam = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cam.items()}
    fx = float(cam['fx'])

    img = np.asarray(Image.open(Path(args.source) / 'images' / args.image)
                     .convert('RGB').resize((args.W, args.H))).astype(np.float32) / 255.0
    img = img * args.img_alpha + (1 - args.img_alpha)   # fade toward white

    def edge_radius_px(P):
        uv, zc = project_uv_z(P.to(device), cam, args.H, args.W)
        e = edges.numpy()
        zmid = 0.5 * (zc[e[:, 0]] + zc[e[:, 1]])
        rpx = fx * er.numpy() / np.clip(zmid, 1e-3, None)
        return uv, zc, rpx

    uv_b, z_b, rpx_b = edge_radius_px(P_before)
    uv_a, z_a, rpx_a = edge_radius_px(P_after)

    allu = np.concatenate([uv_b[:, 0], uv_a[:, 0]]); allv = np.concatenate([uv_b[:, 1], uv_a[:, 1]])
    m = 45
    x0 = max(0, int(allu.min()) - m); x1 = min(args.W, int(allu.max()) + m)
    y0 = max(0, int(allv.min()) - m); y1 = min(args.H, int(allv.max()) + m)
    crop = (x0, y0, x1, y1); cw, ch = x1 - x0, y1 - y0

    fig, axes = plt.subplots(1, 2, figsize=(2 * cw / 70.0, ch / 70.0), dpi=70)
    draw_overlay(axes[0], img, uv_b, z_b, edges, rpx_b,
                 f'BEFORE (motion_out)\nchamfer to dense branch = {cd["before (motion_out)"]:.3f} cm', crop)
    draw_overlay(axes[1], img, uv_a, z_a, edges, rpx_a,
                 f'AFTER (motion_in)\nchamfer to dense branch = {cd["after (motion_in)"]:.3f} cm', crop)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
