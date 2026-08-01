"""Motion-driven StPr densification, constrained by GaussianPlant appearance + binding.

Thesis: where the observed motion cannot be explained by a StPr's single rigid transform
(its bound AppGas bend WITHIN it), the StPr is under-articulated. Splitting it along its
axis and re-optimizing the children under GaussianPlant's binding + appearance constraints
lets the motion be explained WITHOUT breaking the static reconstruction.

Controlled synthetic setup (aligned strpr_branch + point_cloud_clean, newplant / gsplant_results):
1. Global gentle FK sway, PLUS an extra bend applied to the TIP half of one target StPr t*
   -> t*'s bound AppGas move non-rigidly. Render the observed video.
2. Per-StPr non-rigidity residual (Kabsch on each StPr's bound AppGas) flags t*.
3. COARSE fit: each StPr = one rigid transform (its cluster Kabsch). t* cannot bend -> residual.
4. DENSIFY: split t* into base/tip children.
5. CONSTRAINED re-opt: Adam on the children (center, scale, rot, colour, opacity) under
   binding (Gaussian NLL of each child's bound AppGas) + appearance (StPr render vs AppGas
   render). This keeps the children valid GS primitives.
6. SPLIT fit: each child = one rigid transform -> the bend is now representable.

Metrics: t*-region motion residual (coarse vs split), appearance PSNR StPr-vs-AppGas
(preserved through the split), binding Mahalanobis (children fit their points).
Outputs under outputs/rerun_2026-07/strpr_densify/.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio

from data.gaussian_plant_loader import _load_app_gaussians
from data.colmap_loader import read_cameras_bin, read_images_bin, colmap_camera_to_renderer
from models.renderer.gaussian_renderer import GaussianRenderer
from models.structure.strpr_motion import (load_strpr_branch, gaussian_lbs_binding,
                                           apply_strpr_lbs_full, estimate_node_motion_excl)
from models.structure.strpr_densify import split_strpr, gaussian_nll, render_strpr_gaussians
from simulation.strpr_chain import build_strpr_tree, fk


def kabsch(P0, Pt):
    c0 = P0.mean(0); ct = Pt.mean(1)
    Pc = P0 - c0; Qc = Pt - ct[:, None, :]
    H = torch.einsum('ni,tnj->tij', Pc, Qc)
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.transpose(-1, -2) @ U.transpose(-1, -2)))
    D = torch.eye(3, device=P0.device).expand(H.shape[0], 3, 3).clone(); D[:, 2, 2] = d
    R = Vt.transpose(-1, -2) @ D @ U.transpose(-1, -2)
    pred = torch.einsum('tij,nj->tni', R, Pc) + ct[:, None, :]
    return R, pred


def per_strpr_residual(top1, ap_rest, ap_obs, S, min_pts=8):
    """Kabsch residual [S] of each StPr's exclusive AppGas cluster (non-rigidity)."""
    res = torch.zeros(S, device=ap_rest.device)
    for i in range(S):
        m = top1 == i
        if int(m.sum()) < min_pts:
            continue
        _, pred = kabsch(ap_rest[m], ap_obs[:, m])
        res[i] = (ap_obs[:, m] - pred).norm(dim=-1).mean()
    return res


def deform_by_cluster_rigid(strpr, binding, ap_rest, ap_obs):
    """Reconstruct AppGas by each StPr's own rigid cluster fit (coarse/split renderer)."""
    obs_pos, R_obs, _, _ = estimate_node_motion_excl(strpr, binding, ap_rest, ap_obs,
                                                     torch.empty(0, 2, dtype=torch.long,
                                                                 device=ap_rest.device))
    return obs_pos, R_obs


def render_traj(rnd, app, binding, strpr, pos_t, rot_t, cam, ap_rest):
    out = []
    for t in range(pos_t.shape[0]):
        xyz, quat = apply_strpr_lbs_full(binding, strpr, pos_t[t], rot_t[t], ap_rest, app.rots)
        img = rnd.render_frame(xyz, app.scales, quat, app.opacities, app.colors, cam, shs=None).clamp(0, 1)
        out.append((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    return out


def psnr(a, b):
    a = a.astype(np.float32) / 255; b = b.astype(np.float32) / 255
    return 20 * math.log10(1.0) - 10 * math.log10(max(((a - b) ** 2).mean(), 1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strpr', default='/mnt/data/gaussianplant_data/gsplant_results/strpr_branch.ply')
    ap.add_argument('--appgas', default='/mnt/data/gaussianplant_data/gsplant_results/point_cloud_clean.ply')
    ap.add_argument('--scene', default='/mnt/data/gaussianplant_data/newplant4')
    ap.add_argument('--out', default='outputs/rerun_2026-07/strpr_densify')
    ap.add_argument('--frames', type=int, default=16)
    ap.add_argument('--bend', type=float, default=0.6, help='extra tip-bend angle (rad)')
    ap.add_argument('--amp', type=float, default=0.25, help='global sway amp (rad)')
    ap.add_argument('--reopt-iters', type=int, default=300)
    ap.add_argument('--lambda-appear', type=float, default=1.0)
    ap.add_argument('--height', type=int, default=832)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    strpr = load_strpr_branch(args.strpr).to(dev)
    app = _load_app_gaussians(Path(args.appgas))
    app = type(app)(**{k: getattr(app, k).to(dev) for k in
                       ['xyz', 'scales', 'rots', 'opacities', 'colors', 'shs']})
    ap_rest = app.xyz
    S = len(strpr)
    tree = build_strpr_tree(strpr).to(dev)
    binding = gaussian_lbs_binding(ap_rest, strpr, k=1)
    top1 = binding.idx[:, 0]

    # camera
    src = Path(args.scene)
    cams = read_cameras_bin(src / 'sparse/0/cameras.bin')
    imgs = read_images_bin(src / 'sparse/0/images.bin')
    first = sorted(imgs.values(), key=lambda r: r['name'])[0]
    ci = cams[first['cam_id']]
    H = args.height; W = int(round(H * ci['w'] / ci['h'])) // 2 * 2
    cam = colmap_camera_to_renderer(first, ci, H, W)
    cam = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in cam.items()}
    rnd = GaussianRenderer(image_height=H, image_width=W, sh_degree=0).to(dev)

    # pick target StPr: a branch StPr with the most bound AppGas (visible effect)
    cnt = torch.bincount(top1.clamp_min(0), minlength=S)
    cnt[strpr.label < 0.5] = 0
    tstar = int(cnt.argmax().item())
    print(f'[target] StPr {tstar}: {int(cnt[tstar])} bound AppGas, half_len={strpr.half_len[tstar]:.3f}')

    # ---- GT motion: global FK sway + extra bend on t*'s TIP-half AppGas ----
    T = args.frames
    depth = torch.zeros(S, dtype=torch.long)
    for i in tree.order.tolist():
        p = int(tree.parent[i].item()); depth[i] = 0 if p < 0 else depth[p] + 1
    dmax = depth.to(dev).max().clamp_min(1)
    axis = strpr.axis
    wind = torch.tensor([1., 0., .3], device=dev); wind = wind / wind.norm()
    bend_ax = torch.cross(axis, wind.expand_as(axis), dim=-1)
    bend_ax = bend_ax / bend_ax.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    ts = torch.arange(T, device=dev).float()
    phi = 2 * math.pi * torch.rand(S, device=dev)
    phase = torch.sin(2 * math.pi * ts[:, None] / T + phi[None, :])            # [T,S]
    theta = (args.amp / float(dmax)) * phase[:, :, None] * bend_ax[None, :, :].expand(T, S, 3)
    pos_gt = torch.zeros(T, S, 3, device=dev); rot_gt = torch.zeros(T, S, 3, 3, device=dev)
    for t in range(T):
        pos_gt[t], rot_gt[t] = fk(tree, theta[t])
    ap_gt, _ = apply_strpr_lbs_full(binding, strpr, pos_gt[0], rot_gt[0], ap_rest, app.rots)  # rest==input
    ap_obs = torch.stack([apply_strpr_lbs_full(binding, strpr, pos_gt[t], rot_gt[t],
                                               ap_rest, app.rots)[0] for t in range(T)])
    # extra bend: rotate the TIP half of t*'s AppGas about the t* midpoint
    m_t = (top1 == tstar)
    a_t = axis[tstar]; c_t = strpr.xyz[tstar]
    proj = (ap_rest[m_t] - c_t) @ a_t
    tip = proj > 0                                                            # tip half
    pivot = c_t + 0.0 * a_t
    bend_axis_t = bend_ax[tstar]
    idx_t = m_t.nonzero(as_tuple=False).flatten()
    idx_tip = idx_t[tip]
    for t in range(T):
        ang = args.bend * math.sin(2 * math.pi * t / T)
        K = torch.tensor([[0, -bend_axis_t[2], bend_axis_t[1]],
                          [bend_axis_t[2], 0, -bend_axis_t[0]],
                          [-bend_axis_t[1], bend_axis_t[0], 0]], device=dev)
        Rb = torch.eye(3, device=dev) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
        ap_obs[t, idx_tip] = (ap_obs[t, idx_tip] - pivot) @ Rb.T + pivot
    gt_frames = []
    for t in range(T):
        img = rnd.render_frame(ap_obs[t], app.scales, app.rots, app.opacities, app.colors, cam, shs=None).clamp(0, 1)
        gt_frames.append((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    # ---- residual: t* should be the max ----
    res = per_strpr_residual(top1, ap_rest, ap_obs, S)
    order = torch.argsort(res, descending=True)
    print(f'[residual] top StPr by non-rigidity: '
          f'{[(int(i), round(float(res[i]),4)) for i in order[:5]]}')
    print(f'[residual] t*={tstar} rank={int((order==tstar).nonzero().item())}, res={float(res[tstar]):.4f}')

    # ---- COARSE render: each StPr one rigid fit ----
    pos_c, rot_c = deform_by_cluster_rigid(strpr, binding, ap_rest, ap_obs)
    coarse_frames = render_traj(rnd, app, binding, strpr, pos_c, rot_c, cam, ap_rest)

    # ---- DENSIFY t* + constrained re-opt of the children ----
    strpr2, info = split_strpr(strpr, [tstar])
    binding2 = gaussian_lbs_binding(ap_rest, strpr2, k=1)
    top1_2 = binding2.idx[:, 0]
    cb, ct2 = info['children'][0]

    # AppGas render (teacher) for the appearance constraint
    ap_teacher = rnd.render_frame(ap_rest, app.scales, app.rots, app.opacities, app.colors, cam, shs=None).clamp(0, 1).detach()
    plant_mask = (ap_teacher.sum(0) > 0.03).float()

    # learnable child params (init from split)
    ctr = strpr2.xyz[[cb, ct2]].clone().requires_grad_(True)
    logscale = strpr2.scale[[cb, ct2]].clamp_min(1e-6).log().clone().requires_grad_(True)
    quat = strpr2.quat[[cb, ct2]].clone().requires_grad_(True)
    col = strpr2.colors[[cb, ct2]].clone().requires_grad_(True)
    opa = strpr2.opacity[[cb, ct2]].clamp(1e-4, 1 - 1e-4).logit().clone().requires_grad_(True)
    opt = torch.optim.Adam([ctr, logscale, quat, col, opa], lr=0.01)
    from models.structure.strpr_motion import _quat_to_mat
    appear0 = psnr((render_strpr_gaussians(rnd, strpr2, cam).permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8),
                   (ap_teacher.permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
    for it in range(args.reopt_iters):
        opt.zero_grad()
        q = quat / quat.norm(dim=1, keepdim=True).clamp_min(1e-8)
        Rc = _quat_to_mat(q); sc = logscale.exp()
        # binding: each child fits its bound AppGas
        L_bind = 0.0
        for j, ch in enumerate([cb, ct2]):
            m = (top1_2 == ch)
            if int(m.sum()) >= 6:
                L_bind = L_bind + gaussian_nll(ap_rest[m], ctr[j], Rc[j], sc[j])
        # appearance: full StPr render (with updated children) vs AppGas teacher
        xyz_all = strpr2.xyz.clone(); xyz_all[[cb, ct2]] = ctr
        sc_all = strpr2.scale.clone(); sc_all[[cb, ct2]] = sc
        q_all = strpr2.quat.clone(); q_all[[cb, ct2]] = q
        col_all = strpr2.colors.clone(); col_all[[cb, ct2]] = col.clamp(0, 1)
        op_all = strpr2.opacity.clone(); op_all[[cb, ct2]] = torch.sigmoid(opa)
        q_all = q_all / q_all.norm(dim=1, keepdim=True).clamp_min(1e-8)
        strpr_img = rnd.render_frame(xyz_all, sc_all, q_all, op_all, col_all.clamp(0, 1), cam, shs=None).clamp(0, 1)
        L_app = ((strpr_img - ap_teacher).abs() * plant_mask).mean()
        loss = L_bind + args.lambda_appear * L_app
        loss.backward(); opt.step()
    # write back optimized children
    with torch.no_grad():
        q = quat / quat.norm(dim=1, keepdim=True).clamp_min(1e-8)
        strpr2.xyz[[cb, ct2]] = ctr; strpr2.scale[[cb, ct2]] = logscale.exp()
        strpr2.quat[[cb, ct2]] = q; strpr2.R[[cb, ct2]] = _quat_to_mat(q)
        strpr2.colors[[cb, ct2]] = col.clamp(0, 1); strpr2.opacity[[cb, ct2]] = torch.sigmoid(opa)
    binding2 = gaussian_lbs_binding(ap_rest, strpr2, k=1)
    appear1 = psnr((render_strpr_gaussians(rnd, strpr2, cam).permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8),
                   (ap_teacher.permute(1,2,0).cpu().numpy()*255).astype(np.uint8))

    # ---- SPLIT render: each (now 2) StPr one rigid fit ----
    pos_s, rot_s = deform_by_cluster_rigid(strpr2, binding2, ap_rest, ap_obs)
    split_frames = render_traj(rnd, app, binding2, strpr2, pos_s, rot_s, cam, ap_rest)

    # ---- metrics on t*'s AppGas ----
    def region_resid(binding_, top1_, children_ids):
        m = torch.zeros(ap_rest.shape[0], dtype=torch.bool, device=dev)
        for ch in children_ids:
            m |= (top1_ == ch)
        return m
    reg = m_t                                                   # original t* AppGas
    res_coarse = float((torch.stack([apply_strpr_lbs_full(binding, strpr, pos_c[t], rot_c[t], ap_rest, app.rots)[0]
                                     for t in range(T)])[:, reg] - ap_obs[:, reg]).norm(dim=-1).mean())
    res_split = float((torch.stack([apply_strpr_lbs_full(binding2, strpr2, pos_s[t], rot_s[t], ap_rest, app.rots)[0]
                                    for t in range(T)])[:, reg] - ap_obs[:, reg]).norm(dim=-1).mean())
    # binding Mahalanobis of t* AppGas
    def maha_of(strpr_, top1_, ids):
        vals = []
        for ch in ids:
            m = top1_ == ch
            if m.any():
                loc = (ap_rest[m] - strpr_.xyz[ch]) @ strpr_.R[ch]
                vals.append(((loc ** 2) / (strpr_.scale[ch] ** 2).clamp_min(1e-10)).sum(-1).sqrt())
        return float(torch.cat(vals).mean()) if vals else 0.0
    maha_coarse = maha_of(strpr, top1, [tstar])
    maha_split = maha_of(strpr2, top1_2, [cb, ct2])

    psnr_c = float(np.mean([psnr(a, b) for a, b in zip(coarse_frames, gt_frames)]))
    psnr_s = float(np.mean([psnr(a, b) for a, b in zip(split_frames, gt_frames)]))
    metrics = {
        'target_strpr': tstar, 'n_bound_appgas': int(cnt[tstar]),
        'residual_flags_target': {'rank': int((order == tstar).nonzero().item()),
                                  'res_target': float(res[tstar]), 'res_max': float(res[order[0]])},
        'tstar_region_motion_residual': {'coarse_1strpr': res_coarse, 'split_2strpr': res_split,
                                         'reduction_pct': 100 * (1 - res_split / max(res_coarse, 1e-9))},
        'render_psnr_vs_gt': {'coarse': psnr_c, 'split': psnr_s, 'gain_db': psnr_s - psnr_c},
        'appearance_psnr_strpr_vs_appgas': {'after_split_before_reopt': appear0,
                                            'after_reopt': appear1},
        'binding_mahalanobis_tstar': {'coarse_1strpr': maha_coarse, 'split_2strpr': maha_split},
        'n_strpr': S, 'n_strpr_after': len(strpr2), 'frames': T,
    }
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    def save(name, frames): imageio.mimsave(out / name, frames, fps=10, quality=8)
    save('gt.mp4', gt_frames); save('coarse.mp4', coarse_frames); save('split.mp4', split_frames)
    save('triptych.mp4', [np.concatenate([g, c, s], 1) for g, c, s in zip(gt_frames, coarse_frames, split_frames)])

    print('\n==== RESULTS ====')
    print(f'residual flags t* at rank {metrics["residual_flags_target"]["rank"]} (0=highest)')
    print(f't* region motion residual: coarse {res_coarse:.4f} -> split {res_split:.4f} '
          f'({metrics["tstar_region_motion_residual"]["reduction_pct"]:.1f}% drop)')
    print(f'render PSNR vs GT: coarse {psnr_c:.2f} -> split {psnr_s:.2f} (+{psnr_s-psnr_c:.2f} dB)')
    print(f'appearance PSNR (StPr vs AppGas): split {appear0:.2f} -> after re-opt {appear1:.2f} dB')
    print(f'binding Mahalanobis t*: coarse {maha_coarse:.3f} -> split {maha_split:.3f}')
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
