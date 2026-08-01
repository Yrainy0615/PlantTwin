"""Fuse GaussianPlant static cues (skeleton proximity, shape, colour) with a MOTION
rigidity cue to classify clean AppGas branch/leaf, minimizing Chamfer to GT branch.

Motion model (honest): under gusty wind the skeleton sways (FK); every AppGas rides its
nearest branch segment rigidly, and additionally FLUTTERS with amplitude driven mostly by
being leaf material (physical: thin flaps flutter, stiff branches do not) and modulated by
local planarity. The per-AppGas non-rigidity (local-Kabsch residual over the trajectory) is
then a *direct, noisy observation* of branch-vs-leaf, complementary to the indirect static
proxies. We compare the best fusion classifier WITH vs WITHOUT the motion feature.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import torch
from scripts.chamfer_refine import load_scene, point_seg_dist, local_planarity, cdist_min, eval_selection


def build_tree(mst_xyz, mst_e):
    S = mst_xyz.shape[0]
    adj = [[] for _ in range(S)]
    for u, v in mst_e.tolist():
        adj[u].append(v); adj[v].append(u)
    root = int(mst_xyz[:, 1].argmin())               # lowest = base
    parent = torch.full((S,), -1, dtype=torch.long)
    order = [root]; seen = {root}; head = 0
    while head < len(order):
        u = order[head]; head += 1
        for w in adj[u]:
            if w not in seen:
                seen.add(w); parent[w] = u; order.append(w)
    return parent, torch.tensor(order)


def exp_so3(w):
    th = w.norm(dim=-1, keepdim=True); a = w / th.clamp_min(1e-9)
    K = torch.zeros(*w.shape[:-1], 3, 3, device=w.device)
    K[..., 0, 1] = -a[..., 2]; K[..., 0, 2] = a[..., 1]
    K[..., 1, 0] = a[..., 2]; K[..., 1, 2] = -a[..., 0]
    K[..., 2, 0] = -a[..., 1]; K[..., 2, 1] = a[..., 0]
    I = torch.eye(3, device=w.device).expand_as(K)
    return I + torch.sin(th)[..., None] * K + (1 - torch.cos(th))[..., None] * (K @ K)


def fk_nodes(mst_xyz, parent, order, theta):
    S = mst_xyz.shape[0]; dev = mst_xyz.device
    Rj = exp_so3(theta)
    R = [None]*S; pos = [None]*S
    for i in order.tolist():
        p = int(parent[i])
        if p < 0:
            R[i] = torch.eye(3, device=dev); pos[i] = mst_xyz[i]
        else:
            R[i] = R[p] @ Rj[i]; pos[i] = pos[p] + R[i] @ (mst_xyz[i] - mst_xyz[p])
    return torch.stack(pos), torch.stack(R)


def motion_feature(S, dev, T=12, amp=0.15, flutter=0.06, seed=0, k=16,
                   mat_noise=0.8, branch_flex=0.2, track_noise=0.004):
    """Synthesize sway+flutter, return per-AppGas non-rigidity (log) as motion cue.

    Realism knobs (so motion is a NOISY material observation, not a GT readout):
      mat_noise    : log-normal spread of the hidden stiffness field (class overlap).
      branch_flex  : fraction of leaf-flutter that branches also get (thin tips flex).
      track_noise  : per-frame tracking jitter added to the observed trajectory.
    """
    g = torch.Generator(device=dev).manual_seed(seed)
    mst, e = S['mst'], S['mst_e']
    parent, order = build_tree(mst, e)
    an = cdist_min_idx(S['clean'], mst)                       # nearest node idx
    depth = torch.zeros(mst.shape[0], dtype=torch.long, device=dev)
    for i in order.tolist():
        p = int(parent[i]); depth[i] = 0 if p < 0 else depth[p]+1
    dmax = depth.max().clamp_min(1)
    ax = torch.randn(mst.shape[0], 3, generator=g, device=dev); ax = ax/ax.norm(dim=-1, keepdim=True)
    ts = torch.arange(T, device=dev).float()
    phi = 2*math.pi*torch.rand(mst.shape[0], generator=g, device=dev)
    N = S['clean'].shape[0]
    traj = torch.empty(T, N, 3, device=dev)
    # hidden material floppiness: leaves ~1, branches ~branch_flex, corrupted by log-normal
    # noise so the two classes OVERLAP (a stiff leaf, a whippy branch tip) -> motion imperfect.
    base = branch_flex + (1 - branch_flex) * S['gt_branch'].logical_not().float()
    noise = torch.exp(mat_noise * torch.randn(N, generator=g, device=dev) - 0.5*mat_noise**2)
    famp = flutter * base * noise
    fdir = torch.randn(N, 3, generator=g, device=dev); fdir = fdir/fdir.norm(dim=-1, keepdim=True)
    fph = 2*math.pi*torch.rand(N, generator=g, device=dev)
    for t in range(T):
        theta = (amp/float(dmax))*torch.sin(2*math.pi*ts[t]/T + phi)[:, None]*(depth.float()/dmax)[:, None]*ax
        pos, R = fk_nodes(mst, parent, order, theta)
        rigid = pos[an] + torch.einsum('nij,nj->ni', R[an], S['clean']-mst[an])
        fl = (famp*torch.sin(2*math.pi*ts[t]/T+fph))[:, None]*fdir
        jit = track_noise * torch.randn(N, 3, generator=g, device=dev)
        traj[t] = rigid + fl + jit
    return local_nonrigidity(traj, S['clean'], k=k), traj


def cdist_min_idx(A, Bset, chunk=2048):
    out = torch.empty(A.shape[0], dtype=torch.long, device=A.device)
    for s in range(0, A.shape[0], chunk):
        out[s:s+chunk] = torch.cdist(A[s:s+chunk], Bset).argmin(1)
    return out


def local_nonrigidity(traj, rest, k=16, chunk=1024):
    """per-point residual after best rigid fit of its knn neighborhood over frames."""
    N = rest.shape[0]; dev = rest.device
    out = torch.empty(N, device=dev)
    for s in range(0, N, chunk):
        q = rest[s:s+chunk]
        idx = torch.cdist(q, rest).topk(k, largest=False).indices          # [c,k]
        P0 = rest[idx]                                                     # [c,k,3]
        Pt = traj[:, idx]                                                  # [T,c,k,3]
        c0 = P0.mean(1, keepdim=True); ct = Pt.mean(2, keepdim=True)
        Pc = P0 - c0; Qc = Pt - ct                                        # [c,k,3],[T,c,k,3]
        H = torch.einsum('cki,tckj->tcij', Pc, Qc)
        U, _, Vt = torch.linalg.svd(H)
        d = torch.sign(torch.linalg.det(Vt.transpose(-1,-2) @ U.transpose(-1,-2)))
        D = torch.eye(3, device=dev).expand(H.shape[0], H.shape[1], 3, 3).clone(); D[..., 2, 2] = d
        R = Vt.transpose(-1,-2) @ D @ U.transpose(-1,-2)
        pred = torch.einsum('tcij,ckj->tcki', R, Pc) + ct
        out[s:s+chunk] = (Pt - pred).norm(dim=-1).mean((0, 2))            # mean over T,k
    return out


def fit_lr(feats, y, iters=800, lr=0.05):
    """logistic regression, standardized features. feats [N,F], y [N] bool."""
    X = (feats - feats.mean(0)) / feats.std(0).clamp_min(1e-6)
    w = torch.zeros(X.shape[1], device=X.device, requires_grad=True)
    b = torch.zeros(1, device=X.device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    yf = y.float()
    for _ in range(iters):
        opt.zero_grad()
        logit = X @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yf)
        loss.backward(); opt.step()
    with torch.no_grad():
        prob = torch.sigmoid(X @ w + b)
    return prob.detach()


def best_threshold_chamfer(S, prob, gen, tag):
    """sweep prob threshold, return the selection with min Chamfer."""
    best = None
    for th in torch.linspace(0.2, 0.9, 15):
        r = eval_selection(S, prob > float(th), f'{tag}@{float(th):.2f}', gen=gen)
        if best is None or r['chamfer'] < best['chamfer']:
            best = r
    return best


if __name__ == '__main__':
    dev = 'cuda'; torch.manual_seed(0); gen = torch.Generator(device=dev).manual_seed(0)
    S = load_scene(dev)
    a = S['mst'][S['mst_e'][:, 0]]; b = S['mst'][S['mst_e'][:, 1]]
    S['dsk'] = point_seg_dist(S['clean'], a, b)
    print('computing shape features...')
    S['planarity'], S['linearity'] = local_planarity(S['clean'], k=20)
    rgb = S['color']; S['green'] = rgb[:, 1] - 0.5 * (rgb[:, 0] + rgb[:, 2])
    y = S['gt_branch']
    def col(*names): return torch.stack([S[n] for n in names], 1)
    static = fit_lr(col('dsk', 'linearity', 'planarity', 'green'), y)
    static_best = best_threshold_chamfer(S, static, gen, 'STATIC')
    print(f'\nSTATIC (prox+shape+col): Chamfer {static_best["chamfer"]:.4f} F1 {static_best["f1"]:.2f}')
    print('ORACLE:', round(eval_selection(S, y, 'o', gen=gen)['chamfer'], 4),
          ' ref prox<0.1:', round(eval_selection(S, S['dsk'] < 0.1, 'o', gen=gen)['chamfer'], 4))

    print(f'\nMotion-realism sweep (mat_noise): motion is a NOISY material observation')
    print(f'{"mat_noise":10s} {"motion_only":>12s} {"static":>8s} {"FUSED":>8s}  {"fused F1":>8s}')
    allrows = {'static': static_best}
    for mn in [0.4, 0.8, 1.2, 1.6]:
        S['motion'], _ = motion_feature(S, dev, mat_noise=mn)
        S['logmot'] = (S['motion'] + 1e-6).log()
        m_only = best_threshold_chamfer(S, fit_lr(col('logmot'), y), gen, f'motion_mn{mn}')
        fused = best_threshold_chamfer(S, fit_lr(col('dsk', 'linearity', 'planarity', 'green', 'logmot'), y), gen, f'fused_mn{mn}')
        print(f'{mn:<10.1f} {m_only["chamfer"]:12.4f} {static_best["chamfer"]:8.4f} '
              f'{fused["chamfer"]:8.4f}  {fused["f1"]:8.2f}')
        allrows[f'mn{mn}'] = {'motion_only': m_only, 'fused': fused}
    Path('outputs/rerun_2026-07/chamfer_refine/fuse.json').write_text(
        json.dumps({k: (v if isinstance(v, dict) and 'tag' in v else v) for k, v in allrows.items()},
                   indent=2, default=lambda o: o))
