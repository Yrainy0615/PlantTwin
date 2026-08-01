"""Reduce Chamfer( StPr-bounded branch AppGas , GT branch points ) by fusing
GaussianPlant static cues (colour / shape / skeleton proximity) with a motion
(rigidity) cue. Self-consistent scene: newplant4 feature_pretrain.

This module is the measurement + method harness; scripts call `run(cfg)`.
"""
from __future__ import annotations
import json, math
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData

B = '/mnt/data/gaussianplant_data/newplant4/feature_pretrain/point_cloud/iteration_30000'
SH_C0 = 0.28209479177387814


def _read(path):
    d = PlyData.read(str(path)); v = d['vertex']
    return torch.tensor(np.stack([v['x'], v['y'], v['z']], -1), dtype=torch.float32), v


def cdist_min(A, B_, chunk=2048):
    """min_j |A_i - B_j| for each i, chunked over i to bound memory."""
    out = torch.empty(A.shape[0], device=A.device)
    for s in range(0, A.shape[0], chunk):
        out[s:s+chunk] = torch.cdist(A[s:s+chunk], B_).min(1).values
    return out


def _color_scale(v):
    dc = torch.tensor(np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], -1), dtype=torch.float32)
    color = (SH_C0 * dc + 0.5).clamp(0, 1)
    scale = torch.exp(torch.tensor(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], -1), dtype=torch.float32))
    return color, scale


def load_scene(dev='cuda', base=None):
    base = base or B
    br, vbr = _read(f'{base}/point_cloud_branch.ply')
    lf, vlf = _read(f'{base}/point_cloud_leaf.ply')
    if Path(f'{base}/point_cloud_clean.ply').exists():
        clean_xyz, v = _read(f'{base}/point_cloud_clean.ply')
        color, scale = _color_scale(v)
    else:
        # clean = branch U leaf (these scenes ship no clean file); carry their colours/scales
        cb, sb = _color_scale(vbr); cl, sl = _color_scale(vlf)
        clean_xyz = torch.cat([br, lf], 0)
        color = torch.cat([cb, cl], 0); scale = torch.cat([sb, sl], 0)
    md = PlyData.read(f'{base}/mst.ply')
    mst_xyz = torch.tensor(np.stack([md['vertex']['x'], md['vertex']['y'], md['vertex']['z']], -1), dtype=torch.float32)
    e = md['edge']; mst_e = torch.tensor(np.stack([np.asarray(e['vertex1']), np.asarray(e['vertex2'])], -1), dtype=torch.long)
    S = dict(clean=clean_xyz.to(dev), color=color.to(dev), scale=scale.to(dev),
             branch=br.to(dev), leaf=lf.to(dev), mst=mst_xyz.to(dev), mst_e=mst_e.to(dev))
    # per-clean GT label: nearest of (branch,leaf) file membership
    db = cdist_min(clean_xyz.to(dev), S['branch'])
    dl = cdist_min(clean_xyz.to(dev), S['leaf'])
    S['gt_branch'] = (db < dl)                                    # [Nclean] bool
    return S


def point_seg_dist(pts, a, b, chunk=4096):
    """min distance from each point to the set of segments a->b. pts[N,3] a,b[E,3]."""
    d = b - a; L2 = (d * d).sum(-1).clamp_min(1e-12)
    out = torch.empty(pts.shape[0], device=pts.device)
    for s in range(0, pts.shape[0], chunk):
        p = pts[s:s+chunk]
        t = ((p[:, None] - a[None]) * d[None]).sum(-1) / L2[None]
        t = t.clamp(0, 1)
        proj = a[None] + t[..., None] * d[None]
        out[s:s+chunk] = (p[:, None] - proj).norm(dim=-1).min(1).values
    return out


def local_planarity(xyz, k=20, chunk=2048):
    """per-point (planarity, linearity) from PCA of k-NN. planarity=(l1-l2)/l0, linearity=(l0-l1)/l0."""
    N = xyz.shape[0]
    plan = torch.empty(N, device=xyz.device); lin = torch.empty(N, device=xyz.device)
    for s in range(0, N, chunk):
        q = xyz[s:s+chunk]
        dd = torch.cdist(q, xyz)
        idx = dd.topk(k, largest=False).indices                  # [c,k]
        nb = xyz[idx]                                            # [c,k,3]
        nb = nb - nb.mean(1, keepdim=True)
        cov = torch.einsum('cki,ckj->cij', nb, nb) / k
        ev = torch.linalg.eigvalsh(cov)                          # ascending [c,3]
        l0, l1, l2 = ev[:, 2], ev[:, 1], ev[:, 0]               # descending
        l0c = l0.clamp_min(1e-9)
        lin[s:s+chunk] = (l0 - l1) / l0c
        plan[s:s+chunk] = (l1 - l2) / l0c
    return plan, lin


def chamfer(A, B_, sample=None, gen=None):
    if A.shape[0] == 0:
        return float('inf'), float('inf'), float('inf')
    if sample and A.shape[0] > sample:
        idx = torch.randperm(A.shape[0], generator=gen, device=A.device)[:sample]; A = A[idx]
    d_ab = cdist_min(A, B_).mean()                               # branch-bound -> GT (precision)
    d_ba = cdist_min(B_, A).mean()                               # GT -> branch-bound (recall)
    return float(d_ab), float(d_ba), float(d_ab + d_ba)


def eval_selection(S, is_branch, tag, sample=8000, gen=None):
    sel = S['clean'][is_branch]
    ab, ba, cd = chamfer(sel, S['branch'], sample=sample, gen=gen)
    # classification vs GT
    gt = S['gt_branch']
    tp = int((is_branch & gt).sum()); fp = int((is_branch & ~gt).sum()); fn = int((~is_branch & gt).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1); f1 = 2*prec*rec/max(prec+rec,1e-9)
    return {'tag': tag, 'n_sel': int(is_branch.sum()), 'chamfer': cd, 'cham_precision': ab,
            'cham_recall': ba, 'f1': f1, 'prec': prec, 'rec': rec}


if __name__ == '__main__':
    dev = 'cuda'
    torch.manual_seed(0); gen = torch.Generator(device=dev).manual_seed(0)
    S = load_scene(dev)
    N = S['clean'].shape[0]
    print(f'scene: {N} clean AppGas, GT branch frac={float(S["gt_branch"].float().mean()):.3f}')
    a = S['mst'][S['mst_e'][:, 0]]; b = S['mst'][S['mst_e'][:, 1]]
    dsk = point_seg_dist(S['clean'], a, b)
    print(f'skeleton dist: GT-branch median={float(dsk[S["gt_branch"]].median()):.3f}  '
          f'GT-leaf median={float(dsk[~S["gt_branch"]].median()):.3f}')
    rows = []
    # baseline 1: pure skeleton proximity, sweep tau
    for tau in [0.1, 0.2, 0.3, 0.4, 0.5]:
        rows.append(eval_selection(S, dsk < tau, f'prox<{tau}', gen=gen))
    # baseline 2: colour (brown = branch): greenness low
    rgb = S['color']; green = rgb[:, 1] - 0.5 * (rgb[:, 0] + rgb[:, 2])
    for th in [0.0, 0.05, 0.1]:
        rows.append(eval_selection(S, green < th, f'brown(green<{th})', gen=gen))
    # oracle
    rows.append(eval_selection(S, S['gt_branch'], 'ORACLE(gt)', gen=gen))
    print(f'\n{"tag":22s} {"n":>7s} {"chamfer":>8s} {"prec_d":>7s} {"rec_d":>7s} {"F1":>5s} {"P":>5s} {"R":>5s}')
    for r in rows:
        print(f'{r["tag"]:22s} {r["n_sel"]:7d} {r["chamfer"]:8.4f} {r["cham_precision"]:7.4f} '
              f'{r["cham_recall"]:7.4f} {r["f1"]:5.2f} {r["prec"]:5.2f} {r["rec"]:5.2f}')
    Path('outputs/rerun_2026-07/chamfer_refine/baseline.json').write_text(json.dumps(rows, indent=2))
