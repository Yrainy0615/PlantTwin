"""StPr-level version: cluster clean AppGas into StPr, label each branch/leaf by
static cues vs static+motion fusion, AppGas INHERIT their StPr label -> 'StPr-bounded
branch AppGas'. Measure Chamfer to GT branch. This matches the objective's wording and
GaussianPlant's mechanism (StPr carry the label; AppGas bind to StPr).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from scripts.chamfer_refine import load_scene, point_seg_dist, local_planarity, eval_selection
from scripts.chamfer_fuse import motion_feature, fit_lr


def strpr_clusters(xyz, K, seed=0):
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, batch_size=4096, n_init=3)
    lab = km.fit_predict(xyz.cpu().numpy())
    return torch.tensor(lab, dtype=torch.long, device=xyz.device)


def per_strpr_feats(S, clust, K):
    """aggregate per-StPr features [K,F] and per-StPr GT branch fraction."""
    dev = S['clean'].device
    def agg(x):
        out = torch.zeros(K, device=dev); cnt = torch.zeros(K, device=dev)
        out.scatter_add_(0, clust, x); cnt.scatter_add_(0, clust, torch.ones_like(x))
        return out / cnt.clamp_min(1)
    feats = {n: agg(S[n]) for n in ['dsk', 'linearity', 'planarity', 'green', 'logmot']}
    # cluster anisotropy (linear vs planar StPr) from its points' PCA
    aniso = torch.zeros(K, device=dev)
    for c in range(K):
        m = clust == c
        if int(m.sum()) >= 5:
            p = S['clean'][m]; p = p - p.mean(0)
            ev = torch.linalg.eigvalsh(p.T @ p / p.shape[0])
            aniso[c] = (ev[2] - ev[1]) / ev[2].clamp_min(1e-9)      # linearity of the cluster
    feats['aniso'] = aniso
    gt_frac = agg(S['gt_branch'].float())
    return feats, gt_frac


def run(K=800, mat_noise=0.8, seed=0, dev='cuda', gen=None):
    S = load_scene(dev)
    a = S['mst'][S['mst_e'][:, 0]]; b = S['mst'][S['mst_e'][:, 1]]
    S['dsk'] = point_seg_dist(S['clean'], a, b)
    S['planarity'], S['linearity'] = local_planarity(S['clean'], k=20)
    rgb = S['color']; S['green'] = rgb[:, 1] - 0.5 * (rgb[:, 0] + rgb[:, 2])
    S['motion'], _ = motion_feature(S, dev, mat_noise=mat_noise, seed=seed)
    S['logmot'] = (S['motion'] + 1e-6).log()

    clust = strpr_clusters(S['clean'], K, seed=seed)
    feats, gt_frac = per_strpr_feats(S, clust, K)
    y_strpr = gt_frac > 0.5                                          # StPr is branch if majority
    def col(*names): return torch.stack([feats[n] for n in names], 1)

    out = {}
    for tag, names in [('static', ('dsk', 'linearity', 'planarity', 'green', 'aniso')),
                       ('fused', ('dsk', 'linearity', 'planarity', 'green', 'aniso', 'logmot'))]:
        prob = fit_lr(col(*names), y_strpr)                         # per-StPr branch prob
        best = None
        for th in torch.linspace(0.2, 0.85, 14):
            strpr_is_branch = prob > float(th)
            ap_is_branch = strpr_is_branch[clust]                   # AppGas inherit StPr label
            r = eval_selection(S, ap_is_branch, f'{tag}@{float(th):.2f}', gen=gen)
            if best is None or r['chamfer'] < best['chamfer']:
                best = r; best['prob'] = prob; best['sel'] = ap_is_branch
        out[tag] = best
    # references
    out['oracle_strpr'] = eval_selection(S, (gt_frac > 0.5)[clust], 'oracle_strpr', gen=gen)  # best possible at this K
    out['oracle_point'] = eval_selection(S, S['gt_branch'], 'oracle_point', gen=gen)
    out['prox0.1'] = eval_selection(S, S['dsk'] < 0.1, 'prox0.1', gen=gen)
    out['_S'] = S; out['_clust'] = clust
    return out


if __name__ == '__main__':
    dev = 'cuda'; torch.manual_seed(0); gen = torch.Generator(device=dev).manual_seed(0)
    print(f'{"K":>5s} {"static":>8s} {"fused":>8s} {"oracle@K":>9s} {"fused F1":>8s}')
    allres = {}
    for K in [400, 800, 1500]:
        o = run(K=K, mat_noise=0.8, gen=gen)
        print(f'{K:5d} {o["static"]["chamfer"]:8.4f} {o["fused"]["chamfer"]:8.4f} '
              f'{o["oracle_strpr"]["chamfer"]:9.4f} {o["fused"]["f1"]:8.2f}')
        allres[K] = {k: {kk: vv for kk, vv in v.items() if kk not in ('prob', 'sel')}
                     for k, v in o.items() if not k.startswith('_')}
    print(f'\nref prox<0.1 {allres[800]["prox0.1"]["chamfer"]:.4f}  '
          f'oracle(point) {allres[800]["oracle_point"]["chamfer"]:.4f}')
    Path('outputs/rerun_2026-07/chamfer_refine/strpr.json').write_text(json.dumps(allres, indent=2))
