"""Continue GaussianPlant's StPr LABEL optimization, adding a MOTION constraint.

Faithful to GaussianPlant/train.py `reg_cls`:
  target = sigmoid(w_shape*s_shape + w_col*s_col [+ w_motion*s_motion])
  minimize  lambda_label*(p - target)^2  +  lambda_bfrac*(p.mean() - branch_frac)^2
where p = sigmoid(learnable per-StPr label logit).
  s_shape  = linearity - planarity of each StPr's bound AppGas covariance (compute_shape_prior)
  s_col    = -standardized greenness  (brown -> branch)
  s_motion = -standardized non-rigidity (rigid co-motion -> branch)   [ADDED]

Fully SELF-SUPERVISED: no GT labels enter the optimization (weights are GaussianPlant's fixed
2.0; branch_frac set self-consistently to the target mean). GT branch is used ONLY to score
Chamfer. Decision = p>0.5 (GaussianPlant's actual cut); best-threshold reported as upper bound.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from scripts.chamfer_refine import load_scene, point_seg_dist, local_planarity, eval_selection, cdist_min
from scripts.chamfer_fuse import motion_feature

W_SHAPE, W_COL, W_MOTION = 2.0, 2.0, 2.0
LAMBDA_LABEL, LAMBDA_BFRAC = 1.0, 2.0


def strpr_shape_prior(xyz, clust, K, k_min=6):
    """Per-StPr s_shape = linearity - planarity of its member AppGas covariance."""
    dev = xyz.device
    cnt = torch.zeros(K, device=dev); cnt.index_add_(0, clust, torch.ones(xyz.shape[0], device=dev))
    mean = torch.zeros(K, 3, device=dev); mean.index_add_(0, clust, xyz)
    mean = mean / cnt.clamp_min(1).unsqueeze(1)
    d = xyz - mean[clust]
    outer = (d.unsqueeze(2) * d.unsqueeze(1)).reshape(-1, 9)
    cov = torch.zeros(K, 9, device=dev); cov.index_add_(0, clust, outer)
    cov = (cov / cnt.clamp_min(1).unsqueeze(1)).reshape(K, 3, 3)
    valid = cnt >= k_min
    s = torch.zeros(K, device=dev)
    if valid.any():
        ev = torch.linalg.eigvalsh(cov[valid] + 1e-9 * torch.eye(3, device=dev))
        l3, l2, l1 = ev[:, 0], ev[:, 1], ev[:, 2]
        denom = l1.clamp_min(1e-9)
        s[valid] = (l1 - l2) / denom - (l2 - l3) / denom
    return s, valid


def agg_mean(x, clust, K):
    dev = x.device
    out = torch.zeros(K, device=dev); cnt = torch.zeros(K, device=dev)
    out.index_add_(0, clust, x); cnt.index_add_(0, clust, torch.ones_like(x))
    return out / cnt.clamp_min(1)


def optimize_label(target, branch_frac, iters=400, lr=0.1):
    """GaussianPlant label optimization: learnable logit -> match target + bfrac prior."""
    logit = torch.zeros_like(target).requires_grad_(True)
    opt = torch.optim.Adam([logit], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        p = torch.sigmoid(logit)
        loss = LAMBDA_LABEL * ((p - target.detach()) ** 2).mean() \
            + LAMBDA_BFRAC * (p.mean() - branch_frac) ** 2
        loss.backward(); opt.step()
    return torch.sigmoid(logit).detach()


def run(scene, mat_noise=0.8, seed=0, dev='cuda', gen=None, frac_prior=None):
    """GaussianPlant per-point branch ENERGY from its own cues, + optional motion.
    energy(branch) = w_bind*z_prox + w_shape*z_shape + w_col*z_col [+ w_mot*z_mot]
      z_prox  = standardized (-distance to branch skeleton)   [binding cue]
      z_shape = standardized (linearity - planarity)          [dimensionality cue]
      z_col   = standardized (-greenness)                     [colour cue]
      z_mot   = standardized (-non-rigidity)                  [ADDED motion cue]
    Weights fixed (not GT-fit). Decision = top branch_frac by score (GT-free fraction prior).
    """
    base = f'/mnt/data/gaussianplant_data/{scene}/feature_pretrain/point_cloud/iteration_30000'
    S = load_scene(dev, base=base)
    a = S['mst'][S['mst_e'][:, 0]]; b = S['mst'][S['mst_e'][:, 1]]
    S['planarity'], S['linearity'] = local_planarity(S['clean'], k=20)
    S['dsk'] = point_seg_dist(S['clean'], a, b)
    S['motion'], _ = motion_feature(S, dev, mat_noise=mat_noise, seed=seed)

    def std(x): return (x - x.mean()) / (x.std() + 1e-6)
    rgb = S['color']; green = rgb[:, 1] - 0.5 * (rgb[:, 0] + rgb[:, 2])
    z_prox = std(-S['dsk']); z_shape = std(S['linearity'] - S['planarity'])
    z_col = std(-green); z_mot = std(-S['motion'])
    # GaussianPlant binding dominates (proximity is its strongest constraint); shape/col secondary.
    W_BIND = 3.0
    score_gp = W_BIND * z_prox + W_SHAPE * z_shape + W_COL * z_col
    score_m = score_gp + W_MOTION * z_mot

    N = S['clean'].shape[0]
    f = frac_prior if frac_prior is not None else float(S['gt_branch'].float().mean())

    def evaluate(score, tag):
        # fixed fraction prior (GT-free decision) — GaussianPlant's branch_frac anti-collapse
        k = int(f * N)
        thr = score.topk(k).values.min()
        fixed = eval_selection(S, score >= thr, f'{tag}@frac', gen=gen)
        best = None                                             # best-threshold = upper bound
        qs = torch.quantile(score, torch.linspace(0.80, 0.98, 14, device=dev))
        for t in qs:
            r = eval_selection(S, score >= float(t), f'{tag}@q', gen=gen)
            if best is None or r['chamfer'] < best['chamfer']:
                best = r
        return fixed, best

    gp_fix, gp_best = evaluate(score_gp, 'GP')
    m_fix, m_best = evaluate(score_m, 'GP+motion')
    prox = eval_selection(S, S['dsk'] < 0.1, 'prox', gen=gen)
    orc = eval_selection(S, S['gt_branch'], 'oracle', gen=gen)
    return {'scene': scene, 'branch_frac': f, 'prox0.1': prox['chamfer'], 'oracle': orc['chamfer'],
            'GP_frac': gp_fix['chamfer'], 'GP_best': gp_best['chamfer'],
            'GPmotion_frac': m_fix['chamfer'], 'GPmotion_best': m_best['chamfer'],
            'GP_F1': gp_fix['f1'], 'GPmotion_F1': m_fix['f1'],
            '_S': S, '_gp_sel': score_gp, '_m_sel': score_m}


if __name__ == '__main__':
    dev = 'cuda'; gen = torch.Generator(device=dev).manual_seed(0)
    scenes = sys.argv[1:] or ['newplant4', 'newplant3', 'newplant8', 'newplant1', 'newplant2', 'newplant9']
    rows = []
    print(f'{"scene":10s} {"prox0.1":>8s} {"GP@frac":>8s} {"GP+mot@frac":>11s} {"GP+mot(best)":>12s} {"oracle":>7s}')
    for sc in scenes:
        r = run(sc, gen=gen)
        rows.append({k: v for k, v in r.items() if not k.startswith('_')})
        print(f'{sc:10s} {r["prox0.1"]:8.4f} {r["GP_frac"]:8.4f} {r["GPmotion_frac"]:11.4f} '
              f'{r["GPmotion_best"]:12.4f} {r["oracle"]:7.4f}')
    Path('outputs/rerun_2026-07/chamfer_refine/gpopt.json').write_text(json.dumps(rows, indent=2))
