"""Run the branch/leaf fusion on given scenes and produce the baseline-vs-FUSED
TP/FP/FN comparison figure (same style as newplant4's viz.png)."""
import sys, torch, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scripts.chamfer_refine import load_scene, point_seg_dist, local_planarity, eval_selection
from scripts.chamfer_fuse import motion_feature, fit_lr

dev = 'cuda'


def prep(scene):
    base = f'/mnt/data/gaussianplant_data/{scene}/feature_pretrain/point_cloud/iteration_30000'
    S = load_scene(dev, base=base)
    a = S['mst'][S['mst_e'][:, 0]]; b = S['mst'][S['mst_e'][:, 1]]
    S['dsk'] = point_seg_dist(S['clean'], a, b)
    S['planarity'], S['linearity'] = local_planarity(S['clean'], k=20)
    rgb = S['color']; S['green'] = rgb[:, 1] - 0.5 * (rgb[:, 0] + rgb[:, 2])
    S['motion'], _ = motion_feature(S, dev, mat_noise=0.8)
    S['logmot'] = (S['motion'] + 1e-6).log()
    return S


def best_sel(S, prob, gen):
    best = None
    for th in torch.linspace(0.15, 0.92, 20):
        sel = prob > float(th); r = eval_selection(S, sel, 'x', gen=gen)
        if best is None or r['chamfer'] < best[0]:
            best = (r['chamfer'], sel)
    return best


def viz_scene(scene, gen):
    S = prep(scene); y = S['gt_branch']
    def col(*n): return torch.stack([S[k] for k in n], 1)
    base = S['dsk'] < 0.1
    base_c = eval_selection(S, base, 'x', gen=gen)['chamfer']
    fused_prob = fit_lr(col('dsk', 'linearity', 'planarity', 'green', 'logmot'), y)
    fused_c, fused = best_sel(S, fused_prob, gen)
    xyz = S['clean'].cpu().numpy(); gt = y.cpu().numpy()
    rs = np.random.RandomState(0)
    fig, ax = plt.subplots(1, 2, figsize=(13, 8))
    for k, (sel, ttl, ch) in enumerate([(base.cpu().numpy(), 'baseline prox<0.1', base_c),
                                        (fused.cpu().numpy(), 'FUSED static+motion', fused_c)]):
        tp = sel & gt; fp = sel & ~gt; fn = ~sel & gt
        A = ax[k]
        for mask, c, s, lab in [(tp, '#2e7d32', 2, 'TP branch'), (fp, '#d1495b', 4, 'FP leaf->branch'),
                                (fn, '#1e6fd6', 6, 'FN missed branch')]:
            idx = np.where(mask)[0]
            if len(idx) > 6000: idx = rs.choice(idx, 6000, replace=False)
            A.scatter(xyz[idx, 0], xyz[idx, 1], s=s, c=c, alpha=0.5, linewidths=0,
                      label=f'{lab} ({int(mask.sum())})')
        A.set_title(f'{ttl}  Chamfer={ch:.4f}', fontsize=11)
        A.set_aspect('equal'); A.axis('off'); A.legend(loc='upper right', fontsize=8)
    fig.suptitle(scene, fontsize=13)
    plt.tight_layout()
    out = f'outputs/rerun_2026-07/chamfer_refine/viz_{scene}.png'
    plt.savefig(out, dpi=110); plt.close()
    print(f'{scene}: baseline {base_c:.4f} -> FUSED {fused_c:.4f}  ({base_c/fused_c:.1f}x)  saved {out}')


if __name__ == '__main__':
    gen = torch.Generator(device=dev).manual_seed(0)
    for scene in sys.argv[1:] or ['newplant1', 'newplant2', 'newplant9']:
        viz_scene(scene, gen)
