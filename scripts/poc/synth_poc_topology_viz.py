"""Visualize the topology PoC: rigidity (3D) vs geometry for branch-graph recovery."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def roc_curve(score, label):
    s = score.numpy(); y = label.numpy().astype(np.int64)
    order = np.argsort(-s); y = y[order]
    P = y.sum(); Nn = len(y) - P
    tpr = np.cumsum(y) / P; fpr = np.cumsum(1 - y) / Nn
    return np.concatenate([[0], fpr]), np.concatenate([[0], tpr])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pt', default='outputs/per_scene_optim/synth_poc/topo_poc.pt')
    ap.add_argument('--out', default='outputs/per_scene_optim/synth_poc/topology.png')
    args = ap.parse_args()
    d = torch.load(args.pt, map_location='cpu', weights_only=False)
    is_true = d['is_true']; geo = d['geo_score']; rig = -d['strain3d']
    strain = d['strain3d']

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # ROC
    fg, tg = roc_curve(geo, is_true); fr, tr = roc_curve(rig, is_true)
    ax[0].plot(fg, tg, label=f"geometry (motion_out) AUC={d['res']['geometry only            [motion_out]'][0]:.3f}", color='#ff9f1c', lw=2)
    ax[0].plot(fr, tr, label=f"rigidity 3D (motion_in) AUC={d['res']['motion rigidity 3D       [motion_in]'][0]:.3f}", color='#2ec4ff', lw=2)
    ax[0].plot([0, 1], [0, 1], '--', color='#888', lw=1)
    ax[0].set_xlabel('false positive rate'); ax[0].set_ylabel('true positive rate')
    ax[0].set_title('edge classification ROC\n(true vs spurious candidate edges)', fontsize=11)
    ax[0].legend(fontsize=9, loc='lower right'); ax[0].grid(alpha=0.3)

    # strain histogram (log)
    st = strain.clamp_min(1e-8).log10().numpy()
    ax[1].hist(st[is_true.numpy()], bins=40, alpha=0.7, color='#33d17a', label='true edges', density=True)
    ax[1].hist(st[~is_true.numpy()], bins=40, alpha=0.6, color='#ff3b3b', label='spurious edges', density=True)
    ax[1].set_xlabel('log10(relative edge strain under motion)')
    ax[1].set_title('true segments stay rigid;\nspurious cross-limb edges stretch', fontsize=11)
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

    # precision bar
    names = ['geometry\n(motion_out)', 'rigidity3D\n(motion_in)']
    aucs = [d['res']['geometry only            [motion_out]'][0], d['res']['motion rigidity 3D       [motion_in]'][0]]
    precs = [d['res']['geometry only            [motion_out]'][1], d['res']['motion rigidity 3D       [motion_in]'][1]]
    x = np.arange(2); w = 0.35
    ax[2].bar(x - w / 2, aucs, w, label='ROC-AUC', color='#4f8cff')
    ax[2].bar(x + w / 2, precs, w, label=f'precision@|E*|', color='#2a9d8f')
    ax[2].set_xticks(x); ax[2].set_xticklabels(names, fontsize=9)
    ax[2].set_ylim(0, 1.05); ax[2].legend(fontsize=9); ax[2].grid(axis='y', alpha=0.3)
    ax[2].set_title('branch-graph recovery quality', fontsize=11)
    for i, (au, pr) in enumerate(zip(aucs, precs)):
        ax[2].text(i - w / 2, au + 0.01, f'{au:.2f}', ha='center', fontsize=8)
        ax[2].text(i + w / 2, pr + 0.01, f'{pr:.2f}', ha='center', fontsize=8)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out, dpi=120, bbox_inches='tight'); plt.close()
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
