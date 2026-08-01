"""Visualize the integrated joint optimizer: depth + topology, motion_in vs motion_out."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pt', default='outputs/per_scene_optim/fuse/joint.pt')
    ap.add_argument('--out', default='outputs/per_scene_optim/fuse/joint_verdict.png')
    args = ap.parse_args()
    d = torch.load(args.pt, map_location='cpu', weights_only=False)
    mo, mi = d['results']['motion_out'], d['results']['motion_in']

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    # depth convergence
    for r, name, c in [(mo, 'motion_out', '#ff9f1c'), (mi, 'motion_in', '#2ec4ff')]:
        steps = [h['step'] for h in r['hist']]
        ax[0].plot(steps, [h['depth'] * 100 for h in r['hist']], '-o', ms=3, color=c, label=name)
    ax[0].set_xlabel('step'); ax[0].set_ylabel('node depth RMSE (cm)')
    ax[0].set_title('StPr geometry: depth recovery', fontsize=11)
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

    # topology AUC convergence
    for r, name, c in [(mo, 'motion_out', '#ff9f1c'), (mi, 'motion_in', '#2ec4ff')]:
        steps = [h['step'] for h in r['hist']]
        ax[1].plot(steps, [h['auc'] for h in r['hist']], '-o', ms=3, color=c, label=name)
    ax[1].set_xlabel('step'); ax[1].set_ylabel('topology ROC-AUC')
    ax[1].set_title('branch graph: edge classification', fontsize=11)
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3); ax[1].set_ylim(0.5, 1.02)

    # final bars
    labels = ['depth RMSE\n(cm, lower=better)', 'topo AUC\n(higher=better)', 'topo prec\n(higher=better)']
    vout = [mo['depth'] * 100, mo['auc'], mo['prec']]
    vin = [mi['depth'] * 100, mi['auc'], mi['prec']]
    x = np.arange(3); w = 0.35
    ax[2].bar(x - w / 2, vout, w, label='motion_out', color='#ff9f1c')
    ax[2].bar(x + w / 2, vin, w, label='motion_in', color='#2ec4ff')
    ax[2].set_xticks(x); ax[2].set_xticklabels(labels, fontsize=8)
    ax[2].set_title('joint final verdict', fontsize=11); ax[2].legend(fontsize=9); ax[2].grid(axis='y', alpha=0.3)
    for i, (a, b) in enumerate(zip(vout, vin)):
        ax[2].text(i - w / 2, a, f'{a:.2f}', ha='center', va='bottom', fontsize=8)
        ax[2].text(i + w / 2, b, f'{b:.2f}', ha='center', va='bottom', fontsize=8)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out, dpi=120, bbox_inches='tight'); plt.close()
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
