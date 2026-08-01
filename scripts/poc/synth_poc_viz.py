"""Visualize the synthetic-GT PoC result: node recovery and error breakdown."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data.colmap_loader import (
    read_cameras_bin, read_images_bin, find_image_by_name, colmap_camera_to_renderer,
)
from data.gaussian_plant_loader import load_gaussian_plant_scene
from models.structure.graph_cleanup import root_branch_graph
from models.structure.edge_binding import build_edge_binding, reconstruct


def project(P, cam, H, W):
    M = torch.cat([P, torch.ones(P.shape[0], 1, device=P.device)], -1)
    clip = (cam['proj_matrix'].T @ M.T).T
    w = clip[:, 3:4].clamp_min(1e-6)
    ndc = clip[:, :3] / w
    u = (ndc[:, 0] * 0.5 + 0.5) * W
    v = (ndc[:, 1] * 0.5 + 0.5) * H
    return torch.stack([u, v], -1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pt', default='outputs/per_scene_optim/synth_poc/synth_poc.pt')
    ap.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    ap.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    ap.add_argument('--camera', default='IMG_1388.JPG')
    ap.add_argument('--H', type=int, default=640)
    ap.add_argument('--W', type=int, default=426)
    ap.add_argument('--out', default='outputs/per_scene_optim/synth_poc/recovery.png')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    d = torch.load(args.pt, map_location=device, weights_only=False)
    Pstar = d['Pstar'].to(device); P_init = d['P_init'].to(device); edges = d['edges'].to(device)
    P_out = d['results']['motion_out']['P'].to(device)
    P_in = d['results']['motion_in']['P'].to(device)

    scene = load_gaussian_plant_scene(args.source, args.output_dir, load_tube=True)
    tree = root_branch_graph(scene.branch, scene.tube)
    binding = build_edge_binding(scene.app.xyz.to(device), Pstar, edges, branch_thresh=d['args']['branch_thresh'])

    src = Path(args.source)
    cams = read_cameras_bin(src / 'sparse' / '0' / 'cameras.bin')
    imgs = read_images_bin(src / 'sparse' / '0' / 'images.bin')
    rec = find_image_by_name(imgs, args.camera)
    cam = colmap_camera_to_renderer(rec, cams[rec['cam_id']], args.H, args.W)
    cam = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cam.items()}

    from models.renderer.gaussian_renderer import GaussianRenderer
    renderer = GaussianRenderer(image_height=args.H, image_width=args.W, sh_degree=0).to(device)
    frame = renderer.render_frame(reconstruct(Pstar, edges, binding), scene.app.scales.to(device),
                                  scene.app.rots.to(device), scene.app.opacities.to(device),
                                  scene.app.colors.to(device), cam).clamp(0, 1).detach().permute(1, 2, 0).cpu().numpy()

    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 6))
    branch = (tree.depth > 1).cpu().numpy()
    titles = [('init (perturbed)', P_init, '#ff3b3b'), ('motion_out', P_out, '#ff9f1c'), ('motion_in', P_in, '#2ec4ff')]
    for i, (name, P, col) in enumerate(titles):
        axp = fig.add_subplot(1, 4, i + 1)
        axp.imshow(frame); axp.axis('off')
        uvs = project(Pstar, cam, args.H, args.W)
        uv = project(P, cam, args.H, args.W)
        axp.scatter(uvs[branch, 0], uvs[branch, 1], s=8, c='#33d17a', label='true', zorder=2)
        axp.scatter(uv[branch, 0], uv[branch, 1], s=8, c=col, label=name, zorder=3)
        for j in np.where(branch)[0]:
            axp.plot([uvs[j, 0], uv[j, 0]], [uvs[j, 1], uv[j, 1]], c=col, lw=0.4, alpha=0.6, zorder=1)
        r = d['results']['motion_out' if i == 1 else ('motion_in' if i == 2 else 'motion_out')]
        sub = (f"full {r['rmse_full']*100:.1f} depth {r['rmse_depth']*100:.1f}cm" if i > 0
               else f"full {d['init']['full']*100:.1f} depth {d['init']['depth']*100:.1f}cm")
        axp.set_title(f'{name}\n{sub}', fontsize=10)
        axp.legend(loc='lower right', fontsize=7)

    # bar chart
    axb = fig.add_subplot(1, 4, 4)
    labels = ['init', 'motion_out', 'motion_in']
    full = [d['init']['full'], d['results']['motion_out']['rmse_full'], d['results']['motion_in']['rmse_full']]
    depth = [d['init']['depth'], d['results']['motion_out']['rmse_depth'], d['results']['motion_in']['rmse_depth']]
    plane = [d['init']['plane'], d['results']['motion_out']['rmse_plane'], d['results']['motion_in']['rmse_plane']]
    x = np.arange(3); w = 0.26
    axb.bar(x - w, np.array(full) * 100, w, label='full', color='#888')
    axb.bar(x, np.array(plane) * 100, w, label='image-plane', color='#2a9d8f')
    axb.bar(x + w, np.array(depth) * 100, w, label='depth', color='#e76f51')
    axb.set_xticks(x); axb.set_xticklabels(labels, fontsize=9)
    axb.set_ylabel('node RMSE (cm)'); axb.legend(fontsize=8); axb.set_title('error breakdown', fontsize=10)
    axb.grid(axis='y', alpha=0.3)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out, dpi=120, bbox_inches='tight'); plt.close()
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
