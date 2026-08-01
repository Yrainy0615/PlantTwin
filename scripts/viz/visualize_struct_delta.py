"""Overlay learned `delta_rest_pos` on the rendered canonical-pose frame.

Each tree node is drawn as a dot (sized by |delta|), with a red arrow pointing
in the direction of its delta vector (scaled for visibility). The biggest
arrows are where the residual-driven training says the existing topology can't
explain the observed motion — i.e., candidates for densification.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.colmap_loader import (
    read_cameras_bin, read_images_bin, find_image_by_name, colmap_camera_to_renderer,
)
from data.gaussian_plant_loader import load_gaussian_plant_scene
from models.structure.graph_cleanup import root_branch_graph
from models.structure.leaf_attachment import infer_leaf_attachments
from models.structure.skinning import compute_skinning, BoneSkinning, apply_skinning


def project(points: torch.Tensor, camera: dict, H: int, W: int) -> torch.Tensor:
    P = camera['proj_matrix'].T
    homog = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=-1)
    clip = (P @ homog.T).T
    w = clip[:, 3:4].clamp_min(1e-6)
    ndc = clip[:, :3] / w
    u = (ndc[:, 0] * 0.5 + 0.5) * W
    v = (ndc[:, 1] * 0.5 + 0.5) * H
    return torch.stack([u, v], dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    parser.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    parser.add_argument('--params', required=True)
    parser.add_argument('--colmap-image', default='IMG_1388.JPG')
    parser.add_argument('--H', type=int, default=1088)
    parser.add_argument('--W', type=int, default=736)
    parser.add_argument('--arrow-scale', type=float, default=20.0,
                        help='Visual amplification of delta vectors (real magnitude is in meters).')
    parser.add_argument('--top-k', type=int, default=12,
                        help='Annotate the K nodes with largest |delta|.')
    parser.add_argument('--skin-max-dist', type=float, default=0.3)
    parser.add_argument('--out', default='outputs/per_scene_optim/struct_delta_overlay.png')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    scene = load_gaussian_plant_scene(args.source, args.output_dir)

    p = torch.load(args.params, map_location='cpu', weights_only=False)
    if 'densified_tree' in p:
        from models.structure.graph_cleanup import RootedBranchTree
        dt = p['densified_tree']
        tree = RootedBranchTree(
            nodes=dt['nodes'].cpu(), root_idx=int(dt['root_idx']),
            parent=dt['parent'].cpu(), depth=dt['depth'].cpu(),
            edges_oriented=dt['edges_oriented'].cpu(), edge_length=dt['edge_length'].cpu(),
            edge_radius=dt['edge_radius'].cpu(), edge_type=dt['edge_type'].cpu(),
            subtree_size=dt['subtree_size'].cpu(),
        )
        print(f'[viz] using densified tree from params: N={tree.nodes.shape[0]}')
    else:
        tree = root_branch_graph(scene.branch, scene.tube)
    attachments = infer_leaf_attachments(scene.leaves, tree)
    skinning = compute_skinning(scene.app.xyz, tree, attachments, max_dist=args.skin_max_dist)

    # Camera
    sparse = Path(args.source) / 'sparse' / '0'
    cams = read_cameras_bin(sparse / 'cameras.bin')
    imgs = read_images_bin(sparse / 'images.bin')
    rec = find_image_by_name(imgs, args.colmap_image)
    cam = colmap_camera_to_renderer(rec, cams[rec['cam_id']], args.H, args.W)
    cam = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cam.items()}

    if 'delta_rest_pos' not in p:
        raise SystemExit('params has no delta_rest_pos — train with --learn-struct first.')
    delta = p['delta_rest_pos'].to(device)
    nodes = tree.nodes.to(device)
    nodes_eff = nodes + delta

    # Render canonical pose with the modified rest (= what the optimizer believes the structure is)
    from models.renderer.gaussian_renderer import GaussianRenderer
    renderer = GaussianRenderer(image_height=args.H, image_width=args.W, sh_degree=0).to(device)
    skin_d = BoneSkinning(
        bone_idx=skinning.bone_idx.to(device),
        leaf_idx=skinning.leaf_idx.to(device),
        local_bone=skinning.local_bone.to(device),
        local_leaf=skinning.local_leaf.to(device),
    )
    N = tree.nodes.shape[0]
    I3 = torch.eye(3, device=device).expand(N, 3, 3).contiguous()
    leaf_parent = torch.tensor(
        [int(tree.edges_oriented[a.parent_edge_idx, 0].item()) for a in attachments],
        dtype=torch.long, device=device,
    )
    leaf_centers_rest = torch.stack([a.disk_center for a in attachments]).to(device) + delta[leaf_parent]
    I_leaf = torch.eye(3, device=device).expand(len(attachments), 3, 3).contiguous()
    ap_rest = apply_skinning(
        skin_d, nodes_eff, I3, tree.edges_oriented.to(device),
        leaf_centers_rest, I_leaf, scene.app.xyz.to(device),
    )
    # Re-correct ApP positions to preserve canonical pose: subtract delta[parent]
    # — match how training applies the delta (LBS uses local_eff = local - delta[parent]).
    if (skin_d.bone_idx >= 0).any():
        mask = skin_d.bone_idx >= 0
        parent_idx = tree.edges_oriented.to(device)[skin_d.bone_idx[mask].clamp_min(0), 0]
        # ap_rest currently uses local_bone unchanged; offset back by delta[parent]
        ap_rest[mask] = ap_rest[mask] - delta[parent_idx]
    if (skin_d.leaf_idx >= 0).any():
        mask = skin_d.leaf_idx >= 0
        l_idx = skin_d.leaf_idx[mask].clamp_min(0)
        ap_rest[mask] = ap_rest[mask] - delta[leaf_parent[l_idx]]

    frame = renderer.render_frame(
        ap_rest, scene.app.scales.to(device), scene.app.rots.to(device),
        scene.app.opacities.to(device), scene.app.colors.to(device), cam, shs=None,
    ).clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy()

    # Project node centers + delta endpoints
    uv0 = project(nodes, cam, args.H, args.W).cpu().numpy()
    uv1 = project(nodes + delta * args.arrow_scale, cam, args.H, args.W).cpu().numpy()

    delta_mag = delta.norm(dim=-1).cpu().numpy()
    p_mag = float(np.percentile(delta_mag, 95))
    print(f'delta |.| stats: min={delta_mag.min():.4e} median={np.median(delta_mag):.4e} '
          f'p95={p_mag:.4e} max={delta_mag.max():.4e}')

    fig, ax = plt.subplots(1, 1, figsize=(args.W / 80.0, args.H / 80.0), dpi=80)
    ax.imshow(frame)
    ax.set_axis_off()

    # Color nodes by depth (cool→warm), size by delta magnitude
    depth = tree.depth.cpu().numpy()
    sizes = 20 + (delta_mag / (p_mag + 1e-8)).clip(0, 2.0) * 80
    ax.scatter(uv0[:, 0], uv0[:, 1], s=sizes, c=depth, cmap='viridis', alpha=0.85, edgecolors='white', linewidths=0.4)

    # Arrows for non-trivial deltas
    thresh = max(p_mag * 0.3, delta_mag.max() * 0.05)
    for i in range(len(nodes)):
        if delta_mag[i] < thresh:
            continue
        ax.annotate(
            '', xy=(uv1[i, 0], uv1[i, 1]), xytext=(uv0[i, 0], uv0[i, 1]),
            arrowprops=dict(arrowstyle='->', color='#ff3030', lw=1.4, alpha=0.9),
        )

    # Highlight top-K
    top_idx = np.argsort(delta_mag)[-args.top_k:][::-1]
    for rank, i in enumerate(top_idx):
        ax.text(
            uv0[i, 0] + 5, uv0[i, 1] - 5,
            f'#{rank+1} n{i}  |Δ|={delta_mag[i] * 100:.2f}cm',
            color='#ffd000', fontsize=8, weight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.55),
        )

    bbox_extent = float((tree.nodes.max(0).values - tree.nodes.min(0).values).max())
    ax.text(
        12, 24,
        f'arrow scale x{args.arrow_scale}   bbox={bbox_extent*100:.1f}cm   '
        f'mean |Δ|={delta_mag.mean()*100:.2f}cm   max |Δ|={delta_mag.max()*100:.2f}cm',
        color='white', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.55),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
