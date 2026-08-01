"""Generate a clean wind-sway video of a reconstructed plant (background static).

Branch tree is the motion basis. A prescribed, bounded per-joint sway (amplitude growing
toward the tips) is applied via FK; branch AppGas ride their edge and leaf AppGas swing
rigidly with their attachment node (edge_binding). Background/pot AppGas (far from any
branch edge and any leaf cluster) are held at rest, so only the plant moves.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio

from data.gaussian_plant_loader import load_gaussian_plant_scene
from data.colmap_loader import read_cameras_bin, read_images_bin, find_image_by_name, colmap_camera_to_renderer
from models.structure.graph_cleanup import root_branch_graph
from models.structure.edge_binding import build_edge_binding, reconstruct_traj
from simulation.articulated_chain import ArticulatedChain
from models.renderer.gaussian_renderer import GaussianRenderer


def kinematic_theta(depth, n_frames, amp, device, seed=0):
    """Prescribed bounded joint-angle sway [T,N,3]; amplitude scales with depth."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    N = depth.shape[0]
    axis = torch.randn(N, 3, generator=g).to(device); axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    phase = torch.rand(N, 1, generator=g).to(device) * 2 * math.pi
    dscale = (depth.float() / depth.float().clamp_min(1).max()).unsqueeze(-1).to(device)
    freq = 1.0 + 0.5 * torch.rand(N, 1, generator=g).to(device)
    t = torch.linspace(0, 2 * math.pi, n_frames, device=device).view(-1, 1, 1)
    sway = torch.sin(freq.unsqueeze(0) * t + phase.unsqueeze(0)) * (amp * dscale).unsqueeze(0)
    return sway * axis.unsqueeze(0)                                   # [T,N,3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    ap.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    ap.add_argument('--ref-image', default='IMG_1388.JPG')
    ap.add_argument('--H', type=int, default=1024); ap.add_argument('--W', type=int, default=684)
    ap.add_argument('--frames', type=int, default=72); ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--sway-amp', type=float, default=0.06)
    ap.add_argument('--branch-thresh', type=float, default=0.3)
    ap.add_argument('--leaf-thresh', type=float, default=0.22, help='keep leaf AppGas within this of a leaf cluster')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='outputs/rerun_2026-07/gen/wind_sway.mp4')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    dev = torch.device(args.device)
    scene = load_gaussian_plant_scene(args.source, args.output_dir)
    tree = root_branch_graph(scene.branch, scene.tube)
    P = tree.nodes.to(dev); edges = tree.edges_oriented.to(dev)
    ap_xyz = scene.app.xyz.to(dev)
    scales = scene.app.scales.to(dev); rots = scene.app.rots.to(dev)
    opac = scene.app.opacities.to(dev); colors = scene.app.colors.to(dev)
    leaf_cent = (torch.stack([c.points.mean(0) for c in scene.leaves], 0).to(dev) if scene.leaves else None)

    binding = build_edge_binding(ap_xyz, P, edges, branch_thresh=args.branch_thresh, leaf_centroids=leaf_cent)
    is_branch = binding['is_branch']
    # plant mask: branch-bound OR close to a leaf cluster; everything else = background (static)
    if leaf_cent is not None:
        d_leaf = torch.cdist(ap_xyz, leaf_cent).min(dim=1).values
        is_plant = is_branch | (d_leaf < args.leaf_thresh)
    else:
        is_plant = is_branch
    print(f'plant AppGas {int(is_plant.sum())}/{ap_xyz.shape[0]} (branch {int(is_branch.sum())}), rest static')

    chain = ArticulatedChain(tree).to(dev)
    theta_t = kinematic_theta(tree.depth.to(dev), args.frames, args.sway_amp, dev, seed=args.seed)
    pos_t, rot_t = [], []
    with torch.no_grad():
        for i in range(args.frames):
            p, r = chain.fk(theta_t[i], rest_pos=P)
            pos_t.append(p); rot_t.append(r)
        pos_t = torch.stack(pos_t); rot_t = torch.stack(rot_t)
        ap_traj = reconstruct_traj(pos_t, edges, binding, rot_t=rot_t)      # [T,M,3]

    sp = Path(args.source) / 'sparse' / '0'
    cams = read_cameras_bin(sp / 'cameras.bin'); imgs = read_images_bin(sp / 'images.bin')
    rec = find_image_by_name(imgs, args.ref_image)
    cam = colmap_camera_to_renderer(rec, cams[rec['cam_id']], args.H, args.W)
    cam = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in cam.items()}
    rnd = GaussianRenderer(image_height=args.H, image_width=args.W, sh_degree=0).to(dev)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, quality=8, macro_block_size=1)
    ipm = is_plant.view(-1, 1)
    with torch.no_grad():
        for t in range(args.frames):
            ap_t = torch.where(ipm, ap_traj[t], ap_xyz)
            img = rnd.render_frame(ap_t, scales, rots, opac, colors, cam, shs=None).clamp(0, 1)
            writer.append_data((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    writer.close()
    print(f'wrote {args.out}  ({args.frames} frames, amp {args.sway_amp}rad, {args.H}x{args.W})')


if __name__ == '__main__':
    main()
