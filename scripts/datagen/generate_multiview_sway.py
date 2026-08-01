"""Multi-view CONSISTENT swaying video: the SAME 4D deformation rendered from several
cameras at once, tiled into a grid. Because the plant is one 3D structure deforming under
one prescribed motion, the views are consistent by construction (unlike video-diffusion).

Cameras are novel views on the reconstruction's valid front arc (it is front-captured only),
obtained by rotating the reference COLMAP camera about the plant's vertical axis.
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
from scripts.datagen.generate_wind_sway import kinematic_theta


def Ry(a, device, dtype):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device=device, dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    ap.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    ap.add_argument('--ref-image', default='IMG_1388.JPG')
    ap.add_argument('--H', type=int, default=512); ap.add_argument('--W', type=int, default=342)
    ap.add_argument('--frames', type=int, default=48); ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--sway-amp', type=float, default=0.06)
    ap.add_argument('--branch-thresh', type=float, default=0.3); ap.add_argument('--leaf-thresh', type=float, default=0.22)
    ap.add_argument('--azimuths', type=float, nargs='+', default=[-30, -10, 10, 30])
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='outputs/rerun_2026-07/gen/multiview_sway.mp4')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    dev = torch.device(args.device)
    scene = load_gaussian_plant_scene(args.source, args.output_dir)
    tree = root_branch_graph(scene.branch, scene.tube)
    P = tree.nodes.to(dev); edges = tree.edges_oriented.to(dev); C = P.mean(0)
    ap_xyz = scene.app.xyz.to(dev)
    gs = dict(scales=scene.app.scales.to(dev), rots=scene.app.rots.to(dev),
              opac=scene.app.opacities.to(dev), col=scene.app.colors.to(dev))
    leaf_cent = (torch.stack([c.points.mean(0) for c in scene.leaves], 0).to(dev) if scene.leaves else None)

    binding = build_edge_binding(ap_xyz, P, edges, branch_thresh=args.branch_thresh, leaf_centroids=leaf_cent)
    is_branch = binding['is_branch']
    d_leaf = torch.cdist(ap_xyz, leaf_cent).min(dim=1).values if leaf_cent is not None else torch.full((ap_xyz.shape[0],), 1e9, device=dev)
    is_plant = (is_branch | (d_leaf < args.leaf_thresh)).view(-1, 1)

    chain = ArticulatedChain(tree).to(dev)
    theta_t = kinematic_theta(tree.depth.to(dev), args.frames, args.sway_amp, dev, seed=args.seed)
    with torch.no_grad():
        pos_t = []; rot_t = []
        for i in range(args.frames):
            p, r = chain.fk(theta_t[i], rest_pos=P); pos_t.append(p); rot_t.append(r)
        pos_t = torch.stack(pos_t); rot_t = torch.stack(rot_t)
        ap_traj = reconstruct_traj(pos_t, edges, binding, rot_t=rot_t)

    sp = Path(args.source) / 'sparse' / '0'
    cams = read_cameras_bin(sp / 'cameras.bin'); imgs = read_images_bin(sp / 'images.bin')
    rec = find_image_by_name(imgs, args.ref_image)
    base = colmap_camera_to_renderer(rec, cams[rec['cam_id']], args.H, args.W)
    base = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in base.items()}
    W2C = base['view_matrix'].transpose(0, 1); R, t = W2C[:3, :3], W2C[:3, 3]
    I3 = torch.eye(3, device=dev, dtype=R.dtype)
    view_cams = []
    for az in args.azimuths:
        Rot = Ry(math.radians(az), dev, R.dtype)
        Rp = R @ Rot; tp = R @ ((I3 - Rot) @ C) + t
        Wc = torch.eye(4, device=dev, dtype=R.dtype); Wc[:3, :3] = Rp; Wc[:3, 3] = tp
        view_cams.append({**base, 'view_matrix': Wc.transpose(0, 1).contiguous()})

    rnd = GaussianRenderer(image_height=args.H, image_width=args.W, sh_degree=0).to(dev)
    nc = len(view_cams); cols = int(math.ceil(math.sqrt(nc))); rows = int(math.ceil(nc / cols))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, quality=8, macro_block_size=1)
    with torch.no_grad():
        for tt in range(args.frames):
            ap_t = torch.where(is_plant, ap_traj[tt], ap_xyz)
            tiles = []
            for cam in view_cams:
                img = rnd.render_frame(ap_t, gs['scales'], gs['rots'], gs['opac'], gs['col'], cam, shs=None).clamp(0, 1)
                tiles.append(img)
            # pad to rows*cols
            while len(tiles) < rows * cols:
                tiles.append(torch.ones_like(tiles[0]))
            grid_rows = [torch.cat(tiles[r * cols:(r + 1) * cols], dim=2) for r in range(rows)]
            grid = torch.cat(grid_rows, dim=1)
            writer.append_data((grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    writer.close()
    print(f'wrote {args.out}  ({args.frames} frames, {nc} views az={args.azimuths}, tile {rows}x{cols})')


if __name__ == '__main__':
    main()
