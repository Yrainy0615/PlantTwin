"""Generate novel-view multi-view data of a reconstructed plant by orbiting a COLMAP
camera around the plant's vertical (world +Y) axis.

The reconstruction is front-captured only, so a full 360° orbit shows empty geometry on
the back; we sweep a valid front arc (±az_range) and optionally a small elevation, giving
a smooth novel-view multi-view clip on faithful geometry. Reuses the exact COLMAP camera
convention (rotate R,t about the centroid) — no coordinate-system guessing.
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
from models.renderer.gaussian_renderer import GaussianRenderer


def Ry(a, device, dtype):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device=device, dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    ap.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    ap.add_argument('--ref-image', default='IMG_1388.JPG')
    ap.add_argument('--H', type=int, default=1024)
    ap.add_argument('--W', type=int, default=684)
    ap.add_argument('--az-range', type=float, default=45.0, help='± azimuth degrees (front arc)')
    ap.add_argument('--el-range', type=float, default=6.0, help='± elevation degrees')
    ap.add_argument('--frames', type=int, default=72)
    ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--out', default='outputs/rerun_2026-07/gen/multiview_orbit.mp4')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    dev = torch.device(args.device)
    scene = load_gaussian_plant_scene(args.source, args.output_dir)
    tree = root_branch_graph(scene.branch, scene.tube)
    C = tree.nodes.mean(0).to(dev)

    sp = Path(args.source) / 'sparse' / '0'
    cams = read_cameras_bin(sp / 'cameras.bin'); imgs = read_images_bin(sp / 'images.bin')
    rec = find_image_by_name(imgs, args.ref_image)
    cam = colmap_camera_to_renderer(rec, cams[rec['cam_id']], args.H, args.W)
    cam = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in cam.items()}
    W2C = cam['view_matrix'].transpose(0, 1)
    R, t = W2C[:3, :3], W2C[:3, 3]

    rnd = GaussianRenderer(image_height=args.H, image_width=args.W, sh_degree=0).to(dev)
    g = dict(means3D=scene.app.xyz.to(dev), scales=scene.app.scales.to(dev),
             rotations=scene.app.rots.to(dev), opacities=scene.app.opacities.to(dev),
             colors=scene.app.colors.to(dev))
    I3 = torch.eye(3, device=dev, dtype=R.dtype)

    # ping-pong azimuth (smooth there-and-back), gentle elevation bob
    ts = np.linspace(0, 1, args.frames, endpoint=False)
    az = np.sin(2 * np.pi * ts) * math.radians(args.az_range)
    el = np.sin(4 * np.pi * ts) * math.radians(args.el_range)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, quality=8, macro_block_size=1)
    Rx = lambda a: torch.tensor([[1,0,0],[0,math.cos(a),-math.sin(a)],[0,math.sin(a),math.cos(a)]],
                                device=dev, dtype=R.dtype)
    for phi, th in zip(az, el):
        Rot = Ry(float(phi), dev, R.dtype) @ Rx(float(th))
        Rp = R @ Rot
        tp = R @ ((I3 - Rot) @ C) + t
        W2Cp = torch.eye(4, device=dev, dtype=R.dtype); W2Cp[:3, :3] = Rp; W2Cp[:3, 3] = tp
        c2 = {**cam, 'view_matrix': W2Cp.transpose(0, 1).contiguous()}
        img = rnd.render_frame(g['means3D'], g['scales'], g['rotations'], g['opacities'],
                               g['colors'], c2, shs=None).clamp(0, 1)
        writer.append_data((img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8))
    writer.close()
    print(f'wrote {args.out}  ({args.frames} frames, az±{args.az_range}° el±{args.el_range}°, {args.H}x{args.W})')


if __name__ == '__main__':
    main()
