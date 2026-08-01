"""Render the synthesized sway+flutter motion (used as the rigidity cue in the Chamfer
experiment) as a video, per scene, from its first COLMAP camera."""
import sys, math
import numpy as np, torch
import imageio.v2 as imageio
from pathlib import Path
from data.gaussian_plant_loader import _load_app_gaussians
from data.colmap_loader import read_cameras_bin, read_images_bin, colmap_camera_to_renderer
from models.renderer.gaussian_renderer import GaussianRenderer
from scripts.chamfer_refine import load_scene, local_planarity
from scripts.chamfer_fuse import motion_feature

dev = 'cuda'


def render_scene(scene, T=30):
    base = f'/mnt/data/gaussianplant_data/{scene}/feature_pretrain/point_cloud/iteration_30000'
    S = load_scene(dev, base=base)
    S['planarity'], S['linearity'] = local_planarity(S['clean'], k=20)
    # full GS params for rendering, SAME order as load_scene clean = [branch, leaf]
    ab = _load_app_gaussians(Path(f'{base}/point_cloud_branch.ply'))
    al = _load_app_gaussians(Path(f'{base}/point_cloud_leaf.ply'))
    scales = torch.cat([ab.scales, al.scales]).to(dev)
    rots = torch.cat([ab.rots, al.rots]).to(dev)
    opac = torch.cat([ab.opacities, al.opacities]).to(dev)
    color = torch.cat([ab.colors, al.colors]).to(dev)
    # synthesize motion (visible sway + flutter)
    _, traj = motion_feature(S, dev, T=T, amp=0.22, flutter=0.09, mat_noise=0.8)
    # camera
    src = Path(f'/mnt/data/gaussianplant_data/{scene}')
    cams = read_cameras_bin(src / 'sparse/0/cameras.bin')
    imgs = read_images_bin(src / 'sparse/0/images.bin')
    first = sorted(imgs.values(), key=lambda r: r['name'])[0]
    ci = cams[first['cam_id']]
    H = 800; W = int(round(H * ci['w'] / ci['h'])) // 2 * 2
    cam = colmap_camera_to_renderer(first, ci, H, W)
    cam = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in cam.items()}
    rnd = GaussianRenderer(image_height=H, image_width=W, sh_degree=0).to(dev)
    frames = []
    for t in range(T):
        img = rnd.render_frame(traj[t], scales, rots, opac, color, cam, shs=None).clamp(0, 1)
        frames.append((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    out = f'outputs/rerun_2026-07/chamfer_refine/motion_{scene}.mp4'
    imageio.mimsave(out, frames, fps=15, quality=8)
    disp = np.mean([np.abs(np.asarray(frames[t], float) - np.asarray(frames[0], float)).mean() for t in range(T)])
    print(f'{scene}: {W}x{H} {T}f -> {out}  (mean motion {disp:.1f})')


if __name__ == '__main__':
    for sc in sys.argv[1:] or ['newplant1', 'newplant2', 'newplant9']:
        render_scene(sc)
