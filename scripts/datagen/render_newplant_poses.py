"""Render a reference pose for each GaussianPlant reconstruction (newplant1..9),
using that scene's FIRST COLMAP image as the reference camera.

Only AppGas + COLMAP are needed (no branch tree), so this works for all newplants.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image

from data.gaussian_plant_loader import _find_app_ply, _load_app_gaussians
from data.colmap_loader import read_cameras_bin, read_images_bin, colmap_camera_to_renderer
from models.renderer.gaussian_renderer import GaussianRenderer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='/mnt/data/gaussianplant_data')
    ap.add_argument('--plants', nargs='+', default=[f'newplant{i}' for i in range(1, 10)])
    ap.add_argument('--height', type=int, default=832)
    ap.add_argument('--out-root', default='outputs/rerun_2026-07/gen/newplant_poses')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    dev = torch.device(args.device)
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    for name in args.plants:
        src = Path(args.data_root) / name
        try:
            app = _load_app_gaussians(_find_app_ply(src))
            cams = read_cameras_bin(src / 'sparse' / '0' / 'cameras.bin')
            imgs = read_images_bin(src / 'sparse' / '0' / 'images.bin')
            # FIRST image (sorted by name) as reference camera
            first = sorted(imgs.values(), key=lambda r: r['name'])[0]
            cam_intr = cams[first['cam_id']]
            aspect = cam_intr['w'] / cam_intr['h']
            H = args.height
            W = int(round(H * aspect)) // 2 * 2
            cam = colmap_camera_to_renderer(first, cam_intr, H, W)
            cam = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in cam.items()}
            rnd = GaussianRenderer(image_height=H, image_width=W, sh_degree=0).to(dev)
            img = rnd.render_frame(app.xyz.to(dev), app.scales.to(dev), app.rots.to(dev),
                                   app.opacities.to(dev), app.colors.to(dev), cam, shs=None).clamp(0, 1)
            arr = (img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(f'{args.out_root}/{name}.png')
            print(f'{name}: pose {W}x{H} from {first["name"]}  ({app.xyz.shape[0]} AppGas)')
        except Exception as e:
            print(f'{name}: FAILED -> {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
