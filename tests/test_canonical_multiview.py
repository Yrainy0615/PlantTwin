"""Canonical multi-view parity test.

Render the (frozen) AppGas at rest from several COLMAP cameras and compare to the GT
images. Establishes (a) that the workspace renderer + COLMAP cameras reproduce the
GaussianPlant static reconstruction across views, and (b) the baseline static-fit PSNR
that any structure optimization must preserve. This is the prerequisite for using a
canonical multi-view RGB term to anchor learnable StPr / branch-node positions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.colmap_loader import (
    read_cameras_bin, read_images_bin, find_image_by_name, colmap_camera_to_renderer,
)
from data.gaussian_plant_loader import load_gaussian_plant_scene


def load_gt(images_dir: Path, masks_dir: Path | None, name: str, H: int, W: int, device):
    im = Image.open(images_dir / name).convert('RGB').resize((W, H), Image.BILINEAR)
    gt = torch.tensor(np.asarray(im), dtype=torch.float32, device=device) / 255.0  # [H,W,3]
    mask = None
    if masks_dir is not None:
        cand = list(masks_dir.glob(Path(name).stem + '.*'))
        if cand:
            mm = Image.open(cand[0]).convert('L').resize((W, H), Image.NEAREST)
            mask = (torch.tensor(np.asarray(mm), dtype=torch.float32, device=device) / 255.0 > 0.5).float()
    return gt, mask


def psnr(a, b):
    mse = ((a - b) ** 2).mean().clamp_min(1e-10)
    return float(-10.0 * torch.log10(mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    ap.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    ap.add_argument('--views', nargs='+', default=['IMG_1388.JPG', 'IMG_1400.JPG', 'IMG_1410.JPG', 'IMG_1420.JPG'])
    ap.add_argument('--H', type=int, default=768)
    ap.add_argument('--W', type=int, default=512)
    ap.add_argument('--white-bg', action='store_true')
    ap.add_argument('--out', default='outputs/per_scene_optim/canonical_multiview.png')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    scene = load_gaussian_plant_scene(args.source, args.output_dir)
    src = Path(args.source)
    cams = read_cameras_bin(src / 'sparse' / '0' / 'cameras.bin')
    imgs = read_images_bin(src / 'sparse' / '0' / 'images.bin')
    masks_dir = src / 'masks' if (src / 'masks').exists() else None

    from models.renderer.gaussian_renderer import GaussianRenderer
    renderer = GaussianRenderer(image_height=args.H, image_width=args.W, sh_degree=0).to(device)
    bg = 1.0 if args.white_bg else 0.0
    xyz = scene.app.xyz.to(device); scales = scene.app.scales.to(device)
    rots = scene.app.rots.to(device); opac = scene.app.opacities.to(device)
    colors = scene.app.colors.to(device)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(args.views), figsize=(3 * len(args.views), 6))
    psnrs = []
    for j, name in enumerate(args.views):
        rec = find_image_by_name(imgs, name)
        if rec is None:
            print(f'[skip] {name} not in COLMAP'); continue
        cam = colmap_camera_to_renderer(rec, cams[rec['cam_id']], args.H, args.W)
        cam = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cam.items()}
        render = renderer.render_frame(xyz, scales, rots, opac, colors, cam, shs=None).clamp(0, 1)
        render = render.permute(1, 2, 0)  # [H,W,3]
        gt, mask = load_gt(src / 'images', masks_dir, name, args.H, args.W, device)
        if mask is not None:
            m = mask.unsqueeze(-1)
            comp_render = render * m + bg * (1 - m)
            comp_gt = gt * m + bg * (1 - m)
            p = psnr(comp_render, comp_gt)
        else:
            p = psnr(render, gt)
        psnrs.append(p)
        print(f'{name}: PSNR={p:.2f} dB' + ('' if mask is not None else ' (no mask)'))
        axes[0, j].imshow(render.detach().cpu().numpy()); axes[0, j].set_title(f'{name}\nrender {p:.1f}dB', fontsize=9); axes[0, j].axis('off')
        axes[1, j].imshow(gt.detach().cpu().numpy()); axes[1, j].set_title('GT', fontsize=9); axes[1, j].axis('off')

    print(f'\nmean PSNR over {len(psnrs)} views: {np.mean(psnrs):.2f} dB')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out, dpi=110, bbox_inches='tight'); plt.close(fig)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
