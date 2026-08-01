"""Render the canonical (rest-pose) plant through the COLMAP camera and
overlay it with the video's first frame. Useful for spotting alignment issues
that the optimizer can't fix on its own (e.g. video diffusion drifted the plant
off-axis from the seed image)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision.io as tvio
from torchvision.transforms.functional import resize

from data.gaussian_plant_loader import load_gaussian_plant_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    parser.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    parser.add_argument('--video', required=True)
    parser.add_argument('--colmap-image', default='IMG_1388.JPG')
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=348)
    parser.add_argument('--out', default='outputs/canonical_vs_video.png')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    scene = load_gaussian_plant_scene(args.source, args.output_dir)

    from data.colmap_loader import (
        read_cameras_bin, read_images_bin, find_image_by_name, colmap_camera_to_renderer,
    )
    sparse = Path(args.source) / 'sparse' / '0'
    cams = read_cameras_bin(sparse / 'cameras.bin')
    imgs = read_images_bin(sparse / 'images.bin')
    rec = find_image_by_name(imgs, args.colmap_image)
    cam = colmap_camera_to_renderer(rec, cams[rec['cam_id']], args.H, args.W)
    cam = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cam.items()}

    from models.renderer.gaussian_renderer import GaussianRenderer
    renderer = GaussianRenderer(image_height=args.H, image_width=args.W, sh_degree=0).to(device)
    rendered = renderer.render_frame(
        scene.app.xyz.to(device),
        scene.app.scales.to(device),
        scene.app.rots.to(device),
        scene.app.opacities.to(device),
        scene.app.colors.to(device),
        cam,
        shs=None,
    ).clamp(0, 1).detach().cpu()

    frames, _, _ = tvio.read_video(args.video, pts_unit='sec')
    first = frames[0].permute(2, 0, 1).float() / 255.0
    first = resize(first[None], [args.H, args.W])[0]

    fig, axes = plt.subplots(1, 3, figsize=(args.W * 3 / 80.0, args.H / 80.0), dpi=80)
    axes[0].imshow(rendered.permute(1, 2, 0).numpy())
    axes[0].set_title('rendered canonical (COLMAP cam)', fontsize=8); axes[0].set_axis_off()
    axes[1].imshow(first.permute(1, 2, 0).numpy())
    axes[1].set_title('video frame 0 (after diffusion)', fontsize=8); axes[1].set_axis_off()
    blend = (rendered * 0.5 + first * 0.5).clamp(0, 1)
    axes[2].imshow(blend.permute(1, 2, 0).numpy())
    axes[2].set_title('50/50 blend', fontsize=8); axes[2].set_axis_off()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=80, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
