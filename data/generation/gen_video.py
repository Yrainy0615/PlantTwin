"""
Generate plant deformation videos using Stable Video Diffusion (SVD).
Takes rendered static frames from 3DGS and generates motion sequences
with varying force/interaction types.

Usage:
    python data/generation/gen_video.py --input_dir data/plants_3dgs --output_dir data/plants_video
    python data/generation/gen_video.py --input_dir data/plants_3dgs --smoke_test
"""
import os
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'
import sys
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

FORCE_CONFIGS = {
    "wind_light": {
        "motion_bucket_id": 80,
        "description": "gentle breeze causing subtle leaf movement",
    },
    "wind_intense": {
        "motion_bucket_id": 180,
        "description": "strong wind causing large plant sway",
    },
    "external_light": {
        "motion_bucket_id": 100,
        "description": "light touch or poke causing local deformation",
    },
    "external_intense": {
        "motion_bucket_id": 200,
        "description": "strong push causing significant bending",
    },
    "drag_light": {
        "motion_bucket_id": 60,
        "description": "slow drag or gravity settling, minimal motion",
    },
}


def load_svd_pipeline(model_id="stabilityai/stable-video-diffusion-img2vid-xt"):
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16",
    )
    pipe.to("cuda")
    return pipe


def generate_video_from_image(pipe, image_path, output_dir, force_type,
                                num_frames=25, fps=7, seed=42):
    """Generate motion video from a static plant image with specified force type."""
    cfg = FORCE_CONFIGS[force_type]
    image = Image.open(image_path).convert("RGB").resize((1024, 576))

    generator = torch.manual_seed(seed)
    frames = pipe(
        image,
        num_frames=num_frames,
        decode_chunk_size=8,
        generator=generator,
        motion_bucket_id=cfg["motion_bucket_id"],
    ).frames[0]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = output_dir / "motion.mp4"
    export_to_video(frames, str(video_path), fps=fps)

    meta = {
        "source_image": str(image_path),
        "force_type": force_type,
        "force_description": cfg["description"],
        "motion_bucket_id": cfg["motion_bucket_id"],
        "num_frames": num_frames,
        "fps": fps,
        "seed": seed,
    }
    with open(output_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    return video_path


def get_canonical_frame(plant_dir):
    """Get canonical frame, using pre-rendered one if available."""
    canonical = plant_dir / "canonical_frame.png"
    if canonical.exists():
        return canonical

    ply_path = plant_dir / "gaussian.ply"
    if not ply_path.exists():
        return None

    print(f"    Rendering canonical frame for {plant_dir.name}")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../third_party/TRELLIS'))
    from trellis.representations import Gaussian
    from trellis.utils import render_utils

    gaussian = Gaussian.load_ply(str(ply_path))
    video = render_utils.render_video(gaussian)['color']
    frame = Image.fromarray(video[0])
    frame.save(str(canonical))
    return canonical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/plants_3dgs',
                       help='Directory with plant 3DGS (from gen_3dgs.py)')
    parser.add_argument('--output_dir', type=str, default='data/plants_video')
    parser.add_argument('--model', type=str, default='stabilityai/stable-video-diffusion-img2vid-xt')
    parser.add_argument('--num_frames', type=int, default=25)
    parser.add_argument('--force_types', type=str, nargs='+',
                       default=list(FORCE_CONFIGS.keys()),
                       choices=list(FORCE_CONFIGS.keys()),
                       help='Force types to generate videos for')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke_test', action='store_true',
                       help='Quick test: limit to 20 videos total')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    plant_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    if not plant_dirs:
        print(f"No plant directories found in {input_dir}")
        return

    if args.smoke_test:
        plant_dirs = plant_dirs[:4]
        args.force_types = args.force_types[:5]
        max_videos = 20
        print(f"[SMOKE TEST] Limited to {len(plant_dirs)} plants, up to {max_videos} videos total")

    total = len(plant_dirs) * len(args.force_types)
    if args.smoke_test:
        total = min(total, max_videos)

    print(f"Loading SVD pipeline: {args.model}")
    pipe = load_svd_pipeline(args.model)

    print(f"Generating videos: {len(plant_dirs)} plants x {len(args.force_types)} force types = {total} videos")
    print(f"Force types: {args.force_types}")

    count = 0
    for i, plant_dir in enumerate(plant_dirs):
        canonical = get_canonical_frame(plant_dir)
        if canonical is None:
            print(f"  Skip {plant_dir.name}: no gaussian.ply or canonical_frame.png")
            continue

        for force_type in args.force_types:
            if args.smoke_test and count >= max_videos:
                break

            video_out = Path(args.output_dir) / f"{plant_dir.name}_{force_type}"
            count += 1
            print(f"[{count}/{total}] {plant_dir.name} | {force_type}")
            generate_video_from_image(
                pipe, canonical, video_out,
                force_type=force_type,
                num_frames=args.num_frames,
                seed=args.seed,
            )

        if args.smoke_test and count >= max_videos:
            break

    print(f"Done. Generated {count} videos in {args.output_dir}")


if __name__ == '__main__':
    main()
