"""
Generate plant deformation videos using Wan2.1 image-to-video.
Takes rendered static frames from 3DGS and generates motion sequences
with varying force/interaction types.

Usage:
    python data/generation/gen_video.py --input_dir data/plants_3dgs --output_dir data/plants_video
    python data/generation/gen_video.py --input_dir data/plants_3dgs --smoke_test

Multi-GPU (shard mode):
    CUDA_VISIBLE_DEVICES=0 python data/generation/gen_video.py \\
        --input_dir data/plants_3dgs --output_dir data/plants_video \\
        --shard_id 0 --num_shards 8
"""
import os
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'
import sys
import re
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageFilter
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video

# Motion templates: {plant_desc} is filled at runtime from the directory name.
FORCE_CONFIGS = {
    "wind_light": {
        "prompt_template": (
            "A {plant_desc} swaying left and right in a gentle breeze, "
            "leaves trembling, branches oscillating softly side to side, "
            "natural wind-driven plant motion, static background, high quality"
        ),
        "guidance_scale": 6.5,
        "seed_offset": 0,
        "description": "gentle breeze causing subtle leaf movement",
    },
    "wind_intense": {
        "prompt_template": (
            "A {plant_desc} violently shaking in strong wind, "
            "branches whipping left and right, leaves fluttering chaotically, "
            "large amplitude swaying motion, static background, high quality"
        ),
        "guidance_scale": 7.5,
        "seed_offset": 100,
        "description": "strong wind causing large plant sway",
    },
    "external_light": {
        "prompt_template": (
            "A {plant_desc} tapped lightly, one branch bending and slowly "
            "bouncing back to rest, gentle elastic rebound, "
            "static background, high quality"
        ),
        "guidance_scale": 7.0,
        "seed_offset": 200,
        "description": "light touch causing local deformation",
    },
    "external_intense": {
        "prompt_template": (
            "A {plant_desc} struck hard, branches bending sharply sideways "
            "then springing back with strong elastic rebound, "
            "static background, high quality"
        ),
        "guidance_scale": 8.0,
        "seed_offset": 300,
        "description": "strong push causing significant bending",
    },
    "drag_light": {
        "prompt_template": (
            "A {plant_desc} swaying very slowly and settling, "
            "minimal side-to-side oscillation gradually dying down to rest, "
            "static background, high quality"
        ),
        "guidance_scale": 6.0,
        "seed_offset": 400,
        "description": "slow drag or gravity settling, minimal motion",
    },
}

NEGATIVE_PROMPT = (
    "turntable, 360 degree rotation, orbit, product shot, product photography, "
    "camera rotation, camera pan, camera zoom, camera movement, "
    "static, no motion, frozen, low quality, blur, watermark, text"
)


def dir_name_to_plant_desc(dir_name: str) -> str:
    """Parse directory name like 'a_single_adenium_..._s42' → 'adenium plant'."""
    # Strip trailing seed suffix (_s<digits>)
    name = re.sub(r'_s\d+$', '', dir_name)
    # Replace underscores with spaces
    name = name.replace('_', ' ')
    # Truncate to first ~8 words to keep prompt concise
    words = name.split()[:8]
    return ' '.join(words)


def preprocess_image(image_path: Path, width: int, height: int) -> Image.Image:
    """Replace black background with white, then letterbox-resize to target.

    No cropping is performed so that the camera geometry (azimuth=0,
    elevation=14, radius=2.0) used by GaussianRenderer stays consistent
    with the Wan2.1 input image.  The renderer and the video will share
    the same plant scale / position in frame, keeping the temporal-diff
    training loss meaningful.
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)

    # Replace near-black background pixels (3DGS renders have black bg)
    black_mask = (arr < 15).all(axis=-1)
    arr[black_mask] = 255
    img = Image.fromarray(arr)

    # Letterbox into target canvas — no crop, pose preserved.
    # Use light gray (not white) to avoid the model's "white background = turntable" prior.
    img.thumbnail((width, height), Image.LANCZOS)
    canvas = Image.new("RGB", (width, height), (200, 200, 200))
    x = (width - img.width) // 2
    y = (height - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def load_wan_pipeline(model_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"):
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    return pipe


def generate_video_from_image(pipe, image_path, output_dir, force_type,
                               plant_desc, num_frames=81, fps=16, seed=42,
                               height=480, width=832):
    cfg = FORCE_CONFIGS[force_type]

    output_dir = Path(output_dir)
    video_path = output_dir / "motion.mp4"
    if video_path.exists():
        print(f"    Skip (exists): {output_dir.name}")
        return video_path

    prompt = cfg["prompt_template"].format(plant_desc=plant_desc)
    image = preprocess_image(Path(image_path), width, height)

    generator = torch.Generator(device="cpu").manual_seed(seed + cfg["seed_offset"])
    frames = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        height=height,
        width=width,
        num_frames=num_frames,
        guidance_scale=cfg["guidance_scale"],
        generator=generator,
    ).frames[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(video_path), fps=fps)

    meta = {
        "source_image": str(image_path),
        "plant_desc": plant_desc,
        "force_type": force_type,
        "force_description": cfg["description"],
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "guidance_scale": cfg["guidance_scale"],
        "num_frames": num_frames,
        "fps": fps,
        "seed": seed + cfg["seed_offset"],
        "height": height,
        "width": width,
    }
    with open(output_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    return video_path


def get_canonical_frame(plant_dir):
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
    parser.add_argument('--input_dir', type=str, default='data/plants_3dgs')
    parser.add_argument('--output_dir', type=str, default='data/plants_video')
    parser.add_argument('--model', type=str, default='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers')
    parser.add_argument('--num_frames', type=int, default=81)
    parser.add_argument('--fps', type=int, default=16)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=832)
    parser.add_argument('--force_types', type=str, nargs='+',
                       default=list(FORCE_CONFIGS.keys()),
                       choices=list(FORCE_CONFIGS.keys()))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--smoke_test', action='store_true',
                       help='Quick test: 2 plants × 2 force types, 33 frames')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    all_plant_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    if not all_plant_dirs:
        print(f"No plant directories found in {input_dir}")
        return

    if args.smoke_test:
        all_plant_dirs = all_plant_dirs[:2]
        args.force_types = args.force_types[:2]
        args.num_frames = min(args.num_frames, 33)
        args.num_shards = 1
        args.shard_id = 0
        print(f"[SMOKE TEST] 2 plants × {args.force_types} × {args.num_frames} frames")

    my_plant_dirs = all_plant_dirs[args.shard_id::args.num_shards]
    total_mine = len(my_plant_dirs) * len(args.force_types)
    shard_label = f"shard {args.shard_id}/{args.num_shards}" if args.num_shards > 1 else "single"

    print(f"[{shard_label}] {len(my_plant_dirs)}/{len(all_plant_dirs)} plants → {total_mine} videos")
    print(f"Force types: {args.force_types}")

    if not my_plant_dirs:
        print("No work for this shard, exiting.")
        return

    print(f"Loading Wan2.1 pipeline: {args.model}")
    pipe = load_wan_pipeline(args.model)

    count = 0
    for plant_dir in my_plant_dirs:
        canonical = get_canonical_frame(plant_dir)
        if canonical is None:
            print(f"  Skip {plant_dir.name}: no gaussian.ply or canonical_frame.png")
            continue

        plant_desc = dir_name_to_plant_desc(plant_dir.name)

        for force_type in args.force_types:
            video_out = Path(args.output_dir) / f"{plant_dir.name}_{force_type}"
            count += 1
            print(f"[{count}/{total_mine}] {plant_dir.name} | {force_type} | desc: {plant_desc}")
            generate_video_from_image(
                pipe, canonical, video_out,
                force_type=force_type,
                plant_desc=plant_desc,
                num_frames=args.num_frames,
                fps=args.fps,
                seed=args.seed,
                height=args.height,
                width=args.width,
            )

    print(f"Done. Shard {args.shard_id} generated {count} videos in {args.output_dir}")


if __name__ == '__main__':
    main()
