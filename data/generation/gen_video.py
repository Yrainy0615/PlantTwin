"""
Generate plant motion videos using Wan2.2 I2V (A14B MoE).

Standard mode (30 steps):
    python data/generation/gen_video.py --input_dir data/plants_3dgs

LightX2V 4-step mode:
    python data/generation/gen_video.py --input_dir data/plants_3dgs \
        --lightx2v_high <high_noise_lora.safetensors> \
        --lightx2v_low  <low_noise_lora.safetensors>

Multi-GPU (shard mode):
    CUDA_VISIBLE_DEVICES=0 python data/generation/gen_video.py \
        --input_dir data/plants_3dgs --output_dir data/plants_video \
        --shard_id 0 --num_shards 8
"""
import os
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'
import sys
import re
import argparse
from pathlib import Path

import gc
import torch
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

FORCE_CONFIGS = {
    "wind_left_gentle": {
        "prompt_template": "{plant_desc}, gentle breeze from the left",
        "guidance_scale": 5.0,
        "seed_offset": 0,
    },
    "wind_right_gentle": {
        "prompt_template": "{plant_desc}, gentle breeze from the right",
        "guidance_scale": 5.0,
        "seed_offset": 100,
    },
    "wind_left_strong": {
        "prompt_template": "{plant_desc}, strong wind from the left",
        "guidance_scale": 5.0,
        "seed_offset": 200,
    },
    "wind_right_strong": {
        "prompt_template": "{plant_desc}, strong wind from the right",
        "guidance_scale": 5.0,
        "seed_offset": 300,
    },
    # --- Material-aware, WHOLE-BRANCH sway (woody = stiff, branches move as a unit) ---
    "wb_gentle": {
        "prompt_template": (
            "{plant_desc}. A gentle steady breeze. The plant is woody and stiff, so its "
            "branches sway slowly and TOGETHER as one coherent unit; the whole crown moves "
            "in unison and the leaves barely flutter on their own. The base stays anchored. "
            "brightly front-lit with bright natural daylight, vivid green foliage, plain light-gray background, single static camera, no zoom, no pan."
        ),
        "guidance_scale": 5.0,
        "seed_offset": 500,
    },
    "wb_side": {
        "prompt_template": (
            "{plant_desc}. A steady wind blowing from the left. The stiff woody trunk bends "
            "slightly and ALL branches swing together in the same direction like a single "
            "rigid body, then sway back. The base stays anchored, the branches and leaves "
            "stay intact. brightly front-lit with bright natural daylight, vivid green foliage, plain light-gray background, single static camera, no zoom, no pan."
        ),
        "guidance_scale": 5.0,
        "seed_offset": 600,
    },
    "wb_gust": {
        "prompt_template": (
            "{plant_desc}. A single gust of wind, then calm. The elastic woody branches move "
            "together with a springy sway and return to rest as one unit; no individual leaf "
            "flutter, no branch breaking. The base stays anchored. brightly front-lit with "
            "bright natural daylight, vivid green foliage, plain light-gray background, single "
            "static camera, no zoom, no pan."
        ),
        "guidance_scale": 5.5,
        "seed_offset": 700,
    },
    "hand_pull": {
        "prompt_template": (
            "{plant_desc}. A human hand enters from the right side of the frame "
            "and firmly grips the main woody stem of the plant near its base. "
            "The hand gently shakes the stem from side to side with a small, "
            "slow oscillation. Because the hand holds the main stem, the whole "
            "plant sways together — every branch and leaf moves in unison with "
            "the stem, like a single rigid body wobbling on top of the pot. "
            "The pot itself remains completely still. The branches and leaves "
            "stay intact, no breaking, no tearing, no leaves falling. "
            "Realistic indoor lighting, single static camera, no zoom, no pan."
        ),
        "guidance_scale": 5.5,
        "seed_offset": 400,
    },
}

HAND_PULL_NEGATIVE = (
    "broken branch, snapped stem, branch breaking, fallen leaves, ripped leaves, "
    "leaves detaching, violent motion, fast pull, aggressive yank, "
    "only one branch moving, isolated leaf flutter, partial motion, "
    "leaves moving independently of stem, pot moving, pot tipping, "
    "camera shake, camera movement, zoom, pan, "
)

NEGATIVE_PROMPT = (
    "turntable, rotation, orbit, camera movement, "
    "static, frozen, low quality, blurry, watermark, text, "
    "silhouette, backlit, underexposed, turning dark, darkening, "
    "black shadow, night, dim lighting, color loss"
)


def dir_name_to_plant_desc(dir_name: str) -> str:
    name = re.sub(r'_s\d+$', '', dir_name)
    name = name.replace('_', ' ')
    return name


def preprocess_image(image_path: Path, max_area: int, mod: int) -> tuple:
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)

    black_mask = (arr < 15).all(axis=-1)
    arr[black_mask] = 255
    img = Image.fromarray(arr)

    aspect_ratio = img.height / img.width
    h = int(round(np.sqrt(max_area * aspect_ratio))) // mod * mod
    w = int(round(np.sqrt(max_area / aspect_ratio))) // mod * mod
    img = img.resize((w, h), Image.LANCZOS)
    return img, h, w


def load_pipeline(model_id, lightx2v_high=None, lightx2v_low=None,
                   sequential_offload=False):
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32,
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id, vae=vae, torch_dtype=torch.bfloat16,
    )

    if lightx2v_high and lightx2v_low:
        print(f"Loading LightX2V high-noise LoRA: {lightx2v_high}")
        pipe.load_lora_weights(lightx2v_high, adapter_name="lightx2v_high")
        pipe.fuse_lora(adapter_names=["lightx2v_high"])
        pipe.unload_lora_weights()

        if pipe.transformer_2 is not None:
            print(f"Loading LightX2V low-noise LoRA: {lightx2v_low}")
            pipe.transformer, pipe.transformer_2 = pipe.transformer_2, pipe.transformer
            pipe.load_lora_weights(lightx2v_low, adapter_name="lightx2v_low")
            pipe.fuse_lora(adapter_names=["lightx2v_low"])
            pipe.unload_lora_weights()
            pipe.transformer, pipe.transformer_2 = pipe.transformer_2, pipe.transformer

    if sequential_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()
    return pipe


def generate_video(pipe, image_path, output_dir, force_type, plant_desc,
                   num_frames, fps, seed, max_area, num_inference_steps):
    cfg = FORCE_CONFIGS[force_type]
    output_dir = Path(output_dir)
    video_path = output_dir / "motion.mp4"
    if video_path.exists():
        print(f"    Skip (exists): {output_dir.name}")
        return video_path

    prompt = cfg["prompt_template"].format(plant_desc=plant_desc)
    negative = NEGATIVE_PROMPT
    if force_type == "hand_pull" or force_type.startswith("wb_"):
        negative = HAND_PULL_NEGATIVE + NEGATIVE_PROMPT

    transformer = pipe.transformer if pipe.transformer is not None else pipe.transformer_2
    patch_size = transformer.config.patch_size
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[1]
    mod = pipe.vae_scale_factor_spatial * patch_size
    image, h, w = preprocess_image(Path(image_path), max_area, mod)

    gc.collect()
    torch.cuda.empty_cache()

    generator = torch.Generator(device="cpu").manual_seed(seed + cfg["seed_offset"])
    frames = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative,
        height=h,
        width=w,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=cfg["guidance_scale"],
        generator=generator,
    ).frames[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(video_path), fps=fps)
    gc.collect()
    torch.cuda.empty_cache()

    return video_path


def get_canonical_frame(plant_dir):
    pose = plant_dir / "pose.png"          # prefer a hand-picked / pot-cropped pose if present
    if pose.exists():
        return pose
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
    parser.add_argument('--model', type=str,
                        default='Wan-AI/Wan2.2-I2V-A14B-Diffusers')
    parser.add_argument('--lightx2v_high', type=str, default=None,
                        help='High-noise LightX2V LoRA (.safetensors)')
    parser.add_argument('--lightx2v_low', type=str, default=None,
                        help='Low-noise LightX2V LoRA (.safetensors)')
    parser.add_argument('--num_frames', type=int, default=81)
    parser.add_argument('--fps', type=int, default=16)
    parser.add_argument('--max_area', type=int, default=480 * 832,
                        help='Max pixel area (480*832=480P, 720*1280=720P)')
    parser.add_argument('--num_inference_steps', type=int, default=None,
                        help='Default: 4 with LightX2V, 30 without')
    parser.add_argument('--force_types', type=str, nargs='+',
                        default=list(FORCE_CONFIGS.keys()),
                        choices=list(FORCE_CONFIGS.keys()))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--smoke_test', action='store_true',
                        help='Quick test: 2 plants x 2 force types, 33 frames')
    parser.add_argument('--image_files', type=str, nargs='+',
                        help='Direct image paths (bypasses --input_dir)')
    parser.add_argument('--plant_descs', type=str, nargs='+',
                        help='Plant descriptions, one per --image_files entry')
    parser.add_argument('--sequential_offload', action='store_true',
                        help='Use sequential CPU offload (slower but less VRAM)')
    args = parser.parse_args()

    use_lightx2v = bool(args.lightx2v_high and args.lightx2v_low)
    if (args.lightx2v_high is None) != (args.lightx2v_low is None):
        parser.error("--lightx2v_high and --lightx2v_low must be used together")
    if args.num_inference_steps is None:
        args.num_inference_steps = 4 if use_lightx2v else 30

    # --- Direct image mode ---
    if args.image_files:
        image_files = [Path(p) for p in args.image_files]
        if args.plant_descs:
            if len(args.plant_descs) != len(image_files):
                parser.error("--plant_descs must have same length as --image_files")
            plant_descs = args.plant_descs
        else:
            plant_descs = [p.stem for p in image_files]

        total = len(image_files) * len(args.force_types)
        print(f"[direct image mode] {len(image_files)} images x "
              f"{len(args.force_types)} forces -> {total} videos")
        print(f"Loading pipeline: {args.model}"
              + (f" + LightX2V ({args.num_inference_steps} steps)" if use_lightx2v else ""))
        pipe = load_pipeline(args.model, args.lightx2v_high, args.lightx2v_low,
                             args.sequential_offload)

        count = 0
        for img_path, plant_desc in zip(image_files, plant_descs):
            if not img_path.exists():
                print(f"  Skip (not found): {img_path}")
                continue
            stem = img_path.stem
            for force_type in args.force_types:
                video_out = Path(args.output_dir) / f"{stem}_{force_type}"
                count += 1
                print(f"[{count}/{total}] {stem} | {force_type}")
                generate_video(
                    pipe, img_path, video_out,
                    force_type=force_type,
                    plant_desc=plant_desc,
                    num_frames=args.num_frames,
                    fps=args.fps,
                    seed=args.seed,
                    max_area=args.max_area,
                    num_inference_steps=args.num_inference_steps,
                )
        print(f"Done. {count} videos in {args.output_dir}")
        return

    # --- 3DGS directory mode ---
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
        print(f"[SMOKE TEST] 2 plants x {args.force_types} x {args.num_frames} frames")

    my_plant_dirs = all_plant_dirs[args.shard_id::args.num_shards]
    total_mine = len(my_plant_dirs) * len(args.force_types)
    shard_label = f"shard {args.shard_id}/{args.num_shards}" if args.num_shards > 1 else "single"

    print(f"[{shard_label}] {len(my_plant_dirs)}/{len(all_plant_dirs)} plants "
          f"-> {total_mine} videos")
    print(f"Force types: {args.force_types}")

    if not my_plant_dirs:
        print("No work for this shard, exiting.")
        return

    print(f"Loading pipeline: {args.model}"
          + (f" + LightX2V ({args.num_inference_steps} steps)" if use_lightx2v else ""))
    pipe = load_pipeline(args.model, args.lightx2v_high, args.lightx2v_low)

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
            print(f"[{count}/{total_mine}] {plant_dir.name} | {force_type}")
            generate_video(
                pipe, canonical, video_out,
                force_type=force_type,
                plant_desc=plant_desc,
                num_frames=args.num_frames,
                fps=args.fps,
                seed=args.seed,
                max_area=args.max_area,
                num_inference_steps=args.num_inference_steps,
            )

    print(f"Done. Shard {args.shard_id} generated {count} videos in {args.output_dir}")


if __name__ == '__main__':
    main()
