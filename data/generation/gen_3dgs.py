"""
Generate static 3D Gaussians of plants using TRELLIS text-to-3D pipeline.

Usage:
    python data/generation/gen_3dgs.py --prompt "a potted fern" --output_dir data/plants_3dgs
    python data/generation/gen_3dgs.py --prompt_file configs/plant_prompts.txt --output_dir data/plants_3dgs
    python data/generation/gen_3dgs.py --prompt_file configs/plant_prompts.txt --smoke_test

Multi-GPU (shard mode):
    CUDA_VISIBLE_DEVICES=0 python data/generation/gen_3dgs.py \\
        --prompt_file configs/plant_prompts.txt --total_plants 200 \\
        --shard_id 0 --num_shards 8
"""
import os
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../third_party/TRELLIS'))
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'

import imageio
import numpy as np
import torch
from PIL import Image
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils


def generate_single(pipeline, prompt, output_dir, seed=42):
    """Generate a single plant 3DGS from text prompt."""
    safe_name = prompt.replace(' ', '_').replace('/', '_')[:60]
    sample_dir = Path(output_dir) / f"{safe_name}_s{seed}"

    # Skip if already generated
    if (sample_dir / "gaussian.ply").exists() and (sample_dir / "meta.json").exists():
        print(f"  Skip (exists): {sample_dir.name}")
        return sample_dir

    sample_dir.mkdir(parents=True, exist_ok=True)

    outputs = pipeline.run(
        prompt,
        seed=seed,
        formats=['gaussian'],
        sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
        slat_sampler_params={"steps": 12, "cfg_strength": 7.5},
    )

    gaussian = outputs['gaussian'][0]
    gaussian.save_ply(str(sample_dir / "gaussian.ply"))

    video = render_utils.render_video(gaussian)['color']
    imageio.mimsave(str(sample_dir / "preview.mp4"), video, fps=30)

    canonical_frame = Image.fromarray(video[0])
    canonical_frame.save(str(sample_dir / "canonical_frame.png"))

    meta = {"prompt": prompt, "seed": seed, "ply_path": str(sample_dir / "gaussian.ply")}
    with open(sample_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved to {sample_dir}")
    return sample_dir


def build_work_items(prompts, total_plants, base_seed):
    """
    Build a flat list of (prompt, seed) pairs to reach total_plants.
    Cycles over prompts with increasing seeds until target is reached.
    """
    items = []
    seed_offset = 0
    while len(items) < total_plants:
        seed = base_seed + seed_offset
        for prompt in prompts:
            if len(items) >= total_plants:
                break
            items.append((prompt, seed))
        seed_offset += 1
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, help='Single text prompt')
    parser.add_argument('--prompt_file', type=str, help='File with one prompt per line')
    parser.add_argument('--output_dir', type=str, default='data/plants_3dgs')
    parser.add_argument('--model', type=str, default='microsoft/TRELLIS-text-xlarge')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_seeds', type=int, default=1,
                        help='Seeds per prompt (ignored when --total_plants is set)')
    parser.add_argument('--total_plants', type=int, default=0,
                        help='Target total plants; cycles prompts with increasing seeds to reach count')
    parser.add_argument('--shard_id', type=int, default=0,
                        help='Worker index for multi-GPU parallel generation (0-based)')
    parser.add_argument('--num_shards', type=int, default=1,
                        help='Total number of parallel workers')
    parser.add_argument('--smoke_test', action='store_true', help='Quick test: limit to 10 GS')
    args = parser.parse_args()

    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompts.extend([line.strip() for line in f if line.strip() and not line.startswith('#')])

    if not prompts:
        parser.error("Provide --prompt or --prompt_file")

    if args.smoke_test:
        prompts = prompts[:5]
        args.total_plants = 10
        args.num_shards = 1
        args.shard_id = 0
        print(f"[SMOKE TEST] {args.total_plants} plants total")

    # Build full work list then take this shard's slice
    if args.total_plants > 0:
        all_items = build_work_items(prompts, args.total_plants, args.seed)
    else:
        all_items = [(p, args.seed + s) for p in prompts for s in range(args.num_seeds)]

    my_items = all_items[args.shard_id::args.num_shards]

    shard_label = f"shard {args.shard_id}/{args.num_shards}" if args.num_shards > 1 else "single"
    print(f"[{shard_label}] {len(my_items)}/{len(all_items)} plants assigned to this worker")

    if not my_items:
        print("No work for this shard, exiting.")
        return

    print(f"Loading TRELLIS model: {args.model}")
    pipeline = TrellisTextTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    for idx, (prompt, seed) in enumerate(my_items):
        global_idx = args.shard_id + idx * args.num_shards + 1
        print(f"[{idx + 1}/{len(my_items)} | global ~{global_idx}] '{prompt[:60]}' (seed={seed})")
        generate_single(pipeline, prompt, args.output_dir, seed=seed)

    print(f"Done. Shard {args.shard_id} finished. Outputs in {args.output_dir}")


if __name__ == '__main__':
    main()
