"""
Generate static 3D Gaussians of plants using TRELLIS text-to-3D pipeline.

Usage:
    python data/generation/gen_3dgs.py --prompt "a potted fern" --output_dir data/plants_3dgs
    python data/generation/gen_3dgs.py --prompt_file configs/plant_prompts.txt --output_dir data/plants_3dgs
    python data/generation/gen_3dgs.py --prompt_file configs/plant_prompts.txt --smoke_test
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, help='Single text prompt')
    parser.add_argument('--prompt_file', type=str, help='File with one prompt per line')
    parser.add_argument('--output_dir', type=str, default='data/plants_3dgs')
    parser.add_argument('--model', type=str, default='microsoft/TRELLIS-text-xlarge')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_seeds', type=int, default=1, help='Number of seeds per prompt for diversity')
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
        max_gs = 10
        max_prompts = max_gs // max(args.num_seeds, 1)
        prompts = prompts[:max_prompts]
        print(f"[SMOKE TEST] Limited to {len(prompts)} prompts x {args.num_seeds} seeds = {len(prompts) * args.num_seeds} GS")

    print(f"Loading TRELLIS model: {args.model}")
    pipeline = TrellisTextTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    total = len(prompts) * args.num_seeds
    print(f"Generating {total} plants from {len(prompts)} prompts x {args.num_seeds} seeds")

    for i, prompt in enumerate(prompts):
        for s in range(args.num_seeds):
            seed = args.seed + s
            print(f"[{i * args.num_seeds + s + 1}/{total}] '{prompt}' (seed={seed})")
            generate_single(pipeline, prompt, args.output_dir, seed=seed)

    print(f"Done. All outputs in {args.output_dir}")


if __name__ == '__main__':
    main()
