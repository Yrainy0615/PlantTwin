"""
Test PhysX-Omni VLM on real plant images.
Step 1: Get structured description (parts, materials, hierarchy).
Step 2: Get per-part 3D voxel RLE.
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageFile
from pathlib import Path

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../third_party/PhysX-Omni'))
from rembg import remove

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def remove_background(img: Image.Image) -> Image.Image:
    result = remove(img)
    white_bg = Image.new("RGB", result.size, (255, 255, 255))
    white_bg.paste(result, mask=result.split()[3] if result.mode == "RGBA" else None)
    return white_bg


def generate(model, processor, messages, max_length=32768):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs, do_sample=False, temperature=0, max_length=max_length,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, nargs="+", required=True)
    parser.add_argument("--model_path", type=str,
                        default="third_party/PhysX-Omni/pretrain")
    parser.add_argument("--output_dir", type=str, default="data/physx_vlm_output")
    parser.add_argument("--skip_rembg", action="store_true",
                        help="Skip background removal")
    parser.add_argument("--skip_voxel", action="store_true",
                        help="Only get basic_info, skip voxel RLE generation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = Path("third_party/PhysX-Omni/dataset/example_64_finetune_rle.txt")
    prompt_text = prompt_path.read_text(encoding="utf-8")

    print(f"Loading model from {args.model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    min_pixels, max_pixels = 65536, 262144
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        min_pixels=min_pixels, max_pixels=max_pixels,
    )
    processor.image_processor.min_pixels = min_pixels
    processor.image_processor.max_pixels = max_pixels
    processor.image_processor.size["shortest_edge"] = min_pixels
    processor.image_processor.size["longest_edge"] = max_pixels

    for img_path in args.images:
        name = Path(img_path).stem
        save_dir = output_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"{'='*60}")

        img = Image.open(img_path).convert("RGB")
        if not args.skip_rembg:
            print("  Removing background...")
            img = remove_background(img)
        img_resized = img.resize((512, 512), Image.LANCZOS)
        img_resized.save(str(save_dir / "input.png"))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_resized},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        print("  Step 1: Getting structured description...")
        basic_output = generate(model, processor, messages)
        (save_dir / "basic_info.txt").write_text(basic_output, encoding="utf-8")
        print(f"\n--- basic_info for {name} ---")
        print(basic_output)
        print("---\n")

        if args.skip_voxel:
            print("  Skipping voxel generation (--skip_voxel)")
            continue

        part_count = 0
        while f"l_{part_count}" in basic_output:
            part_count += 1

        print(f"  Found {part_count} parts, generating voxels...")

        from importlib.machinery import SourceFileLoader
        vlm_mod = SourceFileLoader("vlm_demo",
            "third_party/PhysX-Omni/1vlm_demo.py").load_module()

        all_coords = []
        for part_id in range(part_count):
            question = (
                f"Based on the structured description of l_{part_id}, "
                f"generate its 3D voxel (grid=64) in the 3D RLE (linear scan) format. "
                f"Output one run per line as: start_index length"
            )
            messages_voxel = messages.copy()
            messages_voxel.append({"role": "assistant",
                                   "content": [{"type": "text", "text": basic_output}]})
            messages_voxel.append({"role": "user",
                                   "content": [{"type": "text", "text": question}]})

            print(f"  Generating voxel for part l_{part_id}...")
            voxel_output = generate(model, processor, messages_voxel)
            (save_dir / f"coord_{part_id}.txt").write_text(voxel_output, encoding="utf-8")

            runs_by_z = vlm_mod.string_to_runs_by_z_lossless_robust(voxel_output, D=64)
            voxels = vlm_mod.decode_voxel_2drle_by_z(runs_by_z, shape=(64, 64, 64))
            np.save(str(save_dir / f"ind_{part_id}.npy"), voxels)
            print(f"    Part l_{part_id}: {len(voxels)} voxels")

            if len(voxels) > 0:
                import trimesh
                pc = trimesh.points.PointCloud(voxels)
                pc.export(str(save_dir / f"ind_{part_id}.ply"))

            all_coords.append(voxels)

        if all_coords:
            all_voxels = np.concatenate(all_coords)
            np.save(str(save_dir / "allind.npy"), all_voxels)
            print(f"  Total: {len(all_voxels)} voxels across {part_count} parts")

    print("\nDone!")


if __name__ == "__main__":
    main()
