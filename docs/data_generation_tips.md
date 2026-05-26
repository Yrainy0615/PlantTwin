# Data Generation Tips

Lessons learned from building the TRELLIS → Wan2.1 video generation pipeline.

---

## Video Model: SVD → Wan2.1

**Original:** `stabilityai/stable-video-diffusion-img2vid-xt`
**Current:** `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`

SVD is image-only conditioned and controls motion intensity via a single scalar (`motion_bucket_id`). Wan2.1 accepts image + text, which lets you describe the *type* of motion semantically. For plant deformation, this difference is significant because "gentle leaf trembling" and "branches whipping" share similar pixel-level motion magnitude but are semantically very different.

---

## Image Preprocessing

### 1. Replace the Black Background Before Feeding to the Video Model

3DGS renders produce black backgrounds. Feeding a black-bg image to Wan2.1 confuses the motion prior — the model treats the dark surround as part of the scene.

```python
arr = np.array(img)
black_mask = (arr < 15).all(axis=-1)
arr[black_mask] = 255
```

### 2. Letterbox, Don't Crop

Use `Image.thumbnail()` + paste onto a canvas instead of resizing with crop. This keeps the plant at the same scale and position as the 3DGS camera render, which is important: the rendered frame and the generated video frame need to share the same camera geometry for the temporal-diff training loss to be meaningful.

```python
img.thumbnail((width, height), Image.LANCZOS)
canvas = Image.new("RGB", (width, height), (200, 200, 200))
canvas.paste(img, ((width - img.width) // 2, (height - img.height) // 2))
```

### 3. Use Light Gray Canvas, Not White

A pure white background triggers the model's "product photography / turntable" prior, causing it to generate camera rotation rather than plant motion. Light gray (200, 200, 200) is neutral enough to avoid this.

---

## Prompt Engineering for Plant Motion

### Use a Negative Prompt

Without a negative prompt the model frequently generates subtle camera pan/orbit instead of plant deformation. Suppress this explicitly:

```python
NEGATIVE_PROMPT = (
    "turntable, 360 degree rotation, orbit, product shot, product photography, "
    "camera rotation, camera pan, camera zoom, camera movement, "
    "static, no motion, frozen, low quality, blur, watermark, text"
)
```

### Give Each Force Type a Dedicated Prompt Template

A single `motion_bucket_id` cannot distinguish between wind sway, elastic rebound, or gravity settling. Use per-force-type `prompt_template` strings with `{plant_desc}` filled at runtime from the directory name.

| Force Type | Key phrases in prompt |
|---|---|
| `wind_light` | *"swaying left and right in a gentle breeze, leaves trembling, branches oscillating softly"* |
| `wind_intense` | *"violently shaking in strong wind, branches whipping, leaves fluttering chaotically"* |
| `external_light` | *"tapped lightly, one branch bending and slowly bouncing back, gentle elastic rebound"* |
| `external_intense` | *"struck hard, branches bending sharply sideways then springing back, strong elastic rebound"* |
| `drag_light` | *"swaying very slowly and settling, minimal oscillation gradually dying down to rest"* |

### Per-Force Guidance Scale

Stronger perturbations warrant higher `guidance_scale` to prevent the model from reverting to static:

| Force Type | `guidance_scale` |
|---|---|
| `drag_light` | 6.0 |
| `wind_light` | 6.5 |
| `external_light` | 7.0 |
| `wind_intense` | 7.5 |
| `external_intense` | 8.0 |

---

## Video Parameters

| Parameter | SVD (old) | Wan2.1 (current) | Notes |
|---|---|---|---|
| `num_frames` | 25 | 81 | Longer = more visible motion arc |
| `fps` | 7 | 16 | Smoother playback; matches Wan2.1's training distribution |
| Resolution | 1024×576 | 832×480 | Wan2.1's native 480P ratio |

---

## Multi-GPU Parallel Generation

Both `gen_3dgs.py` and `gen_video.py` support `--shard_id` / `--num_shards` for trivially parallel generation across GPUs. Each worker takes every `num_shards`-th item from the full sorted list, so work is deterministically split without coordination.

```bash
# Example: 8 GPUs generating 3DGS
for i in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$i python data/generation/gen_3dgs.py \
        --prompt_file configs/plant_prompts.txt \
        --total_plants 200 \
        --shard_id $i --num_shards 8 &
done
```

Both scripts also skip already-completed outputs (`gaussian.ply` / `motion.mp4` existence check), making restarts safe.
