"""Locate a human hand in a video with Grounding-DINO + SAM, then emit
CoTracker query points sampled inside the hand mask.

Flow (per video):
  1. Run Grounding-DINO with the text prompt "a human hand" on every frame.
  2. Pick the seed frame as the first frame whose top detection passes the
     score threshold (or the user-supplied --seed-frame).
  3. Run SAM with the chosen box -> binary hand mask at the seed frame.
  4. Farthest-point-sample N pixels inside the mask -> query bundle.

Output bundle (.pt):
  {
    'queries'   : [N, 3] (t, x, y) in source-video pixels — t = seed_frame
    'seed_frame': int
    'hand_mask' : [H_src, W_src] uint8
    'hand_box'  : [4] xyxy in source pixels
    'src_HW'    : (H_src, W_src)
    'score'     : float, Grounding-DINO confidence on the seed detection
  }
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.io as tvio
from PIL import Image


def _load_video_pil(path: Path, max_frames: int | None) -> list[Image.Image]:
    frames, _, _ = tvio.read_video(str(path), pts_unit='sec')   # [T, H, W, 3] uint8
    if max_frames is not None:
        frames = frames[:max_frames]
    return [Image.fromarray(f.numpy()) for f in frames]


def _farthest_point_sample_mask(mask: torch.Tensor, n: int, seed: int = 0) -> torch.Tensor:
    """[N, 2] (x, y) FPS samples inside a [H, W] bool mask."""
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if ys.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.float32)
    pts = torch.stack([xs.float(), ys.float()], dim=-1)
    if pts.shape[0] <= n:
        return pts
    g = torch.Generator().manual_seed(seed)
    idx0 = int(torch.randint(pts.shape[0], (1,), generator=g).item())
    sel = [idx0]
    dists = (pts - pts[idx0]).norm(dim=-1)
    for _ in range(1, n):
        nxt = int(dists.argmax().item())
        sel.append(nxt)
        dists = torch.minimum(dists, (pts - pts[nxt]).norm(dim=-1))
    return pts[sel]


def detect(args):
    device = torch.device(args.device)
    images = _load_video_pil(Path(args.video), args.max_frames)
    if not images:
        raise SystemExit('Empty video')
    W, H = images[0].size
    print(f'Loaded {len(images)} frames at {W}x{H}')

    from transformers import (
        AutoProcessor,
        GroundingDinoForObjectDetection,
        SamModel,
        SamProcessor,
    )

    print(f'Loading Grounding-DINO ({args.gdino_model})...')
    gdino_proc = AutoProcessor.from_pretrained(args.gdino_model)
    gdino = GroundingDinoForObjectDetection.from_pretrained(args.gdino_model).to(device).eval()

    text = args.text
    if not text.endswith('.'):
        text = text + '.'

    # Sweep frames; pick the highest-score detection that also satisfies the
    # spatial constraint (default: box right-edge is in the right half of the
    # frame, matching our hand-pull prompt where the hand enters from the right).
    seed_frame_arg = args.seed_frame
    if seed_frame_arg is not None:
        candidate_iter = [(seed_frame_arg, images[seed_frame_arg])]
    else:
        candidate_iter = list(enumerate(images))[args.skip_first:]

    per_frame: list[tuple[int, float, torch.Tensor]] = []
    with torch.no_grad():
        for t, img in candidate_iter:
            inputs = gdino_proc(images=img, text=text, return_tensors='pt').to(device)
            outputs = gdino(**inputs)
            target_sizes = torch.tensor([img.size[::-1]]).to(device)
            try:
                results = gdino_proc.post_process_grounded_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                )
            except TypeError:
                results = gdino_proc.post_process_grounded_object_detection(
                    outputs,
                    input_ids=inputs['input_ids'],
                    target_sizes=target_sizes,
                    threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                )
            scores = results[0]['scores']
            boxes = results[0]['boxes']
            if scores.numel() == 0:
                continue
            # Spatial constraint
            for s, b in zip(scores.tolist(), boxes.tolist()):
                x1, y1, x2, y2 = b
                box_right_norm = x2 / W
                if box_right_norm < args.min_right_norm:
                    continue
                per_frame.append((t, s, torch.tensor(b)))

    if not per_frame:
        raise SystemExit(
            'No hand detection passed score+spatial filter. '
            'Lower --box-threshold or --min-right-norm, or set --seed-frame.'
        )

    # Pick the best (highest score) overall
    per_frame.sort(key=lambda x: -x[1])
    seed_frame, seed_score, seed_box_xyxy = per_frame[0]
    print('[gdino] top candidates that passed filter:')
    for (t, s, b) in per_frame[:5]:
        print(f'    frame {t:3d}  score={s:.3f}  box={[round(v,1) for v in b.tolist()]}')
    print(f'Seed frame: {seed_frame}  (score {seed_score:.3f})  box: {seed_box_xyxy.tolist()}')

    print(f'Loading SAM ({args.sam_model})...')
    sam_proc = SamProcessor.from_pretrained(args.sam_model)
    sam = SamModel.from_pretrained(args.sam_model).to(device).eval()

    seed_img = images[seed_frame]
    with torch.no_grad():
        box = seed_box_xyxy.tolist()
        sam_inputs = sam_proc(
            seed_img,
            input_boxes=[[box]],
            return_tensors='pt',
        ).to(device)
        sam_out = sam(**sam_inputs)
        # masks: [batch, 1, K, H, W] — pick the highest-iou mask
        masks = sam_proc.image_processor.post_process_masks(
            sam_out.pred_masks.cpu(),
            sam_inputs['original_sizes'].cpu(),
            sam_inputs['reshaped_input_sizes'].cpu(),
        )[0]                                                  # [1, K, H, W] uint8
        iou_scores = sam_out.iou_scores[0, 0].cpu()           # [K]
        best = int(iou_scores.argmax().item())
        hand_mask = masks[0, best].bool()                     # [H, W]

    coverage = hand_mask.float().mean().item()
    print(f'[sam] hand mask coverage: {coverage * 100:.2f}%, iou_score={iou_scores[best].item():.3f}')

    queries_xy = _farthest_point_sample_mask(hand_mask, args.n_queries, seed=args.seed)
    if queries_xy.shape[0] == 0:
        raise SystemExit('SAM returned empty hand mask; bad detection.')
    print(f'Sampled {queries_xy.shape[0]} hand query points')

    queries = torch.cat([
        torch.full((queries_xy.shape[0], 1), float(seed_frame)),
        queries_xy,
    ], dim=-1)                                                 # [N, 3] (t, x, y)

    bundle = {
        'queries': queries,
        'seed_frame': int(seed_frame),
        'hand_mask': hand_mask.to(torch.uint8),
        'hand_box': seed_box_xyxy,
        'src_HW': (H, W),
        'score': seed_score,
        'text': text,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)
    print(f'  saved -> {out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--text', default='a human hand')
    parser.add_argument('--seed-frame', type=int, default=None,
                        help='Force which frame to seed from. Default: first frame with a detection above threshold.')
    parser.add_argument('--n-queries', type=int, default=40)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--gdino-model', default='IDEA-Research/grounding-dino-tiny')
    parser.add_argument('--sam-model', default='facebook/sam-vit-base')
    parser.add_argument('--box-threshold', type=float, default=0.25)
    parser.add_argument('--text-threshold', type=float, default=0.20)
    parser.add_argument('--skip-first', type=int, default=5,
                        help='Skip the first N frames when searching (hand usually does not appear immediately).')
    parser.add_argument('--min-right-norm', type=float, default=0.55,
                        help='Require the detection box right-edge to be at >= this fraction of image width '
                             '(hand enters from the right in our prompt). Set 0 to disable.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    detect(args)


if __name__ == '__main__':
    main()
