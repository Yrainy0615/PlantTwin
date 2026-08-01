"""Precompute per-frame plant masks (SAM) and optical flow (RAFT) for a target video.

Both targets are non-differentiable supervision signals consumed by `VideoLossBundle`.
Run once per video; the optimizer then loads the cached tensors.

This is a thin glue script — pick the backend that is installed in your env.
Backends supported, in order of preference:
  masks: SAM2 (`pip install sam2`) > SAM-1 (`segment-anything`) > simple bg-subtract
  flow:  `torchvision.models.optical_flow` (RAFT) — included in torchvision

Output:
  <out_dir>/masks.pt   [T, 1, H, W] in [0, 1]
  <out_dir>/flow.pt    [T-1, 2, H, W]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.io as tvio
from torchvision.transforms.functional import resize


def load_video_frames(path: Path, max_frames: int | None = None, resize_hw: tuple[int, int] | None = None) -> torch.Tensor:
    frames, _, info = tvio.read_video(str(path), pts_unit='sec')   # [T, H, W, 3], uint8
    frames = frames.permute(0, 3, 1, 2).float() / 255.0
    if max_frames is not None:
        frames = frames[:max_frames]
    if resize_hw is not None:
        frames = resize(frames, list(resize_hw))
    return frames


def compute_masks_sam(frames: torch.Tensor, device: torch.device) -> torch.Tensor | None:
    """Return [T, 1, H, W] in [0, 1], or None if SAM is not installed."""
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        try:
            from sam2.build_sam import build_sam2_video_predictor
            print('[video_targets] SAM2 detected — please integrate your SAM2 inference here.')
            return None
        except ImportError:
            print('[video_targets] SAM not installed — falling back to simple bg-subtraction. '
                  'Install with: pip install segment-anything OR pip install sam2')
            return None
    # We do not auto-download checkpoints; user must wire up their own SAM init.
    print('[video_targets] SAM v1 detected but checkpoint loading not auto-configured. '
          'Edit this script to load your checkpoint.')
    return None


def compute_masks_bg_subtract(frames: torch.Tensor, blur_kernel: int = 9) -> torch.Tensor:
    """Cheap fallback: take per-pixel temporal variance, threshold; not for real use."""
    var = frames.var(dim=0, keepdim=True)
    saliency = var.mean(dim=1, keepdim=True)
    s = saliency / (saliency.max() + 1e-6)
    return s.expand(frames.shape[0], -1, -1, -1)


def compute_flow_raft(frames: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run RAFT (small) from torchvision.models.optical_flow."""
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    weights = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_small(weights=weights, progress=False).to(device).eval()
    flows = []
    with torch.no_grad():
        for t in range(frames.shape[0] - 1):
            f0 = frames[t:t + 1].to(device)
            f1 = frames[t + 1:t + 2].to(device)
            x0, x1 = transforms(f0, f1)
            out = model(x0, x1)
            flow = out[-1]  # final iteration
            flows.append(flow.squeeze(0).cpu())
    return torch.stack(flows, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--H', type=int, default=256)
    parser.add_argument('--W', type=int, default=256)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no-mask', action='store_true', help='Skip mask precompute.')
    parser.add_argument('--no-flow', action='store_true', help='Skip flow precompute.')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    frames = load_video_frames(Path(args.video), args.max_frames, (args.H, args.W))
    print(f'Loaded {frames.shape[0]} frames at {frames.shape[2]}x{frames.shape[3]}')

    if not args.no_mask:
        masks = compute_masks_sam(frames, device)
        if masks is None:
            print('[video_targets] using temporal-variance pseudo masks (placeholder).')
            masks = compute_masks_bg_subtract(frames)
        torch.save(masks.cpu(), out_dir / 'masks.pt')
        print(f'  saved masks {tuple(masks.shape)} -> {out_dir / "masks.pt"}')

    if not args.no_flow:
        flow = compute_flow_raft(frames, device)
        torch.save(flow.cpu(), out_dir / 'flow.pt')
        print(f'  saved flow {tuple(flow.shape)} -> {out_dir / "flow.pt"}')


if __name__ == '__main__':
    main()
