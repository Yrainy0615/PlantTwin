"""Track 2D keypoints across a video with CoTracker (loaded via torch.hub).

Two seeding modes:
  --grid N        : track an N x N grid laid down at frame 0 (or --seed-frame)
  --queries u v t : track explicit points (repeatable). t is the seed frame.

Output bundle (single .pt):
    {
      'tracks'     : [T, N, 2]   pixel xy
      'visibility' : [T, N]      bool
      'video_shape': (T, H, W)
      'seed_frame' : int
      'H', 'W'     : int
    }

We do NOT decide which tracks are "hand vs. stem" here — that filtering is the
downstream step (see `scripts/precompute_contact.py --tracks <out>`). Keeping
the tracker output topology-free means we can re-use the same tracking pass
for several purposes (hand pin, stem deformation supervision, etc.).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.io as tvio
from torchvision.transforms.functional import resize


def load_video(path: Path, max_frames: int | None, H: int, W: int) -> torch.Tensor:
    frames, _, _ = tvio.read_video(str(path), pts_unit='sec')
    frames = frames.permute(0, 3, 1, 2).float()        # [T, 3, H0, W0], in [0, 255]
    if max_frames is not None:
        frames = frames[:max_frames]
    return resize(frames, [H, W])


def _load_cotracker(device: torch.device, model_name: str):
    """Load CoTracker3 via torch.hub. Falls back to v2 if v3 unavailable."""
    for candidate in (model_name, 'cotracker3_offline', 'cotracker2'):
        try:
            return torch.hub.load('facebookresearch/co-tracker', candidate).to(device).eval()
        except Exception as e:
            print(f'[cotracker] {candidate} unavailable: {e}')
    raise RuntimeError('CoTracker could not be loaded from torch.hub.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--H', type=int, default=256)
    parser.add_argument('--W', type=int, default=256)
    parser.add_argument('--grid', type=int, default=20, help='grid_size for dense tracking')
    parser.add_argument('--queries', type=float, nargs='*',
                        help='Explicit query points as triples (u v t), repeatable.')
    parser.add_argument('--query-bundle', default=None,
                        help='Path to a hand-query bundle (e.g. from detect_hand_grounded_sam.py).')
    parser.add_argument('--also-grid', action='store_true',
                        help='When using --query-bundle, also include a dense grid of stem candidates.')
    parser.add_argument('--seed-frame', type=int, default=0,
                        help='Frame index where the grid (or queries) is anchored.')
    parser.add_argument('--model', default='cotracker3_offline')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-viz', default=None, help='Optional path to .mp4 visualization.')
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    video = load_video(Path(args.video), args.max_frames, args.H, args.W)   # [T, 3, H, W]
    T, _, H, W = video.shape
    print(f'Loaded video {T}x{H}x{W}')

    model = _load_cotracker(device, args.model)
    print(f'Loaded CoTracker on {device}')

    video_in = video.unsqueeze(0).to(device)   # [1, T, 3, H, W]

    # Build queries from any of: --queries, --query-bundle, --also-grid
    query_chunks: list[tuple[str, torch.Tensor]] = []   # (origin_tag, [N, 3] t,x,y in source res)

    if args.queries:
        if len(args.queries) % 3 != 0:
            raise SystemExit('--queries must be triples (u v t)')
        triples = [args.queries[i:i + 3] for i in range(0, len(args.queries), 3)]
        q = torch.tensor(triples, dtype=torch.float32)
        q_txy = torch.stack([q[:, 2], q[:, 0], q[:, 1]], dim=-1)  # (t, x, y)
        query_chunks.append(('cli', q_txy))

    if args.query_bundle is not None:
        b = torch.load(args.query_bundle, map_location='cpu', weights_only=False)
        bundle_queries = b['queries']                              # [N, 3] (t, x, y) in src pixels
        src_H, src_W = b['src_HW']
        # Scale into the loaded video's H, W
        sx = W / src_W
        sy = H / src_H
        scaled = bundle_queries.clone()
        scaled[:, 1] = scaled[:, 1] * sx
        scaled[:, 2] = scaled[:, 2] * sy
        query_chunks.append(('hand', scaled))
        print(f'Loaded {scaled.shape[0]} hand queries from {args.query_bundle} '
              f'(seed frame {int(scaled[0, 0].item())})')

    queries_tensor = None
    origin_tags: list[str] | None = None
    if query_chunks:
        if args.also_grid:
            # Add a grid at frame 0 alongside the bundle queries
            gs = args.grid
            xs = torch.linspace(0, W - 1, gs)
            ys = torch.linspace(0, H - 1, gs)
            gx, gy = torch.meshgrid(xs, ys, indexing='xy')
            grid = torch.stack([
                torch.zeros_like(gx).flatten(),
                gx.flatten(),
                gy.flatten(),
            ], dim=-1)
            query_chunks.append(('grid', grid))
            print(f'Added {grid.shape[0]} grid queries at frame 0')
        all_q = torch.cat([c[1] for c in query_chunks], dim=0)
        origin_tags = []
        for tag, c in query_chunks:
            origin_tags.extend([tag] * c.shape[0])
        queries_tensor = all_q[None].to(device)

    with torch.no_grad():
        if queries_tensor is not None:
            pred_tracks, pred_vis = model(video_in, queries=queries_tensor)
        else:
            pred_tracks, pred_vis = model(video_in, grid_size=args.grid)

    tracks = pred_tracks[0].cpu()        # [T, N, 2]
    visibility = pred_vis[0].cpu().bool()
    print(f'Tracks: T={tracks.shape[0]}, N={tracks.shape[1]}; visible fraction '
          f'{visibility.float().mean().item():.3f}')

    bundle = {
        'tracks': tracks,
        'visibility': visibility,
        'video_shape': (T, H, W),
        'seed_frame': args.seed_frame,
        'H': H, 'W': W,
    }
    if origin_tags is not None:
        bundle['query_origin'] = origin_tags
    torch.save(bundle, out)
    print(f'  saved -> {out}')

    if args.save_viz:
        try:
            from cotracker.utils.visualizer import Visualizer  # type: ignore
            vis = Visualizer(save_dir=str(Path(args.save_viz).parent), pad_value=0,
                              fps=int(16), linewidth=2, tracks_leave_trace=-1)
            vis.visualize(video_in.cpu(), pred_tracks.cpu(), pred_vis.cpu(),
                          filename=Path(args.save_viz).stem)
            print(f'  viz -> {args.save_viz}')
        except Exception as e:
            # Fallback: simple per-frame scatter overlay (matplotlib)
            print(f'[viz] CoTracker visualizer unavailable ({e}); writing matplotlib overlay.')
            _write_matplotlib_overlay(video, tracks, visibility, Path(args.save_viz))


def _write_matplotlib_overlay(video, tracks, visibility, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    T, _, H, W = video.shape
    fig, ax = plt.subplots(1, 1, figsize=(W / 100.0, H / 100.0), dpi=100)
    writer = FFMpegWriter(fps=16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(out_path), dpi=100):
        for t in range(T):
            ax.clear()
            ax.imshow(video[t].permute(1, 2, 0).byte().numpy())
            mask = visibility[t]
            ax.scatter(tracks[t, mask, 0], tracks[t, mask, 1], c='red', s=4)
            ax.set_axis_off()
            writer.grab_frame()
    plt.close(fig)
    print(f'  matplotlib viz -> {out_path}')


if __name__ == '__main__':
    main()
