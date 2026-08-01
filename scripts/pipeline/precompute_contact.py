"""Precompute per-frame contact anchor for a target video.

For each frame:
1. Detect a hand region in image space (mediapipe / ultralytics / manual click).
2. Choose a 2D contact pixel pin (default: hand centroid).
3. Project all branch graph nodes (and optionally leaf disk centers) into the
   image plane using the renderer's camera, then pick the nearest projected
   node as the contact anchor.

Output:
  <out>/contact.pt    dict with:
    'anchor_node_id': [T] long (-1 = no contact this frame)
    'pixel_pin'    : [T, 2] float (image-space u, v); NaN where no contact
    'camera'       : dict (the camera used for projection)

If no hand detector is installed, the script writes a "no-contact" bundle so
the downstream optimizer can still run (contact-force ~ 0 across frames).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.io as tvio
from torchvision.transforms.functional import resize


def load_video(path: Path, max_frames: int | None, H: int, W: int) -> torch.Tensor:
    frames, _, _ = tvio.read_video(str(path), pts_unit='sec')
    frames = frames.permute(0, 3, 1, 2).float() / 255.0
    if max_frames is not None:
        frames = frames[:max_frames]
    return resize(frames, [H, W])


def detect_hand_pixels(frames: torch.Tensor) -> torch.Tensor | None:
    """Return [T, 2] (u, v) hand centroid or None if no detector available."""
    try:
        import mediapipe as mp  # noqa: F401
    except ImportError:
        try:
            from ultralytics import YOLO  # noqa: F401
        except ImportError:
            return None
    # Hooks for actual detection live here; left as TODO until a detector is wired.
    print('[contact] Detector library detected but inference not auto-configured. '
          'Edit this function to call your detector.')
    return None


def _motion_consistency_filter(
    tracks: torch.Tensor,        # [T, N, 2]
    visibility: torch.Tensor,    # [T, N]
    candidate: torch.Tensor,     # [N] bool — tracks to consider
    neighbor_dist: float,
    min_neighbors: int = 3,
) -> torch.Tensor:
    """PhysTwin-style track sanity filter: reject points whose motion is
    incoherent with their spatial neighbors at frame 0.

    For each candidate track at frame 0 we build a KDTree over the other
    candidates' positions; we require >= `min_neighbors` within `neighbor_dist`
    AND that the track's mean motion is within `neighbor_dist / 2` of the
    neighborhood mean motion. Pure-noise / occlusion-jump tracks fall out.
    """
    from scipy.spatial import cKDTree

    N = tracks.shape[1]
    keep = candidate.clone()
    cand_idx = candidate.nonzero().flatten().tolist()
    # Skip consistency filter when the candidate set is too sparse for the
    # KDTree to have any neighbors at all (PhysTwin assumes dense control
    # points; a 20x20 grid filtered to 5% leaves ~20 points, which is sparse).
    if len(cand_idx) < 20:
        print(f'[contact] motion-consistency skipped: only {len(cand_idx)} candidates')
        return keep
    if len(cand_idx) < min_neighbors + 1:
        return keep

    motion_vec = (tracks[-1] - tracks[0])                              # [N, 2]
    pos0 = tracks[0]                                                   # [N, 2]
    pts = pos0[cand_idx].numpy()
    tree = cKDTree(pts)
    for local_i, gi in enumerate(cand_idx):
        nbrs = tree.query_ball_point(pts[local_i], r=neighbor_dist)
        nbrs = [j for j in nbrs if j != local_i]
        if len(nbrs) < min_neighbors:
            keep[gi] = False
            continue
        nbr_global = torch.tensor([cand_idx[j] for j in nbrs], dtype=torch.long)
        local_mv = motion_vec[gi]
        nbr_mv_mean = motion_vec[nbr_global].mean(dim=0)
        if (local_mv - nbr_mv_mean).norm().item() > neighbor_dist / 2.0:
            keep[gi] = False
    return keep


def _split_hand_vs_stem(
    tracks: torch.Tensor,
    visibility: torch.Tensor,
    moving: torch.Tensor,
    src_W: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Without explicit masks, split a "moving" set into hand vs stem using a
    simple heuristic that matches our prompt ("hand enters from the right"):

    - hand cluster: rightmost half of the frame, large per-track motion
    - stem cluster: leftward of the hand cluster, present from frame 0

    Returns (hand_mask [N], stem_mask [N]).
    """
    N = tracks.shape[1]
    motion_per_track = (tracks[1:] - tracks[:-1]).norm(dim=-1).sum(dim=0)
    # Take the moving-cluster mean x to define the split
    mv = moving.nonzero().flatten()
    if mv.numel() == 0:
        empty = torch.zeros(N, dtype=torch.bool)
        return empty, empty
    cluster_pos = tracks[0, mv, 0]                                     # x at frame 0
    motion_in_cluster = motion_per_track[mv]
    # Median-by-motion gives a reasonable split between the high-motion hand
    # (typically near the right) and the lower-motion stem points that follow it.
    cutoff_motion = torch.quantile(motion_in_cluster, 0.6)
    is_hand_like = motion_in_cluster >= cutoff_motion
    # Within high-motion, prefer the right-half tracks for the hand label
    right_half = cluster_pos > (src_W * 0.55)
    hand_within = is_hand_like & right_half
    # If right-half hand candidates are too sparse, fall back to top-motion only.
    if hand_within.sum() < max(1, mv.numel() // 8):
        hand_within = is_hand_like

    hand_mask = torch.zeros(N, dtype=torch.bool)
    stem_mask = torch.zeros(N, dtype=torch.bool)
    hand_mask[mv[hand_within]] = True
    stem_mask[mv[~hand_within]] = True
    return hand_mask, stem_mask


def _farthest_point_sample(points2d: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of k FPS samples from a [M, 2] point set."""
    M = points2d.shape[0]
    if M <= k:
        return torch.arange(M)
    selected = [int(torch.randint(M, (1,)).item())]
    dists = (points2d - points2d[selected[0]]).norm(dim=-1)
    for _ in range(1, k):
        nxt = int(dists.argmax().item())
        selected.append(nxt)
        new_d = (points2d - points2d[nxt]).norm(dim=-1)
        dists = torch.minimum(dists, new_d)
    return torch.tensor(selected, dtype=torch.long)


def tracks_to_contact_bundle(
    tracks_path: Path,
    target_H: int,
    target_W: int,
    motion_percentile: float = 80.0,
    min_visible_frac: float = 0.7,
    neighbor_dist_frac: float = 0.06,
    max_hand: int = 30,
    max_stem: int = 30,
    plant_mask_path: Path | None = None,
) -> dict:
    """Reduce a CoTracker dense-grid output into a contact bundle.

    Mirrors PhysTwin's track filtering (visibility -> motion -> consistency)
    and adds a hand/stem split + FPS subsample so the optimizer can use the
    pin trajectory directly. All output coordinates are in the **target**
    resolution (`target_H x target_W`), i.e. the optimizer's render canvas.
    """
    bundle = torch.load(tracks_path, map_location='cpu', weights_only=False)
    tracks = bundle['tracks']           # [T, N, 2] in source pixels
    visibility = bundle['visibility']    # [T, N]
    src_H, src_W = bundle['H'], bundle['W']

    vis_frac = visibility.float().mean(dim=0)
    keep_vis = vis_frac >= min_visible_frac
    diff = (tracks[1:] - tracks[:-1]).norm(dim=-1)
    pair_vis = visibility[1:] & visibility[:-1]
    motion = (diff * pair_vis.float()).sum(dim=0)
    motion = torch.where(keep_vis, motion, torch.full_like(motion, -1.0))
    if (motion >= 0).sum() == 0:
        raise SystemExit('[contact] No tracks pass visibility filter; lower --min-visible-frac.')

    thresh = torch.quantile(motion[motion >= 0], motion_percentile / 100.0)
    moving = (motion >= thresh) & keep_vis
    print(f'[contact] visibility-kept {keep_vis.sum()}/{keep_vis.numel()}, '
          f'moving (>={motion_percentile:.0f}%ile) {moving.sum()}/{moving.numel()}')

    neighbor_dist_px = neighbor_dist_frac * src_W
    moving = _motion_consistency_filter(tracks, visibility, moving, neighbor_dist_px)
    print(f'[contact] after motion-consistency: {moving.sum()} tracks (KDTree r={neighbor_dist_px:.1f}px)')

    # Prefer the query_origin tag when present (set by track_video_cotracker.py
    # when seeded from a Grounded-SAM hand bundle). Falls back to the geometric
    # split when running on a pure grid.
    origin_tags = bundle.get('query_origin')
    if origin_tags is not None:
        N = tracks.shape[1]
        hand_mask = torch.tensor([t == 'hand' for t in origin_tags], dtype=torch.bool)
        stem_mask = torch.tensor([t in ('grid', 'cli') for t in origin_tags], dtype=torch.bool)
        # Keep only ones that survived the motion-consistency filter (hand by
        # definition moves; the visibility/motion-consistency filter still applies)
        hand_mask = hand_mask & moving
        stem_mask = stem_mask & moving
        print(f'[contact] split via query_origin: hand {hand_mask.sum()} / stem {stem_mask.sum()}')
    else:
        hand_mask, stem_mask = _split_hand_vs_stem(tracks, visibility, moving, src_W)
        print(f'[contact] split via heuristic: hand {hand_mask.sum()} / stem {stem_mask.sum()}')

    # PhysTwin-style object-mask filter on the stem cluster: require the track
    # to start *inside* a GroundedSAM plant mask at that mask's seed frame.
    # This drops grid points that happened to move with diffusion-noise on the
    # background.
    if plant_mask_path is not None:
        pm = torch.load(plant_mask_path, map_location='cpu', weights_only=False)
        plant_mask = pm['hand_mask'].bool()                   # field name reused from detector script
        seed_t = int(pm['seed_frame'])
        m_H, m_W = pm['src_HW']
        # Mask is in source-video pixels (detector); tracks are at the CoTracker
        # input resolution (src_H/src_W stored in the tracks bundle). Resize the
        # mask to track-space with nearest-neighbor.
        if (m_H, m_W) != (src_H, src_W):
            plant_mask = (
                torch.nn.functional.interpolate(
                    plant_mask.float()[None, None],
                    size=(src_H, src_W),
                    mode='nearest',
                )[0, 0]
            ).bool()
            print(f'[contact] resized plant mask {(m_H, m_W)} -> {(src_H, src_W)}')
            m_H, m_W = src_H, src_W
        # Check each stem track's position at frame seed_t
        keep_stem = stem_mask.clone()
        stem_idx_all = stem_mask.nonzero().flatten().tolist()
        kept = 0
        for n in stem_idx_all:
            if not visibility[seed_t, n]:
                keep_stem[n] = False
                continue
            xy = tracks[seed_t, n]
            xi, yi = int(xy[0].item()), int(xy[1].item())
            if not (0 <= xi < m_W and 0 <= yi < m_H and bool(plant_mask[yi, xi])):
                keep_stem[n] = False
            else:
                kept += 1
        stem_mask = keep_stem
        print(f'[contact] plant-mask filter: stem {stem_mask.sum()} (was {len(stem_idx_all)}, '
              f'mask coverage {plant_mask.float().mean().item() * 100:.1f}%)')

    # FPS subsample each cluster (using frame-0 positions)
    def _subsample(mask, k):
        idx_in = mask.nonzero().flatten()
        if idx_in.numel() == 0:
            return idx_in
        pts0 = tracks[0, idx_in]
        sub = _farthest_point_sample(pts0, k)
        return idx_in[sub]

    hand_idx = _subsample(hand_mask, max_hand)
    stem_idx = _subsample(stem_mask, max_stem)

    sx = target_W / src_W
    sy = target_H / src_H

    def _scale(xy):
        out = xy.clone()
        out[..., 0] = out[..., 0] * sx
        out[..., 1] = out[..., 1] * sy
        return out

    # Per-frame pin: centroid of hand cluster (fallback to all-moving if hand empty)
    pin = torch.full((tracks.shape[0], 2), float('nan'))
    pin_mask = hand_mask if hand_mask.any() else moving
    for t in range(tracks.shape[0]):
        m = pin_mask & visibility[t]
        if m.any():
            pin[t] = tracks[t, m].mean(dim=0)
    pin = _scale(pin)

    return {
        'pixel_pin': pin,                                  # [T, 2]
        'hand_tracks': _scale(tracks[:, hand_idx]),        # [T, K_h, 2]
        'hand_visibility': visibility[:, hand_idx],
        'stem_tracks': _scale(tracks[:, stem_idx]),        # [T, K_s, 2]
        'stem_visibility': visibility[:, stem_idx],
        'src_HW': (src_H, src_W),
        'target_HW': (target_H, target_W),
    }


def project_nodes_to_pixels(nodes: torch.Tensor, camera: dict, H: int, W: int) -> torch.Tensor:
    """Project [N, 3] world points to [N, 2] pixel coords using the renderer's camera.

    Uses the same view + projection matrices as the gaussian rasterizer.
    """
    proj = camera['proj_matrix']               # [4, 4]; rasterizer stores (P @ V).T, so re-transpose
    P = proj.T                                  # [4, 4]
    N = nodes.shape[0]
    homog = torch.cat([nodes, torch.ones(N, 1, device=nodes.device)], dim=-1)  # [N, 4]
    clip = (P @ homog.T).T                      # [N, 4]
    ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-6)
    # ndc in [-1, 1]; map to pixel coords
    u = (ndc[:, 0] * 0.5 + 0.5) * W
    v = (ndc[:, 1] * 0.5 + 0.5) * H
    return torch.stack([u, v], dim=-1)


def pick_anchor_for_pixel(pixel_xy: torch.Tensor, pixel_pin: torch.Tensor) -> int:
    """Index of the projected node closest to the hand pin."""
    d = (pixel_xy - pixel_pin[None, :]).norm(dim=-1)
    return int(d.argmin().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    parser.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--H', type=int, default=256)
    parser.add_argument('--W', type=int, default=256)
    parser.add_argument('--azimuth', type=float, default=30.0)
    parser.add_argument('--elevation', type=float, default=15.0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--tracks', default=None,
                        help='Path to CoTracker bundle (.pt). When given, use track-based pin extraction (PhysTwin-style) instead of a hand detector.')
    parser.add_argument('--motion-percentile', type=float, default=80.0)
    parser.add_argument('--min-visible-frac', type=float, default=0.7)
    parser.add_argument('--neighbor-dist-frac', type=float, default=0.06,
                        help='KDTree consistency radius as a fraction of source width.')
    parser.add_argument('--max-hand', type=int, default=30)
    parser.add_argument('--max-stem', type=int, default=30)
    parser.add_argument('--plant-mask', default=None,
                        help='Path to a plant-mask bundle from detect_hand_grounded_sam.py '
                             '(with --text "potted plant"). Filters stem tracks to those starting inside the mask.')
    parser.add_argument('--colmap-image', default=None,
                        help='When set, use the COLMAP camera for this image name '
                             '(e.g. IMG_1388.JPG) instead of the default azimuth/elevation render camera.')
    args = parser.parse_args()

    device = torch.device(args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    from data.gaussian_plant_loader import load_gaussian_plant_scene
    from models.structure.graph_cleanup import root_branch_graph

    scene = load_gaussian_plant_scene(args.source, args.output_dir)
    tree = root_branch_graph(scene.branch, scene.tube)

    frames = load_video(Path(args.video), args.max_frames, args.H, args.W)
    print(f'Loaded {frames.shape[0]} frames at {args.H}x{args.W}')

    if args.colmap_image is not None:
        from data.colmap_loader import (
            read_cameras_bin, read_images_bin, find_image_by_name, colmap_camera_to_renderer,
        )
        sparse_dir = Path(args.source) / 'sparse' / '0'
        cams_db = read_cameras_bin(sparse_dir / 'cameras.bin')
        imgs_db = read_images_bin(sparse_dir / 'images.bin')
        rec = find_image_by_name(imgs_db, args.colmap_image)
        if rec is None:
            raise SystemExit(f'{args.colmap_image} not in COLMAP images.bin')
        cam_rec = cams_db[rec['cam_id']]
        camera = colmap_camera_to_renderer(rec, cam_rec, args.H, args.W)
        # Move to device
        camera = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in camera.items()}
        print(f'[contact] using COLMAP camera for {args.colmap_image}')
    else:
        from models.renderer.gaussian_renderer import GaussianRenderer
        renderer = GaussianRenderer(image_height=args.H, image_width=args.W, sh_degree=0).to(device)
        target = tree.nodes.mean(0).to(device)
        bbox_extent = (tree.nodes.max(0).values - tree.nodes.min(0).values).max().item()
        camera = renderer.get_camera(
            azimuth=args.azimuth, elevation=args.elevation, radius=float(2.0 * bbox_extent), target=target,
        )

    nodes_px = project_nodes_to_pixels(tree.nodes.to(device), camera, args.H, args.W).cpu()

    if args.tracks is not None:
        tb = tracks_to_contact_bundle(
            Path(args.tracks),
            target_H=args.H, target_W=args.W,
            motion_percentile=args.motion_percentile,
            min_visible_frac=args.min_visible_frac,
            neighbor_dist_frac=args.neighbor_dist_frac,
            max_hand=args.max_hand, max_stem=args.max_stem,
            plant_mask_path=Path(args.plant_mask) if args.plant_mask else None,
        )
        pins = tb['pixel_pin']
        T = pins.shape[0]
        anchors = torch.full((T,), -1, dtype=torch.long)
        for t in range(T):
            if not torch.isnan(pins[t]).any():
                anchors[t] = pick_anchor_for_pixel(nodes_px, pins[t])
        # Stem-track supervision: per-frame, pre-pair each visible stem track
        # with the nearest projected node, so the optimizer can later evaluate
        # a 2D point-projection loss without re-projecting the whole tree.
        stem_tracks = tb['stem_tracks']
        stem_vis = tb['stem_visibility']
        stem_anchor_per_track = torch.full((stem_tracks.shape[1],), -1, dtype=torch.long)
        if stem_tracks.shape[1] > 0:
            # Use frame-0 pixel of each stem track
            for k in range(stem_tracks.shape[1]):
                pt = stem_tracks[0, k]
                if not torch.isnan(pt).any():
                    stem_anchor_per_track[k] = pick_anchor_for_pixel(nodes_px, pt)
        bundle = {
            'anchor_node_id': anchors,
            'pixel_pin': pins,
            'hand_tracks': tb['hand_tracks'],
            'hand_visibility': tb['hand_visibility'],
            'stem_tracks': stem_tracks,
            'stem_visibility': stem_vis,
            'stem_track_node_id': stem_anchor_per_track,
            'src_HW': tb['src_HW'],
            'target_HW': tb['target_HW'],
            'camera': {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in camera.items()},
        }
        print(f'[contact] anchors per frame: unique nodes={len(set(anchors.tolist()))}, '
              f'most common={torch.bincount(anchors.clamp(min=0)).argmax().item() if (anchors>=0).any() else "n/a"}')
    else:
        pins = detect_hand_pixels(frames)
        if pins is None:
            print('[contact] No detector available -> writing no-contact bundle.')
            T = frames.shape[0]
            bundle = {
                'anchor_node_id': torch.full((T,), -1, dtype=torch.long),
                'pixel_pin': torch.full((T, 2), float('nan')),
                'camera': {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in camera.items()},
            }
        else:
            T = pins.shape[0]
            anchors = torch.tensor([pick_anchor_for_pixel(nodes_px, pins[t]) for t in range(T)], dtype=torch.long)
            bundle = {
                'anchor_node_id': anchors,
                'pixel_pin': pins,
                'camera': {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in camera.items()},
            }

    torch.save(bundle, out)
    print(f'  saved contact bundle -> {out}')


if __name__ == '__main__':
    main()
