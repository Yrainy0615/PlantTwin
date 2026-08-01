"""Overlay the contact bundle (hand cluster, stem cluster, per-frame pin,
projected branch nodes, chosen anchor) on top of the source video.

Use this to sanity-check that:
- the hand cluster (red) actually sits on the moving hand region
- the stem cluster (blue) sits on the part of the plant that follows
- the per-frame pin (yellow circle) is close to where you'd expect contact
- the chosen anchor node (green star) lies inside the plant near the hand
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision.io as tvio
from matplotlib.animation import FFMpegWriter
from torchvision.transforms.functional import resize


def _project_nodes(nodes_world: torch.Tensor, cam: dict, H: int, W: int) -> torch.Tensor:
    P = cam['proj_matrix'].T
    homog = torch.cat([nodes_world, torch.ones(nodes_world.shape[0], 1)], dim=-1)
    clip = (P @ homog.T).T
    ndc = clip[:, :3] / clip[:, 3:4].clamp_min(1e-6)
    u = (ndc[:, 0] * 0.5 + 0.5) * W
    v = (ndc[:, 1] * 0.5 + 0.5) * H
    return torch.stack([u, v], dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--contact', required=True)
    parser.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    parser.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    parser.add_argument('--out', required=True)
    parser.add_argument('--H', type=int, default=256)
    parser.add_argument('--W', type=int, default=256)
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--plant-mask', default=None,
                        help='Optional plant_mask.pt to overlay as a contour.')
    parser.add_argument('--hide-nodes', action='store_true',
                        help='Suppress the gray projected branch nodes.')
    args = parser.parse_args()

    frames, _, _ = tvio.read_video(args.video, pts_unit='sec')
    frames = frames.permute(0, 3, 1, 2).float() / 255.0
    frames = resize(frames, [args.H, args.W])

    bundle = torch.load(args.contact, map_location='cpu', weights_only=False)
    pin = bundle['pixel_pin']                    # [T, 2]
    hand = bundle.get('hand_tracks')             # [T, K_h, 2]
    hand_v = bundle.get('hand_visibility')
    stem = bundle.get('stem_tracks')             # [T, K_s, 2]
    stem_v = bundle.get('stem_visibility')
    anchors = bundle.get('anchor_node_id')       # [T]
    cam = bundle.get('camera', {})

    from data.gaussian_plant_loader import load_gaussian_plant_scene
    from models.structure.graph_cleanup import root_branch_graph
    scene = load_gaussian_plant_scene(args.source, args.output_dir)
    tree = root_branch_graph(scene.branch, scene.tube)

    # Re-project (we only saved the camera as tensors; bring to cpu)
    cam_cpu = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in cam.items()}
    nodes_px = _project_nodes(tree.nodes, cam_cpu, args.H, args.W)

    plant_mask_resized = None
    if args.plant_mask is not None:
        pm = torch.load(args.plant_mask, map_location='cpu', weights_only=False)
        m = pm['hand_mask'].bool()                                  # field reused for object masks
        plant_mask_resized = (
            torch.nn.functional.interpolate(
                m.float()[None, None], size=(args.H, args.W), mode='nearest',
            )[0, 0]
        ).bool().numpy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(args.W / 80.0, args.H / 80.0), dpi=80)
    writer = FFMpegWriter(fps=args.fps)
    with writer.saving(fig, str(out_path), dpi=80):
        for t in range(frames.shape[0]):
            ax.clear()
            ax.imshow(frames[t].permute(1, 2, 0).numpy())
            ax.set_xlim(0, args.W); ax.set_ylim(args.H, 0); ax.set_axis_off()

            if not args.hide_nodes:
                # Projected branch nodes (light gray) — note: our render camera
                # is not aligned to the video camera, so this is for reference
                # only and will not lie exactly on the plant pixels.
                ax.scatter(nodes_px[:, 0], nodes_px[:, 1], c='gray', s=1.5, alpha=0.18)

            if plant_mask_resized is not None:
                ax.contour(plant_mask_resized.astype(float),
                           levels=[0.5], colors=['white'], linewidths=0.6, alpha=0.6)

            if stem is not None and stem.shape[1] > 0:
                mask = stem_v[t] if stem_v is not None else torch.ones(stem.shape[1], dtype=torch.bool)
                ax.scatter(stem[t, mask, 0], stem[t, mask, 1], c='deepskyblue', s=10, label='stem' if t == 0 else None)
            if hand is not None and hand.shape[1] > 0:
                mask = hand_v[t] if hand_v is not None else torch.ones(hand.shape[1], dtype=torch.bool)
                ax.scatter(hand[t, mask, 0], hand[t, mask, 1], c='red', s=12, label='hand' if t == 0 else None)

            if not torch.isnan(pin[t]).any():
                ax.scatter([pin[t, 0]], [pin[t, 1]], facecolors='none', edgecolors='yellow',
                           s=200, linewidths=2.0, label='pin' if t == 0 else None)

            if anchors is not None and anchors[t] >= 0:
                p = nodes_px[anchors[t]]
                ax.scatter([p[0]], [p[1]], c='lime', marker='*', s=120,
                           label='anchor node' if t == 0 else None)

            if t == 0:
                ax.legend(loc='upper left', fontsize=7)
            writer.grab_frame()
    plt.close(fig)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
