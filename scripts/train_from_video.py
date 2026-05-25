"""
Train/test physics parameter optimization from a single video.
Self-supervised: predict params → simulate → render → compare with input video.

Usage:
    python scripts/train_from_video.py \
        --video data/plants_video/a_amaryllis_.../motion.mp4 \
        --ply data/plants_3dgs/a_amaryllis_.../gaussian.ply
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../third_party/TRELLIS'))
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'

import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import imageio
from tqdm import tqdm

from simulation.spring_mass import SpringMassSimulator
from models.renderer import GaussianRenderer


def load_video_frames(video_path, n_frames=16, size=256):
    """Load video and return [T, 3, H, W] tensor in [0,1]."""
    reader = imageio.get_reader(str(video_path))
    frames = []
    for i, frame in enumerate(reader):
        if i >= n_frames:
            break
        frames.append(torch.from_numpy(frame).float() / 255.0)
    reader.close()
    while len(frames) < n_frames:
        frames.append(frames[-1])
    video = torch.stack(frames)  # [T, H, W, 3]
    video = video.permute(0, 3, 1, 2)  # [T, 3, H, W]
    video = F.interpolate(video, size=(size, size), mode='bilinear', align_corners=False)
    return video


def load_gaussian_ply(ply_path):
    """Load all Gaussian attributes from PLY."""
    from plyfile import PlyData
    plydata = PlyData.read(str(ply_path))
    v = plydata['vertex']
    xyz = torch.tensor(np.stack([v['x'], v['y'], v['z']], -1), dtype=torch.float32)
    scales = torch.exp(torch.tensor(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], -1), dtype=torch.float32))
    rots = torch.tensor(np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], -1), dtype=torch.float32)
    rots = rots / (rots.norm(dim=1, keepdim=True) + 1e-8)
    opacities = torch.sigmoid(torch.tensor(v['opacity'][:, None], dtype=torch.float32))
    colors = torch.sigmoid(torch.tensor(np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], -1), dtype=torch.float32))
    return xyz, scales, rots, opacities, colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--ply', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/video_train')
    parser.add_argument('--n_frames', type=int, default=8)
    parser.add_argument('--n_sim', type=int, default=512, help='Simulation particles')
    parser.add_argument('--k_neighbors', type=int, default=16)
    parser.add_argument('--n_step', type=int, default=30, help='Integration substeps per frame')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--render_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load target video
    print(f"Loading video: {args.video}")
    target_video = load_video_frames(args.video, n_frames=args.n_frames, size=args.render_size).to(device)
    print(f"  Target video: {target_video.shape}")  # [T, 3, H, W]

    # Load Gaussians
    print(f"Loading PLY: {args.ply}")
    xyz, scales, rots, opacities, colors = load_gaussian_ply(args.ply)
    xyz, scales, rots, opacities, colors = [t.to(device) for t in [xyz, scales, rots, opacities, colors]]
    N = xyz.shape[0]
    print(f"  Gaussians: {N}")

    # Subsample for simulation
    idx = torch.randperm(N, device=device)[:args.n_sim].sort().values
    xyz_sim = xyz[idx]

    # Build simulator - no gravity for initial training (plant is held by pot/root)
    sim = SpringMassSimulator(xyz_sim, k_neighbors=args.k_neighbors, n_step=args.n_step,
                               damping=True, gravity=[0, 0, 0]).to(device)
    print(f"  Simulator: {args.n_sim} particles, KNN={args.k_neighbors}, substeps={args.n_step}, no gravity")

    # Renderer - use same camera as TRELLIS preview (front view)
    renderer = GaussianRenderer(image_height=args.render_size, image_width=args.render_size, fov=40).to(device)
    # TRELLIS render_video uses yaw=0, pitch=0.25rad (~14deg), radius=2
    camera = renderer.get_camera(azimuth=0, elevation=14, radius=2.0, target=xyz.mean(0).detach())

    # Learnable parameters - tuned for TRELLIS plant scale (~0.5 extent)
    log_k = torch.nn.Parameter(torch.full((args.n_sim,), 1.0, device=device))  # k ~ 10 (soft)
    log_damp = torch.nn.Parameter(torch.full((args.n_sim,), 0.0, device=device))  # damp ~ 1.0
    init_vel = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 0.0]], device=device))

    optimizer = torch.optim.Adam([
        {'params': [log_k], 'lr': args.lr},
        {'params': [log_damp], 'lr': args.lr * 0.5},
        {'params': [init_vel], 'lr': args.lr * 2.0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.01)

    print(f"\nTraining for {args.epochs} epochs...")
    best_loss = float('inf')
    losses = []

    for epoch in tqdm(range(args.epochs), desc="Training"):
        optimizer.zero_grad()

        # Clamp parameters to prevent NaN
        with torch.no_grad():
            log_k.data.clamp_(0.0, 3.0)        # k in [1, 1000]
            log_damp.data.clamp_(-1.0, 2.0)    # damp in [0.1, 100]
            init_vel.data.clamp_(-0.5, 0.5)

        # Build physics params
        k = 10 ** log_k
        damp = (10 ** log_damp).unsqueeze(1).expand(-1, args.k_neighbors)
        physics_params = {
            'k': k,
            'm': torch.ones(args.n_sim, device=device),
            'damp': damp,
            'init_velocity': init_vel,
        }

        # Simulate
        traj = sim(physics_params, n_frames=args.n_frames, xyz_all=xyz)

        # Check for NaN in trajectory
        if torch.isnan(traj).any():
            tqdm.write(f"  [Epoch {epoch+1}] NaN in trajectory, skipping")
            continue

        # Render
        rendered_video = renderer.render_trajectory(traj, scales, rots, opacities, colors, camera=camera)

        # Temporal difference loss: compare motion delta, not static appearance
        # target_diff[t] = target[t] - target[0]: motion in target video
        # render_diff[t] = render[t] - render[0]: motion from simulation
        target_diff = target_video[1:] - target_video[0:1]  # [T-1, 3, H, W]
        render_diff = rendered_video[1:] - rendered_video[0:1]

        loss_motion_l1 = F.l1_loss(render_diff, target_diff)
        loss_motion_mse = F.mse_loss(render_diff, target_diff)

        # Also match the first frame (static appearance, lower weight)
        loss_static = F.l1_loss(rendered_video[0:1], target_video[0:1])

        # Regularization
        reg_vel = 0.001 * (init_vel ** 2).sum()

        loss = loss_motion_l1 + 0.5 * loss_motion_mse + 0.01 * loss_static + reg_vel

        loss.backward()

        # Aggressive gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_([log_k, log_damp, init_vel], max_norm=0.5)

        # Skip step if grad is NaN
        if torch.isnan(grad_norm):
            tqdm.write(f"  [Epoch {epoch+1}] NaN grad, skipping")
            optimizer.zero_grad()
            continue

        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            k_val = (10 ** log_k).mean().item()
            d_val = (10 ** log_damp).mean().item()
            vel = init_vel.data.tolist()[0]
            tqdm.write(
                f"  [Epoch {epoch+1:3d}] loss={loss.item():.6f} "
                f"(motion_l1={loss_motion_l1.item():.6f} motion_mse={loss_motion_mse.item():.6f}) "
                f"k={k_val:.1f} damp={d_val:.3f} vel=[{vel[0]:.4f},{vel[1]:.4f},{vel[2]:.4f}] "
                f"gnorm={grad_norm.item():.4f}"
            )

        # Save visualization at intervals
        if (epoch + 1) % 50 == 0:
            save_comparison(target_video, rendered_video, output_dir / f"compare_ep{epoch+1:04d}.png")

    # Save final results
    result = {
        'k': (10 ** log_k).detach().cpu(),
        'damp': (10 ** log_damp).detach().cpu(),
        'init_velocity': init_vel.detach().cpu(),
        'sample_idx': idx.cpu(),
        'final_loss': best_loss,
        'args': vars(args),
    }
    torch.save(result, output_dir / "optimized_params.pt")
    save_comparison(target_video, rendered_video.detach(), output_dir / "final_compare.png")
    print(f"\nDone. Best loss: {best_loss:.5f}")
    print(f"Results saved to {output_dir}")


def ssim_simple(pred, target):
    """Simplified SSIM between [T,3,H,W] tensors."""
    mu_p = F.avg_pool2d(pred.view(-1, 3, pred.shape[2], pred.shape[3]), 11, 1, 5)
    mu_t = F.avg_pool2d(target.view(-1, 3, target.shape[2], target.shape[3]), 11, 1, 5)
    sigma_p = F.avg_pool2d(pred.view(-1, 3, pred.shape[2], pred.shape[3]) ** 2, 11, 1, 5) - mu_p ** 2
    sigma_t = F.avg_pool2d(target.view(-1, 3, target.shape[2], target.shape[3]) ** 2, 11, 1, 5) - mu_t ** 2
    sigma_pt = F.avg_pool2d(
        pred.view(-1, 3, pred.shape[2], pred.shape[3]) * target.view(-1, 3, target.shape[2], target.shape[3]),
        11, 1, 5
    ) - mu_p * mu_t
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p + sigma_t + C2))
    return ssim_map.mean()


def perceptual_loss_simple(pred, target):
    """Simple perceptual-like loss using multi-scale features."""
    loss = 0
    p, t = pred.view(-1, 3, pred.shape[2], pred.shape[3]), target.view(-1, 3, target.shape[2], target.shape[3])
    for scale in [1, 2, 4]:
        if scale > 1:
            p_s = F.avg_pool2d(p, scale)
            t_s = F.avg_pool2d(t, scale)
        else:
            p_s, t_s = p, t
        loss += F.l1_loss(p_s, t_s)
    return loss / 3.0


def save_comparison(target, rendered, path):
    """Save side-by-side comparison of target vs rendered (first frame)."""
    t_frame = (target[0].permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    r_frame = (rendered[0].permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    comparison = np.concatenate([t_frame, r_frame], axis=1)
    Image.fromarray(comparison).save(str(path))


if __name__ == '__main__':
    main()
