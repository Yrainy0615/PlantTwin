"""
Stage 1 SDS-only optimization: optimize spring-mass physics parameters
so that simulated plant motion looks physically plausible via video diffusion SDS.

No target video needed. Only requires:
- Static 3DGS plant (from TRELLIS)
- Text prompt describing desired motion
- Video diffusion model (ModelScope T2V) for SDS guidance

Pipeline:
    learnable params → simulate → render video → SDS loss → update params

Usage:
    python scripts/train_sds_e2e.py \
        --ply data/plants_3dgs/.../gaussian.ply \
        --prompt "a single plant gently swaying in the wind"
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../third_party/TRELLIS'))
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import imageio

from simulation.spring_mass import SpringMassSimulator
from models.renderer import GaussianRenderer
from optimization.sds_guidance import VideoSDSGuidance


def load_gaussian_ply(ply_path):
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


def save_video(frames_tensor, path, fps=8):
    """Save [T, 3, H, W] tensor as mp4."""
    frames = (frames_tensor.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    imageio.mimsave(str(path), list(frames), fps=fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply', type=str, required=True)
    parser.add_argument('--prompt', type=str, default='a single plant gently swaying in the wind')
    parser.add_argument('--output_dir', type=str, default='outputs/sds_e2e')
    parser.add_argument('--n_sim', type=int, default=512)
    parser.add_argument('--k_neighbors', type=int, default=16)
    parser.add_argument('--n_step', type=int, default=50)
    parser.add_argument('--n_frames', type=int, default=16)
    parser.add_argument('--render_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--guidance_scale', type=float, default=50.0)
    parser.add_argument('--sds_weight', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Gaussians
    print(f"Loading PLY: {args.ply}")
    xyz, scales, rots, opacities, colors = load_gaussian_ply(args.ply)
    xyz, scales, rots, opacities, colors = [t.to(device) for t in [xyz, scales, rots, opacities, colors]]
    N = xyz.shape[0]
    print(f"  {N} Gaussians")

    # Subsample for simulation
    idx = torch.randperm(N, device=device)[:args.n_sim].sort().values
    xyz_sim = xyz[idx]

    # Build simulator (no gravity — plant is rooted)
    sim = SpringMassSimulator(
        xyz_sim, k_neighbors=args.k_neighbors, n_step=args.n_step,
        damping=True, gravity=[0, 0, 0]
    ).to(device)
    print(f"  Simulator: {args.n_sim} pts, KNN={args.k_neighbors}, substeps={args.n_step}")

    # Renderer
    renderer = GaussianRenderer(
        image_height=args.render_size, image_width=args.render_size, fov=40
    ).to(device)
    camera = renderer.get_camera(azimuth=0, elevation=14, radius=2.0, target=xyz.mean(0).detach())

    # SDS guidance
    print(f"Loading SDS guidance...")
    guidance = VideoSDSGuidance(guidance_scale=args.guidance_scale)
    text_emb = guidance.encode_prompt(args.prompt)
    print(f"  Prompt: '{args.prompt}'")

    # Learnable physics parameters
    log_k = torch.nn.Parameter(torch.full((args.n_sim,), 1.5, device=device))   # k ~ 30
    log_damp = torch.nn.Parameter(torch.full((args.n_sim,), 0.5, device=device)) # damp ~ 3
    init_vel = torch.nn.Parameter(torch.tensor([[0.05, 0.0, 0.0]], device=device))

    optimizer = torch.optim.Adam([
        {'params': [log_k], 'lr': args.lr},
        {'params': [log_damp], 'lr': args.lr * 0.5},
        {'params': [init_vel], 'lr': args.lr * 3.0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.01)

    print(f"\nSDS optimization for {args.epochs} epochs, {args.n_frames} frames...")
    print(f"  sds_weight={args.sds_weight}, guidance_scale={args.guidance_scale}")

    best_loss = float('inf')
    for epoch in tqdm(range(args.epochs), desc="SDS"):
        optimizer.zero_grad()

        # Clamp params
        with torch.no_grad():
            log_k.data.clamp_(0.0, 3.5)
            log_damp.data.clamp_(-1.0, 2.0)
            init_vel.data.clamp_(-0.3, 0.3)

        # Simulate
        k = 10 ** log_k
        damp = (10 ** log_damp).unsqueeze(1).expand(-1, args.k_neighbors)
        physics_params = {
            'k': k,
            'm': torch.ones(args.n_sim, device=device),
            'damp': damp,
            'init_velocity': init_vel,
        }
        traj = sim(physics_params, n_frames=args.n_frames, xyz_all=xyz)

        if torch.isnan(traj).any():
            tqdm.write(f"  [Epoch {epoch+1}] NaN trajectory, skip")
            continue

        # Render video
        video = renderer.render_trajectory(traj, scales, rots, opacities, colors, camera=camera)
        # video: [T, 3, H, W]

        # SDS loss
        video_hwc = video.permute(0, 2, 3, 1).contiguous()  # [T, H, W, 3]
        loss_sds = guidance.compute_sds_loss(video_hwc, text_emb)

        # Regularization
        reg_smooth = 0.01 * (log_k[1:] - log_k[:-1]).pow(2).mean()
        reg_motion = -0.001 * (traj[-1] - traj[0]).pow(2).mean()  # encourage some motion

        loss = args.sds_weight * loss_sds + reg_smooth + reg_motion

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([log_k, log_damp, init_vel], max_norm=1.0)

        if torch.isnan(grad_norm):
            tqdm.write(f"  [Epoch {epoch+1}] NaN grad, skip")
            optimizer.zero_grad()
            continue

        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        if loss_val < best_loss and not np.isnan(loss_val):
            best_loss = loss_val

        if (epoch + 1) % 25 == 0 or epoch == 0:
            k_val = (10 ** log_k).mean().item()
            d_val = (10 ** log_damp).mean().item()
            vel = init_vel.data[0].tolist()
            disp = (traj[-1] - traj[0]).norm(dim=1).mean().item()
            tqdm.write(
                f"  [Epoch {epoch+1:3d}] loss={loss_val:.2f} sds={loss_sds.item():.1f} "
                f"k={k_val:.1f} damp={d_val:.2f} vel=[{vel[0]:.3f},{vel[1]:.3f},{vel[2]:.3f}] "
                f"disp={disp:.5f} gnorm={grad_norm.item():.3f}"
            )

        # Save video periodically
        if (epoch + 1) % 100 == 0:
            save_video(video.detach(), output_dir / f"sim_ep{epoch+1:04d}.mp4")

    # Final save
    print(f"\nSaving results...")
    result = {
        'k': (10 ** log_k).detach().cpu(),
        'damp': (10 ** log_damp).detach().cpu(),
        'init_velocity': init_vel.detach().cpu(),
        'sample_idx': idx.cpu(),
        'best_loss': best_loss,
        'args': vars(args),
    }
    torch.save(result, output_dir / "params.pt")
    save_video(video.detach(), output_dir / "final_sim.mp4")
    print(f"Done. Best loss: {best_loss:.4f}")
    print(f"  k: [{result['k'].min():.1f}, {result['k'].max():.1f}], mean={result['k'].mean():.1f}")
    print(f"  vel: {result['init_velocity'].tolist()}")
    print(f"  Results in {output_dir}")


if __name__ == '__main__':
    main()
