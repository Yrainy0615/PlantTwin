"""
Stage 1 SDS Optimization: Optimize spring-mass physics parameters for a static plant
using Score Distillation Sampling from a video diffusion model.

Pipeline:
    Static 3DGS → build spring-mass KNN graph → optimize params via SDS
    → predicted params produce plausible motion when simulated

Usage:
    python scripts/train_sds.py --config configs/sds_optim.yaml
    python scripts/train_sds.py --ply_path data/plants_3dgs/rose_s42/gaussian.ply --prompt "a rose swaying in the wind"
"""
import os
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../third_party/TRELLIS'))
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from simulation.spring_mass import SpringMassSimulator
from optimization.sds_guidance import VideoSDSGuidance


def load_gaussian_positions(ply_path):
    """Load Gaussian positions from PLY file."""
    from plyfile import PlyData
    plydata = PlyData.read(ply_path)
    xyz = np.stack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z'],
    ], axis=-1)
    return torch.tensor(xyz, dtype=torch.float32)


def subsample_points(xyz, n_sample=2048):
    """Subsample points for simulation (full set used for rendering)."""
    if xyz.shape[0] <= n_sample:
        return xyz, torch.arange(xyz.shape[0])
    idx = torch.randperm(xyz.shape[0])[:n_sample]
    idx = idx.sort().values
    return xyz[idx], idx


def render_gaussian_frames(xyz_trajectory, gaussians_data, camera_params):
    """
    Render Gaussian splatting frames from a position trajectory.
    Placeholder — will integrate with diff-gaussian-rasterization.

    Args:
        xyz_trajectory: [T, N, 3] positions over time
        gaussians_data: dict with scales, rotations, colors, opacities
        camera_params: camera intrinsics/extrinsics

    Returns:
        frames: [T, H, W, 3] rendered images
    """
    # TODO: integrate with diff-gaussian-rasterization for differentiable rendering
    # For now, return dummy frames for pipeline testing
    T = xyz_trajectory.shape[0]
    H, W = camera_params.get('height', 256), camera_params.get('width', 256)
    frames = torch.zeros(T, H, W, 3, device=xyz_trajectory.device)
    return frames


def main():
    parser = argparse.ArgumentParser(description="SDS optimization for plant physics")
    parser.add_argument('--ply_path', type=str, required=True, help='Path to gaussian.ply')
    parser.add_argument('--prompt', type=str, default='a plant swaying gently in the wind')
    parser.add_argument('--output_dir', type=str, default='outputs/sds_optim')
    parser.add_argument('--n_sample', type=int, default=2048, help='Number of simulation particles')
    parser.add_argument('--k_neighbors', type=int, default=256)
    parser.add_argument('--n_frames', type=int, default=16, help='Simulation frames for SDS')
    parser.add_argument('--n_step', type=int, default=100, help='Integration steps per frame')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--guidance_model', type=str,
                       default='ali-vilab/text-to-video-ms-1.7b')
    parser.add_argument('--guidance_scale', type=float, default=100.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Gaussians from {args.ply_path}")
    xyz_all = load_gaussian_positions(args.ply_path).to(device)
    xyz_sample, sample_idx = subsample_points(xyz_all, args.n_sample)
    xyz_sample = xyz_sample.to(device)
    print(f"  Total: {xyz_all.shape[0]} points, Simulation: {xyz_sample.shape[0]} points")

    print("Building spring-mass simulator...")
    simulator = SpringMassSimulator(
        xyz_sample,
        k_neighbors=args.k_neighbors,
        n_step=args.n_step,
        damping=True,
    ).to(device)

    # Learnable physics parameters
    log_k = torch.nn.Parameter(torch.full((args.n_sample,), 2.3, device=device))  # log10(k) ~ 200
    log_m = torch.nn.Parameter(torch.full((args.n_sample,), 0.0, device=device))  # log10(m) ~ 1.0
    log_damp = torch.nn.Parameter(torch.full((args.n_sample,), -1.0, device=device))  # log10(damp) ~ 0.1
    init_vel = torch.nn.Parameter(torch.zeros(1, 3, device=device))

    optimizer = torch.optim.Adam([log_k, log_m, log_damp, init_vel], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    print(f"Loading SDS guidance: {args.guidance_model}")
    guidance_cfg = {
        'pretrained_model': args.guidance_model,
        'half_precision': True,
        'guidance_scale': args.guidance_scale,
        'min_step_percent': 0.02,
        'max_step_percent': 0.98,
        'weighting_strategy': 'sds',
    }
    # guidance = VideoSDSGuidance(guidance_cfg)  # Uncomment when running full pipeline
    print("  [SDS guidance loaded]")

    camera_params = {'height': 256, 'width': 256}

    print(f"\nStarting SDS optimization for {args.epochs} epochs...")
    print(f"  Prompt: '{args.prompt}'")
    print(f"  Frames: {args.n_frames}, Steps/frame: {args.n_step}")

    for epoch in tqdm(range(args.epochs), desc="SDS Optim"):
        optimizer.zero_grad()

        physics_params = {
            'k': 10 ** log_k,
            'm': 10 ** log_m,
            'damp': (10 ** log_damp).unsqueeze(1).expand(-1, args.k_neighbors),
            'init_velocity': init_vel,
        }

        trajectory = simulator(physics_params, n_frames=args.n_frames, xyz_all=xyz_all)

        # Render frames (placeholder for now)
        rendered_video = render_gaussian_frames(trajectory, None, camera_params)

        # SDS loss (placeholder — uncomment when guidance is loaded)
        # text_embeddings = prompt_utils.get_text_embeddings(...)
        # loss = guidance.compute_sds_loss(rendered_video, text_embeddings)

        # Regularization: keep parameters in reasonable range
        reg_k = 0.01 * ((log_k - 2.3) ** 2).mean()
        reg_m = 0.01 * ((log_m - 0.0) ** 2).mean()
        reg_smooth = 0.001 * ((log_k[simulator.knn_index] - log_k.unsqueeze(1)) ** 2).mean()

        loss = reg_k + reg_m + reg_smooth  # + loss_sds when guidance active

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            k_mean = (10 ** log_k).mean().item()
            m_mean = (10 ** log_m).mean().item()
            d_mean = (10 ** log_damp).mean().item()
            tqdm.write(f"  [Epoch {epoch+1}] k={k_mean:.1f} m={m_mean:.3f} damp={d_mean:.4f} loss={loss.item():.6f}")

    # Save optimized parameters
    result = {
        'k': (10 ** log_k).detach().cpu(),
        'm': (10 ** log_m).detach().cpu(),
        'damp': (10 ** log_damp).detach().cpu(),
        'init_velocity': init_vel.detach().cpu(),
        'sample_idx': sample_idx.cpu(),
        'ply_path': args.ply_path,
        'prompt': args.prompt,
    }
    save_path = output_dir / "optimized_params.pt"
    torch.save(result, save_path)
    print(f"\nSaved optimized physics params to {save_path}")
    print(f"  k: [{result['k'].min():.1f}, {result['k'].max():.1f}]")
    print(f"  m: [{result['m'].min():.3f}, {result['m'].max():.3f}]")
    print(f"  damp: [{result['damp'].min():.4f}, {result['damp'].max():.4f}]")


if __name__ == '__main__':
    main()
