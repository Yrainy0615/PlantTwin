"""
Part-aware plant physics pretrain.

Pipeline:
    Static 3DGS (canonical frame)
    → Gaussian features (pos + SH + cov + opacity) + Structure Info
    → PlantMaterialNetwork (KNNTransformer backbone → spring-mass params)
    → Spring-mass simulation (trunk anchoring + wind drag)
    → Differentiable rendering → video
    → SDS loss + Structure loss → backprop through network

Usage:
    python scripts/train_part_aware_pretrain.py \
        --ply data/plants_3dgs/rose_s42/gaussian.ply \
        --prompt "a rose plant swaying gently in the wind"
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
from models.physics_decoder.plant_material_network import PlantMaterialNetwork
from models.structure import StructureExtractor
from optimization.sds_guidance import VideoSDSGuidance
from optimization.structure_loss import StructureLoss


def load_gaussian_ply(ply_path):
    from plyfile import PlyData
    plydata = PlyData.read(str(ply_path))
    v = plydata['vertex']
    xyz = torch.tensor(np.stack([v['x'], v['y'], v['z']], -1), dtype=torch.float32)
    scales = torch.exp(torch.tensor(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], -1), dtype=torch.float32))
    rots = torch.tensor(np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], -1), dtype=torch.float32)
    rots = rots / (rots.norm(dim=1, keepdim=True) + 1e-8)
    opacities = torch.sigmoid(torch.tensor(v['opacity'][:, None], dtype=torch.float32))

    # SH coefficients — DC band
    sh_dc = torch.tensor(np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], -1), dtype=torch.float32)
    SH_C0 = 0.28209479177387814
    colors = (SH_C0 * sh_dc + 0.5).clamp(0.0, 1.0)

    # Load higher-order SH if available (for feature construction)
    sh_keys = [k for k in v.data.dtype.names if k.startswith('f_rest_')]
    if sh_keys:
        sh_rest = torch.tensor(np.stack([v[k] for k in sh_keys], -1), dtype=torch.float32)
        shs = torch.cat([sh_dc, sh_rest], dim=-1)
    else:
        shs = sh_dc

    return xyz, scales, rots, opacities, colors, shs


def build_gaussian_features(xyz, shs, cov_flat, opacity, normalize=True):
    """
    Construct input features for PlantMaterialNetwork, following OmniPhysGS convention:
    features = cat(pos, normalized_shs, normalized_cov, normalized_opacity)
    """
    n = xyz.shape[0]
    if normalize:
        def norm_feat(x):
            x_flat = x.reshape(n, -1)
            return (x_flat - x_flat.mean(0, keepdim=True)) / (x_flat.std(0, keepdim=True) + 1e-8)
        features = torch.cat([xyz, norm_feat(shs), norm_feat(cov_flat), norm_feat(opacity)], dim=1)
    else:
        features = torch.cat([xyz, shs.reshape(n, -1), cov_flat.reshape(n, -1), opacity.reshape(n, -1)], dim=1)
    return features


def compute_cov_flat(scales, rots):
    """Compute upper-triangular covariance from scales and rotations (6D)."""
    # Build rotation matrix from quaternion
    w, x, y, z = rots[:, 0], rots[:, 1], rots[:, 2], rots[:, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
        2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)

    S = torch.zeros_like(R)
    S[:, 0, 0] = scales[:, 0]
    S[:, 1, 1] = scales[:, 1]
    S[:, 2, 2] = scales[:, 2]

    cov = R @ S @ S.transpose(1, 2) @ R.transpose(1, 2)
    # Upper triangular: xx, xy, xz, yy, yz, zz
    cov_flat = torch.stack([
        cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
        cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2],
    ], dim=-1)
    return cov_flat


def save_video(frames_tensor, path, fps=8):
    frames = (frames_tensor.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    imageio.mimsave(str(path), list(frames), fps=fps)


def main():
    parser = argparse.ArgumentParser(description="Part-aware plant physics pretrain")
    parser.add_argument('--ply', type=str, required=True)
    parser.add_argument('--prompt', type=str, default='a single plant gently swaying in the wind')
    parser.add_argument('--output_dir', type=str, default='outputs/part_aware_pretrain')

    # Structure
    parser.add_argument('--structure_method', type=str, default='heuristic',
                        choices=['heuristic', 'dino', 'dino_cluster'])
    parser.add_argument('--n_parts', type=int, default=3)
    parser.add_argument('--part_embed_dim', type=int, default=32)

    # Network
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--depth', type=int, default=0)
    parser.add_argument('--num_groups', type=int, default=2048)
    parser.add_argument('--group_size', type=int, default=32)

    # Simulation
    parser.add_argument('--n_sim', type=int, default=2048)
    parser.add_argument('--k_neighbors', type=int, default=32)
    parser.add_argument('--n_step', type=int, default=50)
    parser.add_argument('--n_frames', type=int, default=16)
    parser.add_argument('--dt', type=float, default=0.03)
    parser.add_argument('--wind', type=float, nargs=3, default=[0.3, 0.0, 0.0])

    # Rendering
    parser.add_argument('--render_size', type=int, default=256)

    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--internal_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--guidance_scale', type=float, default=50.0)
    parser.add_argument('--sds_weight', type=float, default=1.0)

    # Structure loss weights
    parser.add_argument('--lambda_anchor', type=float, default=10.0)
    parser.add_argument('--lambda_branch', type=float, default=1.0)
    parser.add_argument('--lambda_consistency', type=float, default=0.1)
    parser.add_argument('--lambda_ordering', type=float, default=0.5)
    parser.add_argument('--lambda_attach', type=float, default=1.0)
    parser.add_argument('--lambda_smooth', type=float, default=0.01)

    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load Gaussians ──
    print(f"Loading PLY: {args.ply}")
    xyz, scales, rots, opacities, colors, shs = load_gaussian_ply(args.ply)
    xyz, scales, rots, opacities, colors, shs = [
        t.to(device) for t in [xyz, scales, rots, opacities, colors, shs]
    ]
    N = xyz.shape[0]
    print(f"  {N} Gaussians loaded")

    # ── Subsample for simulation ──
    if N > args.n_sim:
        idx = torch.randperm(N, device=device)[:args.n_sim].sort().values
    else:
        idx = torch.arange(N, device=device)
        args.n_sim = N
    xyz_sim = xyz[idx]
    colors_sim = colors[idx]
    shs_sim = shs[idx]
    scales_sim = scales[idx]
    rots_sim = rots[idx]
    opacities_sim = opacities[idx]

    # ── Structure extraction ──
    print(f"Extracting structure ({args.structure_method})...")
    structure_extractor = StructureExtractor(
        method=args.structure_method,
        n_parts=args.n_parts,
        part_embed_dim=args.part_embed_dim,
    ).to(device)

    part_labels, structure_features = structure_extractor(xyz_sim, colors_sim)
    print(f"  Trunk: {part_labels.trunk_mask.sum()}, "
          f"Branch: {part_labels.branch_mask.sum()}, "
          f"Leaf: {part_labels.leaf_mask.sum()}")

    # ── Build Gaussian features (OmniPhysGS convention) ──
    cov_flat = compute_cov_flat(scales_sim, rots_sim)
    gaussian_features = build_gaussian_features(xyz_sim, shs_sim, cov_flat, opacities_sim)
    gaussian_feat_dim = gaussian_features.shape[1]
    structure_feat_dim = structure_features.shape[1]
    print(f"  Feature dims: gaussian={gaussian_feat_dim}, structure={structure_feat_dim}")

    # ── Build PlantMaterialNetwork ──
    num_groups = min(args.num_groups, args.n_sim // 2)
    material_net = PlantMaterialNetwork(
        gaussian_feat_dim=gaussian_feat_dim,
        structure_feat_dim=structure_feat_dim,
        num_groups=num_groups,
        group_size=args.group_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
    ).to(device)
    print(f"  PlantMaterialNetwork: groups={num_groups}, hidden={args.hidden_size}, depth={args.depth}")

    # ── Build simulator ──
    wind_vel = torch.tensor(args.wind, dtype=torch.float32, device=device)
    sim = SpringMassSimulator(
        xyz_sim, k_neighbors=args.k_neighbors, n_step=args.n_step, dt=args.dt,
        damping=True, gravity=[0, 0, 0], wind_velocity=args.wind,
    ).to(device)
    print(f"  Simulator: {args.n_sim} pts, KNN={args.k_neighbors}, "
          f"substeps={args.n_step}, wind={args.wind}")

    # ── Renderer ──
    renderer = GaussianRenderer(
        image_height=args.render_size, image_width=args.render_size, fov=40
    ).to(device)
    camera = renderer.get_camera(azimuth=0, elevation=14, radius=2.0,
                                 target=xyz.mean(0).detach())

    # ── SDS guidance ──
    print("Loading SDS guidance...")
    guidance = VideoSDSGuidance(guidance_scale=args.guidance_scale)
    text_emb = guidance.encode_prompt(args.prompt)
    print(f"  Prompt: '{args.prompt}'")

    # ── Structure loss ──
    struct_loss_fn = StructureLoss(
        lambda_anchor=args.lambda_anchor,
        lambda_branch=args.lambda_branch,
        lambda_consistency=args.lambda_consistency,
        lambda_ordering=args.lambda_ordering,
        lambda_attach=args.lambda_attach,
        lambda_smooth=args.lambda_smooth,
    )

    # ── Optimizer ──
    optimizer = torch.optim.Adam(
        list(material_net.parameters()) + list(structure_extractor.parameters()),
        lr=args.lr,
    )

    # ── Training loop (OmniPhysGS-style: epochs × internal_epochs) ──
    print(f"\nStarting training: {args.epochs} epochs × {args.internal_epochs} internal")
    total_steps = 0
    for epoch in range(args.epochs):
        material_net.clear_cache()

        for internal in tqdm(range(args.internal_epochs), desc=f"Epoch {epoch}", leave=False):
            optimizer.zero_grad()

            # Forward: network → physics params
            pred_params = material_net(xyz_sim, gaussian_features.detach(), structure_features.detach())

            # Build simulator physics_params dict
            k = pred_params['k']
            damp_expanded = pred_params['damp'].unsqueeze(1).expand(-1, args.k_neighbors)
            physics_params = {
                'k': k,
                'm': pred_params['mass'],
                'damp': damp_expanded,
                'drag': pred_params['drag'],
                'init_velocity': torch.zeros(1, 3, device=device),
            }

            # Simulate
            traj = sim(physics_params, n_frames=args.n_frames, part_labels=part_labels)

            if torch.isnan(traj).any():
                tqdm.write(f"  [Epoch {epoch}, internal {internal}] NaN trajectory, skip")
                continue

            # Render video
            video = renderer.render_trajectory(traj, scales, rots, opacities, colors, camera=camera)

            # SDS loss
            video_hwc = video.permute(0, 2, 3, 1).contiguous()
            loss_sds = guidance.compute_sds_loss(video_hwc, text_emb)

            # Structure loss
            loss_struct, struct_dict = struct_loss_fn(
                traj, pred_params, part_labels, sim.knn_index
            )

            # Motion encouragement (prevent degenerate zero-motion)
            loss_motion = -0.001 * (traj[-1] - traj[0]).pow(2).mean()

            loss = args.sds_weight * loss_sds + loss_struct + loss_motion

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(material_net.parameters(), max_norm=1.0)

            if torch.isnan(grad_norm):
                tqdm.write(f"  [Epoch {epoch}, internal {internal}] NaN grad, skip")
                optimizer.zero_grad()
                continue

            optimizer.step()
            total_steps += 1

            if (internal + 1) % 10 == 0:
                disp = (traj[-1] - traj[0]).norm(dim=1).mean().item()
                tqdm.write(
                    f"  [{epoch}:{internal+1}] loss={loss.item():.2f} "
                    f"sds={loss_sds.item():.1f} struct={loss_struct.item():.3f} "
                    f"k=[{k.min().item():.1f},{k.max().item():.1f}] "
                    f"disp={disp:.5f} gnorm={grad_norm.item():.3f}"
                )

        # Save checkpoint + video per epoch
        save_video(video.detach(), output_dir / f"epoch_{epoch:03d}.mp4")
        ckpt = {
            'epoch': epoch,
            'material_net': material_net.state_dict(),
            'structure_extractor': structure_extractor.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
        }
        torch.save(ckpt, output_dir / f"ckpt_epoch_{epoch:03d}.pt")
        print(f"  Epoch {epoch} saved.")

    # ── Final save ──
    print("\nSaving final results...")
    with torch.no_grad():
        final_params = material_net(xyz_sim, gaussian_features, structure_features)
    result = {
        'k': final_params['k'].cpu(),
        'damp': final_params['damp'].cpu(),
        'drag': final_params['drag'].cpu(),
        'mass': final_params['mass'].cpu(),
        'part_labels': part_labels.labels.cpu(),
        'sample_idx': idx.cpu(),
        'args': vars(args),
    }
    torch.save(result, output_dir / "final_params.pt")
    torch.save(material_net.state_dict(), output_dir / "material_net.pt")
    save_video(video.detach(), output_dir / "final_sim.mp4")

    print(f"Done. Results in {output_dir}")
    print(f"  k: [{result['k'].min():.1f}, {result['k'].max():.1f}]")
    print(f"  Parts: trunk={part_labels.trunk_mask.sum()}, "
          f"branch={part_labels.branch_mask.sum()}, leaf={part_labels.leaf_mask.sum()}")


if __name__ == '__main__':
    main()
