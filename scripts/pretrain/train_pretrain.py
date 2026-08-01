"""
Shared-backbone pretrain: train SpringParamDecoder across all plant samples.

Two-phase design:
  1. Encode phase (once at startup): PlantFeatureEncoder extracts per-Gaussian
     latent [N, H] for each plant from static canonical GS. Cached and frozen.
  2. Train phase (every step): SpringParamDecoder predicts per-spring {k, damp}
     from cached latent → simulate → render → SDS loss.

Only the decoder (EdgeMLP + head_drag + head_mass) is trained.

Usage:
    python scripts/train_pretrain.py --data_dir data/plants_3dgs --epochs 50
    python scripts/train_pretrain.py --data_dir data/plants_3dgs --smoke_test
"""
import os
import sys
import random
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import imageio
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.pretrain_dataset import PlantPretrainDataset
from simulation.spring_mass import SpringMassSimulator
from models.renderer import GaussianRenderer
from models.physics_decoder.plant_material_network import PlantMaterialNetwork
from optimization.sds_guidance import VideoSDSGuidance
from optimization.structure_loss import StructureLoss


def save_video(frames_tensor, path, fps=8):
    frames = (frames_tensor.permute(0, 2, 3, 1).detach().cpu().numpy() * 255
              ).clip(0, 255).astype(np.uint8)
    imageio.mimsave(str(path), list(frames), fps=fps)


def main():
    parser = argparse.ArgumentParser(description="Shared backbone plant physics pretrain")
    parser.add_argument('--data_dir', type=str, default='data/plants_3dgs')
    parser.add_argument('--output_dir', type=str, default='outputs/pretrain')

    # Encoder (frozen after encode phase)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--num_groups', type=int, default=100)
    parser.add_argument('--group_size', type=int, default=32)
    parser.add_argument('--part_embed_dim', type=int, default=32)
    parser.add_argument('--n_parts', type=int, default=3)

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
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--steps_per_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--guidance_scale', type=float, default=50.0)
    parser.add_argument('--sds_weight', type=float, default=0.01)

    # Structure loss (disabled by default for first version)
    parser.add_argument('--use_struct_loss', action='store_true',
                        help='Enable structure loss (disabled by default)')
    parser.add_argument('--lambda_anchor', type=float, default=10.0)
    parser.add_argument('--lambda_branch', type=float, default=1.0)
    parser.add_argument('--lambda_consistency', type=float, default=0.1)
    parser.add_argument('--lambda_ordering', type=float, default=0.5)
    parser.add_argument('--lambda_attach', type=float, default=1.0)
    parser.add_argument('--lambda_smooth', type=float, default=0.01)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--smoke_test', action='store_true',
                        help='Quick test: 2 steps, 2 epochs, 2 plants')
    parser.add_argument('--max_plants', type=int, default=None,
                        help='Limit dataset to first N plants')
    parser.add_argument('--ckpt', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--video_interval', type=int, default=5)
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 2
        args.steps_per_epoch = 2
        args.n_frames = 8
        if args.max_plants is None:
            args.max_plants = 2

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'videos').mkdir(exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ── Dataset ──
    print("Loading dataset...")
    dataset = PlantPretrainDataset(args.data_dir, n_sim=args.n_sim,
                                   max_plants=args.max_plants)

    # ── Network ──
    sample = dataset[0]
    gs_feat_dim = sample['gaussian_features'].shape[1]
    print(f"Gaussian feature dim: {gs_feat_dim}")

    part_embedding = nn.Embedding(args.n_parts, args.part_embed_dim).to(device)

    material_net = PlantMaterialNetwork(
        gaussian_feat_dim=gs_feat_dim,
        structure_feat_dim=args.part_embed_dim,
        num_groups=args.num_groups,
        group_size=args.group_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
    ).to(device)

    encoder_params = sum(p.numel() for p in material_net.encoder.parameters())
    decoder_params = sum(p.numel() for p in material_net.decoder.parameters())
    print(f"Encoder: {encoder_params:,} params (frozen after encode)")
    print(f"Decoder: {decoder_params:,} params (trainable)")

    # ── Encode phase: extract & cache latent for all plants ──
    print("\n[Encode phase] Extracting per-Gaussian latent for all plants...")
    latent_cache = {}
    sim_cache = {}
    for i in tqdm(range(len(dataset)), desc="Encoding"):
        plant = dataset[i]
        xyz_sim = plant['xyz_sim'].to(device)
        gaussian_features = plant['gaussian_features'].to(device)
        dino_labels = plant['dino_labels'].to(device)

        struct_feat = part_embedding(dino_labels)
        latent = material_net.encode(xyz_sim, gaussian_features, struct_feat)

        sim = SpringMassSimulator(
            xyz_sim, k_neighbors=args.k_neighbors,
            dt=args.dt, n_step=args.n_step,
            damping=True, gravity=[0, 0, 0],
            wind_velocity=args.wind,
        ).to(device)

        latent_cache[i] = latent.detach()
        sim_cache[i] = {
            'knn_index': sim.knn_index.detach(),
            'origin_len': sim.origin_len.detach(),
        }

    print(f"  Cached latent for {len(latent_cache)} plants")

    # ── Renderer ──
    renderer = GaussianRenderer(
        image_height=args.render_size, image_width=args.render_size, fov=40,
        bg_color=[0.0, 0.0, 0.0],
    ).to(device)

    # ── SDS guidance ──
    print("Loading SDS guidance...")
    guidance = VideoSDSGuidance(guidance_scale=args.guidance_scale, device=str(device))

    # Pre-compute text embeddings
    print("Pre-computing text embeddings...")
    text_emb_cache = {}
    for i in range(len(dataset)):
        prompt = dataset[i]['prompt']
        if prompt not in text_emb_cache:
            text_emb_cache[prompt] = guidance.encode_prompt(
                f"{prompt}, gently swaying in the wind"
            )
    print(f"  Cached {len(text_emb_cache)} unique prompts")

    # ── Structure loss (optional) ──
    if args.use_struct_loss:
        struct_loss_fn = StructureLoss(
            lambda_anchor=args.lambda_anchor,
            lambda_branch=args.lambda_branch,
            lambda_consistency=args.lambda_consistency,
            lambda_ordering=args.lambda_ordering,
            lambda_attach=args.lambda_attach,
            lambda_smooth=args.lambda_smooth,
        )
        print("Structure loss: ENABLED")
    else:
        struct_loss_fn = None
        print("Structure loss: DISABLED (first version uses SDS only)")

    # ── Optimizer: only decoder + part_embedding ──
    optimizer = torch.optim.Adam(
        list(material_net.decoder.parameters()) + list(part_embedding.parameters()),
        lr=args.lr,
    )

    # ── Resume ──
    start_epoch = 0
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        material_net.decoder.load_state_dict(ckpt['decoder'])
        part_embedding.load_state_dict(ckpt['part_embedding'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    # ── Training loop ──
    print(f"\nTraining: {args.epochs} epochs x {args.steps_per_epoch} steps")
    print(f"  {len(dataset)} plants, n_sim={args.n_sim}, k_neighbors={args.k_neighbors}")
    print(f"  Only decoder is trained ({decoder_params:,} params)")

    log_file = open(output_dir / 'train_log.txt', 'a')

    for epoch in range(start_epoch, args.epochs):
        epoch_losses = []

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch}/{args.epochs}")
        for step in pbar:
            idx = random.randint(0, len(dataset) - 1)
            plant = dataset[idx]

            xyz_sim = plant['xyz_sim'].to(device)
            xyz_full = plant['xyz_full'].to(device)
            dino_labels = plant['dino_labels'].to(device)
            scales_full = plant['scales_full'].to(device)
            rots_full = plant['rots_full'].to(device)
            opacities_full = plant['opacities_full'].to(device)
            colors_full = plant['colors_full'].to(device)

            latent = latent_cache[idx]
            knn_index = sim_cache[idx]['knn_index']

            # Decode
            optimizer.zero_grad()
            params = material_net.decode(latent, knn_index)

            # Simulate on subset; interpolate to full set via xyz_all
            sim = SpringMassSimulator(
                xyz_sim, k_neighbors=args.k_neighbors,
                dt=args.dt, n_step=args.n_step,
                damping=True, gravity=[0, 0, 0],
                wind_velocity=args.wind,
            ).to(device)
            part_labels_obj = type('PartLabels', (), {
                'labels': dino_labels,
                'trunk_mask': dino_labels == 0,
                'branch_mask': dino_labels == 1,
                'leaf_mask': dino_labels == 2,
            })()
            traj_full = sim(params, n_frames=args.n_frames,
                            xyz_all=xyz_full,
                            part_labels=part_labels_obj)

            # Render with full Gaussian set
            camera = renderer.get_camera(
                azimuth=0, elevation=14, radius=2.0,
                target=xyz_full.mean(0).detach(),
            )
            video = renderer.render_trajectory(
                traj_full, scales_full, rots_full,
                opacities_full, colors_full, camera=camera,
            )

            # SDS loss
            text_emb = text_emb_cache[plant['prompt']]
            video_for_sds = video.permute(0, 2, 3, 1)
            loss_sds = guidance.compute_sds_loss(video_for_sds, text_emb)

            loss = args.sds_weight * loss_sds

            # Structure loss (optional) — runs on sim subset
            if struct_loss_fn is not None:
                # Re-run simulation on subset only for structure loss
                sim_sub = SpringMassSimulator(
                    xyz_sim, k_neighbors=args.k_neighbors,
                    dt=args.dt, n_step=args.n_step,
                    damping=True, gravity=[0, 0, 0],
                    wind_velocity=args.wind,
                ).to(device)
                traj_sim = sim_sub(params, n_frames=args.n_frames,
                                   part_labels=part_labels_obj)
                loss_struct, loss_dict = struct_loss_fn(
                    traj_sim, params, part_labels_obj, knn_index,
                )
                loss = loss + loss_struct
            loss.backward()

            for p in material_net.decoder.parameters():
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, 0.0, 0.0, 0.0)
            torch.nn.utils.clip_grad_norm_(material_net.decoder.parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            pbar.set_postfix(loss=f"{loss_val:.4f}", plant=plant['name'][:25])

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        log_line = f"epoch={epoch}, avg_loss={avg_loss:.6f}"
        print(f"  {log_line}")
        log_file.write(log_line + '\n')
        log_file.flush()

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            ckpt_path = output_dir / 'checkpoints' / f'epoch_{epoch:04d}.pt'
            torch.save({
                'decoder': material_net.decoder.state_dict(),
                'encoder': material_net.encoder.state_dict(),
                'part_embedding': part_embedding.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'config': vars(args),
            }, str(ckpt_path))
            print(f"  Saved checkpoint: {ckpt_path}")

        # Save sample video
        if (epoch + 1) % args.video_interval == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                sample_idx = 0
                plant = dataset[sample_idx]
                xyz_sim = plant['xyz_sim'].to(device)
                xyz_full = plant['xyz_full'].to(device)
                dino_labels = plant['dino_labels'].to(device)
                latent = latent_cache[sample_idx]
                knn_index = sim_cache[sample_idx]['knn_index']

                params = material_net.decode(latent, knn_index)
                sim = SpringMassSimulator(
                    xyz_sim, k_neighbors=args.k_neighbors,
                    dt=args.dt, n_step=args.n_step,
                    damping=True, gravity=[0, 0, 0],
                    wind_velocity=args.wind,
                ).to(device)
                pl_obj = type('PartLabels', (), {
                    'labels': dino_labels,
                    'trunk_mask': dino_labels == 0,
                    'branch_mask': dino_labels == 1,
                    'leaf_mask': dino_labels == 2,
                })()
                traj_full = sim(params, n_frames=args.n_frames,
                                xyz_all=xyz_full, part_labels=pl_obj)

                camera = renderer.get_camera(
                    azimuth=0, elevation=14, radius=2.0,
                    target=xyz_full.mean(0).detach(),
                )
                video = renderer.render_trajectory(
                    traj_full,
                    plant['scales_full'].to(device),
                    plant['rots_full'].to(device),
                    plant['opacities_full'].to(device),
                    plant['colors_full'].to(device),
                    camera=camera,
                )
                vid_path = output_dir / 'videos' / f'epoch_{epoch:04d}.mp4'
                save_video(video, vid_path)
                print(f"  Saved video: {vid_path}")

    log_file.close()
    print(f"\nDone. Output: {output_dir}")


if __name__ == '__main__':
    main()
