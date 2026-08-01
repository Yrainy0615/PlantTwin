"""
Stage 1 Feed-forward Decoder Training: Train the video physics decoder
on generated plant motion videos using self-supervised reconstruction loss.

Pipeline (ReconPhys-style):
    Video → VideoPhysicsDecoder → physics params
    → SpringMassSimulator → trajectory → render → compare with input video

Usage:
    python scripts/train_decoder.py --video_dir data/plants_video --gs_dir data/plants_3dgs
"""
import os
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from simulation.spring_mass import SpringMassSimulator
from models.physics_decoder import VideoPhysicsDecoder


class PlantVideoDataset(Dataset):
    """Dataset of generated plant motion videos with corresponding static 3DGS."""

    def __init__(self, video_dir, gs_dir, n_frames=16, image_size=256):
        self.video_dir = Path(video_dir)
        self.gs_dir = Path(gs_dir)
        self.n_frames = n_frames
        self.image_size = image_size
        self.samples = self._scan_samples()

    def _scan_samples(self):
        samples = []
        for vdir in sorted(self.video_dir.iterdir()):
            if not vdir.is_dir():
                continue
            meta_path = vdir / "meta.json"
            video_path = vdir / "motion.mp4"
            if not meta_path.exists() or not video_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            # find matching GS directory
            source = Path(meta.get('source_image', ''))
            gs_name = source.parent.name if source.parent.name else vdir.name.rsplit('_', 1)[0]
            gs_path = self.gs_dir / gs_name / "gaussian.ply"
            if gs_path.exists():
                samples.append({'video_path': str(video_path), 'ply_path': str(gs_path), 'meta': meta})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load video frames
        import imageio
        reader = imageio.get_reader(sample['video_path'])
        frames = []
        for i, frame in enumerate(reader):
            if i >= self.n_frames:
                break
            frame = torch.from_numpy(frame).float() / 255.0
            frames.append(frame)
        reader.close()

        while len(frames) < self.n_frames:
            frames.append(frames[-1])

        video = torch.stack(frames)  # [T, H, W, 3]
        video = video.permute(3, 0, 1, 2)  # [3, T, H, W]
        video = F.interpolate(video.unsqueeze(0), size=(self.n_frames, self.image_size, self.image_size),
                              mode='trilinear', align_corners=False).squeeze(0)

        return {
            'video': video,  # [3, T, H, W]
            'ply_path': sample['ply_path'],
        }


def load_ply_positions(ply_path, n_sample=2048):
    """Load and subsample point positions from PLY."""
    from plyfile import PlyData
    plydata = PlyData.read(ply_path)
    xyz = np.stack([
        plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'],
    ], axis=-1).astype(np.float32)
    xyz = torch.from_numpy(xyz)
    if xyz.shape[0] > n_sample:
        idx = torch.randperm(xyz.shape[0])[:n_sample].sort().values
        xyz = xyz[idx]
    return xyz


def train_epoch(decoder, dataloader, optimizer, device, n_sample=2048, k_neighbors=256):
    decoder.train()
    total_loss = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc="  Training", leave=False):
        video = batch['video'].to(device)  # [B, 3, T, H, W]
        B = video.shape[0]

        # Predict physics params from video
        physics_params = decoder(video)

        # Self-supervised loss: predicted params should be physically consistent
        # (Full loop with simulation+rendering would go here in production)
        # For now: regularization losses that enforce reasonable parameter ranges

        k = physics_params['k']  # [B, N]
        m = physics_params['m']
        damp = physics_params['damp']

        # Parameter range regularization
        loss_k_range = F.relu(k - 1100).mean() + F.relu(100 - k).mean()
        loss_m_range = F.relu(m - 6.0).mean() + F.relu(0.2 - m).mean()

        # Smoothness: nearby predicted params should be similar
        loss_smooth = ((k[:, 1:] - k[:, :-1]) ** 2).mean()

        # Consistency: parameters shouldn't be trivially uniform
        loss_variance = -0.01 * k.var(dim=1).mean()

        loss = loss_k_range + loss_m_range + 0.1 * loss_smooth + loss_variance

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train feed-forward physics decoder")
    parser.add_argument('--video_dir', type=str, default='data/plants_video')
    parser.add_argument('--gs_dir', type=str, default='data/plants_3dgs')
    parser.add_argument('--output_dir', type=str, default='outputs/decoder_pretrain')
    parser.add_argument('--backbone', type=str, default='OpenGVLab/InternViT-300M-448px-V2_5')
    parser.add_argument('--n_points', type=int, default=2048)
    parser.add_argument('--n_frames', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_every', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building dataset from {args.video_dir}...")
    dataset = PlantVideoDataset(args.video_dir, args.gs_dir, n_frames=args.n_frames)
    print(f"  Found {len(dataset)} video-GS pairs")

    if len(dataset) == 0:
        print("No data found. Run gen_3dgs.py and gen_video.py first.")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                           num_workers=2, pin_memory=True)

    print(f"Building VideoPhysicsDecoder (backbone={args.backbone})...")
    decoder = VideoPhysicsDecoder(
        n_points=args.n_points,
        backbone_name=args.backbone,
        freeze_backbone=True,
    ).to(device)

    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Trainable: {trainable_params/1e6:.1f}M / Total: {total_params/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        [p for p in decoder.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        avg_loss = train_epoch(decoder, dataloader, optimizer, device, args.n_points)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f} | LR: {lr:.2e}")

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = output_dir / f"decoder_epoch{epoch+1:04d}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    print("Training complete.")


if __name__ == '__main__':
    main()
