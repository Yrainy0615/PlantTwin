"""
Batch-preprocess DINO features for all plant 3DGS samples.

For each plant:
  1. Load canonical_frame.png → DINOv2-S → patch features [Hf, Wf, 384]
  2. Load gaussian.ply → project Gaussian centers to image via canonical camera
  3. Bilinearly sample DINO features at projected positions → per-Gaussian DINO [N, 384]
  4. KMeans cluster into 3 (trunk/branch/leaf) with height-ordering
  5. Save dino_labels.pt

Usage:
    python scripts/precompute_dino.py --data_dir data/plants_3dgs
    python scripts/precompute_dino.py --data_dir data/plants_3dgs --device cuda:1
"""
import argparse
import math
import sys
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from pathlib import Path
from sklearn.cluster import KMeans

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.pretrain.train_part_aware_pretrain import load_gaussian_ply


def load_dino(model_name="dinov2_vits14", device="cuda"):
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract_dino_features(dino, img_pil, device):
    patch_size = 14
    W_raw, H_raw = img_pil.size
    H = (H_raw // patch_size) * patch_size
    W = (W_raw // patch_size) * patch_size
    img_resized = img_pil.resize((W, H), Image.LANCZOS)

    img_t = torch.from_numpy(np.array(img_resized)).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_t = ((img_t - mean) / std).unsqueeze(0).to(device)

    with torch.no_grad():
        out = dino.forward_features(img_t)
        patch_tokens = out["x_norm_patchtokens"]  # [1, Hf*Wf, 384]

    H_feat = H // patch_size
    W_feat = W // patch_size
    feat_map = patch_tokens[0].reshape(H_feat, W_feat, -1)  # [Hf, Wf, 384]
    return feat_map, H, W


def build_camera(target, device, azimuth=0.0, elevation=14.0, radius=2.0, fov=40.0):
    import utils3d
    fov_rad = math.radians(fov)
    yaw = math.radians(azimuth)
    pitch = math.radians(elevation)

    orig = torch.tensor([
        math.sin(yaw) * math.cos(pitch),
        math.cos(yaw) * math.cos(pitch),
        math.sin(pitch),
    ], device=device, dtype=torch.float32) * radius + target

    extr = utils3d.torch.extrinsics_look_at(
        orig, target, torch.tensor([0., 0., 1.], device=device)
    )

    fov_t = torch.tensor(fov_rad, device=device)
    intr = utils3d.torch.intrinsics_from_fov_xy(fov_t, fov_t)
    fx, fy = intr[0, 0], intr[1, 1]

    perspective = torch.zeros(4, 4, device=device)
    cx, cy = intr[0, 2], intr[1, 2]
    perspective[0, 0] = 2 * fx
    perspective[1, 1] = 2 * fy
    perspective[0, 2] = 2 * cx - 1
    perspective[1, 2] = -2 * cy + 1
    znear, zfar = 0.01, 100.0
    perspective[2, 2] = zfar / (zfar - znear)
    perspective[2, 3] = znear * zfar / (znear - zfar)
    perspective[3, 2] = 1.0

    return extr, perspective  # both 4x4, W2C and projection


def project_gaussians_to_image(xyz, extr, perspective, H, W):
    """
    Project 3D Gaussian centers to 2D image coordinates.
    Returns: uv [N, 2] in pixel space, valid_mask [N] bool.
    """
    N = xyz.shape[0]
    ones = torch.ones(N, 1, device=xyz.device)
    xyz_h = torch.cat([xyz, ones], dim=1)  # [N, 4]

    xyz_cam = (extr @ xyz_h.T).T  # [N, 4]
    xyz_clip = (perspective @ xyz_cam.T).T  # [N, 4]

    w = xyz_clip[:, 3:4].clamp(min=1e-6)
    ndc = xyz_clip[:, :3] / w  # [N, 3], NDC in [-1, 1]

    u = (ndc[:, 0] + 1) * 0.5 * W  # pixel x
    v = (1 - ndc[:, 1]) * 0.5 * H  # pixel y (flip y)

    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (xyz_cam[:, 2] > 0)

    return torch.stack([u, v], dim=1), valid


def sample_features_at_uv(feat_map, uv, H_img, W_img):
    """
    Bilinearly sample feature map at pixel coordinates.
    feat_map: [Hf, Wf, D]
    uv: [N, 2] in pixel space of [H_img, W_img]
    Returns: [N, D]
    """
    Hf, Wf, D = feat_map.shape
    grid_x = uv[:, 0] / W_img * 2 - 1  # normalize to [-1, 1]
    grid_y = uv[:, 1] / H_img * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=1).reshape(1, 1, -1, 2)  # [1, 1, N, 2]

    feat_map_4d = feat_map.permute(2, 0, 1).unsqueeze(0)  # [1, D, Hf, Wf]
    sampled = F.grid_sample(
        feat_map_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=False
    )  # [1, D, 1, N]
    return sampled[0, :, 0, :].T  # [N, D]


def assign_part_labels(dino_feats, xyz, valid_mask, n_clusters=3):
    """
    KMeans cluster DINO features, then assign semantic meaning by height ordering.
    Returns: labels [N] with 0=trunk, 1=branch, 2=leaf
    """
    valid_feats = dino_feats[valid_mask].cpu().numpy()
    valid_xyz = xyz[valid_mask].cpu().numpy()

    if len(valid_feats) < n_clusters:
        return torch.zeros(len(xyz), dtype=torch.long, device=xyz.device)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(valid_feats)

    cluster_heights = []
    for c in range(n_clusters):
        mask = cluster_ids == c
        if mask.sum() > 0:
            cluster_heights.append(valid_xyz[mask, 2].mean())
        else:
            cluster_heights.append(0.0)

    # Sort clusters by height: lowest = trunk (0), middle = branch (1), highest = leaf (2)
    height_order = np.argsort(cluster_heights)
    remap = np.zeros(n_clusters, dtype=np.int64)
    for rank, cluster_id in enumerate(height_order):
        remap[cluster_id] = rank

    labels = torch.zeros(len(xyz), dtype=torch.long, device=xyz.device)
    labels[valid_mask] = torch.from_numpy(remap[cluster_ids]).long().to(xyz.device)
    # Invalid (back-facing) Gaussians: assign by nearest valid Gaussian
    if (~valid_mask).any() and valid_mask.any():
        from torch import cdist
        dists = cdist(xyz[~valid_mask].unsqueeze(0), xyz[valid_mask].unsqueeze(0))[0]
        nearest = dists.argmin(dim=1)
        labels[~valid_mask] = labels[valid_mask][nearest]

    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/plants_3dgs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dino_model", type=str, default="dinov2_vits14")
    parser.add_argument("--n_clusters", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Overwrite existing")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = torch.device(args.device)
    plant_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    print(f"Found {len(plant_dirs)} plant directories")
    print(f"Loading DINOv2 ({args.dino_model})...")
    dino = load_dino(args.dino_model, device)

    done, skipped, failed = 0, 0, 0
    for i, plant_dir in enumerate(plant_dirs):
        name = plant_dir.name
        out_path = plant_dir / "dino_labels.pt"

        if out_path.exists() and not args.force:
            skipped += 1
            continue

        canonical = plant_dir / "canonical_frame.png"
        ply_path = plant_dir / "gaussian.ply"

        if not canonical.exists() or not ply_path.exists():
            print(f"  [{i+1}/{len(plant_dirs)}] SKIP {name}: missing files")
            failed += 1
            continue

        try:
            img = Image.open(canonical).convert("RGB")
            feat_map, H_img, W_img = extract_dino_features(dino, img, device)

            xyz, _, _, _, _, _ = load_gaussian_ply(str(ply_path))
            xyz = xyz.to(device)

            target = xyz.mean(0)
            extr, perspective = build_camera(target, device)
            uv, valid = project_gaussians_to_image(xyz, extr, perspective, H_img, W_img)

            per_gauss_dino = torch.zeros(xyz.shape[0], feat_map.shape[2], device=device)
            if valid.any():
                per_gauss_dino[valid] = sample_features_at_uv(
                    feat_map, uv[valid], H_img, W_img
                )

            labels = assign_part_labels(per_gauss_dino, xyz, valid, args.n_clusters)

            torch.save({
                'labels': labels.cpu(),
                'n_valid': valid.sum().item(),
                'n_total': xyz.shape[0],
                'label_names': ['trunk', 'branch', 'leaf'],
            }, str(out_path))

            counts = [(labels == c).sum().item() for c in range(args.n_clusters)]
            print(f"  [{i+1}/{len(plant_dirs)}] {name}: "
                  f"trunk={counts[0]}, branch={counts[1]}, leaf={counts[2]}, "
                  f"valid={valid.sum().item()}/{xyz.shape[0]}")
            done += 1

        except Exception as e:
            print(f"  [{i+1}/{len(plant_dirs)}] FAIL {name}: {e}")
            failed += 1

    print(f"\nDone: {done}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
