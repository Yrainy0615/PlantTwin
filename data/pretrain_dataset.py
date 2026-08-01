"""
PyTorch Dataset for multi-plant pretrain.
Each item is one plant: loads PLY + pre-computed DINO labels + meta.
Subsamples to n_sim points for simulation.
"""
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class PlantPretrainDataset(Dataset):

    def __init__(self, data_dir, n_sim=2048, device='cpu', max_plants=None):
        self.data_dir = Path(data_dir)
        self.n_sim = n_sim
        self.device = device

        self.plant_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and (d / 'gaussian.ply').exists()
        ])

        # Filter to only plants with DINO labels
        self.plant_dirs = [
            d for d in self.plant_dirs if (d / 'dino_labels.pt').exists()
        ]

        if max_plants is not None:
            self.plant_dirs = self.plant_dirs[:max_plants]

        print(f"PlantPretrainDataset: {len(self.plant_dirs)} plants "
              f"(n_sim={n_sim})")

    def __len__(self):
        return len(self.plant_dirs)

    def __getitem__(self, idx):
        plant_dir = self.plant_dirs[idx]
        name = plant_dir.name

        # Load PLY
        xyz, scales, rots, opacities, colors, shs = _load_gaussian_ply(
            str(plant_dir / 'gaussian.ply')
        )

        # Load DINO labels
        dino_data = torch.load(str(plant_dir / 'dino_labels.pt'), weights_only=True)
        dino_labels = dino_data['labels']  # [N]

        # Load meta for prompt
        meta_path = plant_dir / 'meta.json'
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            prompt = meta.get('prompt', f'a plant swaying in the wind')
        else:
            prompt = 'a plant swaying in the wind'

        # Full set for rendering, subsample for simulation
        N = xyz.shape[0]
        if N > self.n_sim:
            idx_sub = torch.randperm(N)[:self.n_sim].sort().values
        else:
            idx_sub = torch.arange(N)

        xyz_sim = xyz[idx_sub]
        scales_sim = scales[idx_sub]
        rots_sim = rots[idx_sub]
        opacities_sim = opacities[idx_sub]
        shs_sim = shs[idx_sub]
        dino_labels_sim = dino_labels[idx_sub]

        cov_flat = _compute_cov_flat(scales_sim, rots_sim)
        gaussian_features = _build_gaussian_features(xyz_sim, shs_sim, cov_flat, opacities_sim)

        return {
            # Simulation subset (n_sim points)
            'xyz_sim': xyz_sim,
            'gaussian_features': gaussian_features,
            'dino_labels': dino_labels_sim,
            # Full set (for rendering)
            'xyz_full': xyz,
            'scales_full': scales,
            'rots_full': rots,
            'opacities_full': opacities,
            'colors_full': colors,
            'shs_full': shs,  # [N, 48] for sh_degree=3 rendering
            'prompt': prompt,
            'name': name,
        }


def _load_gaussian_ply(ply_path):
    from plyfile import PlyData
    plydata = PlyData.read(str(ply_path))
    v = plydata['vertex']
    xyz = torch.tensor(np.stack([v['x'], v['y'], v['z']], -1), dtype=torch.float32)
    scales = torch.exp(torch.tensor(
        np.stack([v['scale_0'], v['scale_1'], v['scale_2']], -1), dtype=torch.float32
    ))
    rots = torch.tensor(
        np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], -1), dtype=torch.float32
    )
    rots = rots / (rots.norm(dim=1, keepdim=True) + 1e-8)
    opacities = torch.sigmoid(torch.tensor(v['opacity'][:, None], dtype=torch.float32))

    sh_dc = torch.tensor(
        np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], -1), dtype=torch.float32
    )
    # Standard 3DGS DC→RGB conversion: color = SH_C0 * sh_dc + 0.5
    SH_C0 = 0.28209479177387814
    colors = (SH_C0 * sh_dc + 0.5).clamp(0.0, 1.0)

    sh_keys = [k for k in v.data.dtype.names if k.startswith('f_rest_')]
    if sh_keys:
        sh_rest = torch.tensor(np.stack([v[k] for k in sh_keys], -1), dtype=torch.float32)
        shs = torch.cat([sh_dc, sh_rest], dim=-1)
    else:
        shs = sh_dc

    return xyz, scales, rots, opacities, colors, shs


def _compute_cov_flat(scales, rots):
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
    cov_flat = torch.stack([
        cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
        cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2],
    ], dim=-1)
    return cov_flat


def _build_gaussian_features(xyz, shs, cov_flat, opacity, normalize=True):
    n = xyz.shape[0]
    if normalize:
        def norm_feat(x):
            x_flat = x.reshape(n, -1)
            return (x_flat - x_flat.mean(0, keepdim=True)) / (x_flat.std(0, keepdim=True) + 1e-8)
        features = torch.cat([xyz, norm_feat(shs), norm_feat(cov_flat), norm_feat(opacity)], dim=1)
    else:
        features = torch.cat([xyz, shs.reshape(n, -1), cov_flat.reshape(n, -1), opacity.reshape(n, -1)], dim=1)
    return features
