"""
PlantMaterialNetwork: predicts per-spring physics parameters from
pre-computed per-Gaussian latent features.

v2: Two-phase design:
  1. Encode phase (once, on static canonical GS): backbone → per-Gaussian latent [N, H]
  2. Predict phase (every training step): latent → EdgeMLP → per-spring {k, damp}

Backbone: OmniPhysGS KNNTransformer (FPS → GroupEncoder → Transformer).
Per-spring outputs: k [N, K], damp [N, K]
Per-particle outputs: drag [N], mass [N]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

import sys
import os
import importlib.util

_knn_path = os.path.join(
    os.path.dirname(__file__), '../../third_party/OmniPhysGS/src/physics_guided_network/knn_transformer.py'
)
_spec = importlib.util.spec_from_file_location("knn_transformer", os.path.abspath(_knn_path))
_knn_module = importlib.util.module_from_spec(_spec)
sys.modules.setdefault('src', type(sys)('src'))
sys.modules.setdefault('src.utils', type(sys)('src.utils'))
sys.modules.setdefault('src.utils.render_utils', type(sys)('src.utils.render_utils'))
if not hasattr(sys.modules['src.utils.render_utils'], 'particle_position_tensor_to_ply'):
    sys.modules['src.utils.render_utils'].particle_position_tensor_to_ply = lambda *a, **kw: None
_spec.loader.exec_module(_knn_module)

Grouper = _knn_module.Grouper
GroupEncoder = _knn_module.GroupEncoder
Block = _knn_module.Block


class PlantFeatureEncoder(nn.Module):
    """
    Backbone encoder: static Gaussian features → per-Gaussian latent.
    Run ONCE per plant on the canonical (undeformed) state, then cache.
    """

    def __init__(
        self,
        in_channels: int,
        num_groups: int = 100,
        group_size: int = 32,
        hidden_size: int = 128,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.grouper = Grouper(num_groups, group_size)
        self.group_encoder = GroupEncoder(in_channels, hidden_dim=hidden_size // 4, out_dim=hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_groups, hidden_size))
        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

    @torch.no_grad()
    def forward(self, pos: Tensor, features: Tensor) -> Tensor:
        """
        Args:
            pos: [N, 3] canonical Gaussian centers
            features: [N, D] cat(gaussian_features, structure_features)
        Returns:
            per_gaussian_latent: [N, H]
        """
        x = pos.unsqueeze(0)
        f = features.unsqueeze(0)
        neighbors, nearest_center_idx = self.grouper(x, f)

        encoded = self.group_encoder(neighbors)

        if len(self.blocks) > 0:
            encoded = encoded + self.pos_emb
        for block in self.blocks:
            encoded = block(encoded)

        group_feat = encoded.squeeze(0)  # [G, H]
        nearest_center_idx = nearest_center_idx.squeeze(0)  # [N]
        return group_feat[nearest_center_idx]  # [N, H]


class SpringParamDecoder(nn.Module):
    """
    Lightweight decoder: pre-computed latent → per-spring + per-particle params.
    This is the only trainable part during the pretrain loop.
    """

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2),
        )
        self.head_drag = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.head_mass = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, latent: Tensor, knn_index: Tensor) -> dict:
        """
        Args:
            latent: [N, H] pre-computed per-Gaussian latent (frozen)
            knn_index: [N, K] spring topology from simulator
        Returns:
            k: [N, K], damp: [N, K], drag: [N], mass: [N]
        """
        K = knn_index.shape[1]
        feat_i = latent.unsqueeze(1).expand(-1, K, -1)  # [N, K, H]
        feat_j = latent[knn_index]                        # [N, K, H]
        edge_feat = torch.cat([feat_i, feat_j], dim=-1)   # [N, K, 2H]

        spring_raw = self.edge_mlp(edge_feat)              # [N, K, 2]
        k = F.softplus(spring_raw[..., 0])
        damp = F.softplus(spring_raw[..., 1])

        drag = torch.sigmoid(self.head_drag(latent)).squeeze(-1)
        m = 0.1 + F.softplus(self.head_mass(latent)).squeeze(-1)

        return {'k': k, 'damp': damp, 'drag': drag, 'm': m}


class PlantMaterialNetwork(nn.Module):
    """
    Combined encoder + decoder for convenience.

    Two-phase usage:
      1. encode(): run once on canonical GS → per-Gaussian latent [N, H]
      2. decode(): every training step, latent + knn_index → spring params

    During training, only the decoder is optimized.
    """

    def __init__(
        self,
        gaussian_feat_dim: int,
        structure_feat_dim: int = 0,
        num_groups: int = 100,
        group_size: int = 32,
        hidden_size: int = 128,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        in_channels = gaussian_feat_dim + structure_feat_dim

        self.encoder = PlantFeatureEncoder(
            in_channels=in_channels,
            num_groups=num_groups,
            group_size=group_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.decoder = SpringParamDecoder(hidden_size=hidden_size)

    def encode(self, pos, gaussian_features, structure_features=None):
        """
        Extract per-Gaussian latent from static canonical GS. Run once.
        Returns: [N, H] latent (detached, no grad).
        """
        if structure_features is not None:
            features = torch.cat([gaussian_features, structure_features], dim=-1)
        else:
            features = gaussian_features
        return self.encoder(pos, features)

    def decode(self, latent, knn_index):
        """
        Predict per-spring params from cached latent. Run every training step.
        Returns: dict with k [N,K], damp [N,K], drag [N], mass [N]
        """
        return self.decoder(latent, knn_index)

    def forward(self, pos, gaussian_features, structure_features=None, knn_index=None):
        """Full forward for convenience (encode + decode)."""
        latent = self.encode(pos, gaussian_features, structure_features)
        if knn_index is None:
            K = 8
            knn_index = torch.randint(0, pos.shape[0], (pos.shape[0], K), device=pos.device)
        return self.decode(latent, knn_index)
