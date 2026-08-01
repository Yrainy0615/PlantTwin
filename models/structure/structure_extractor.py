"""
Extract plant structure information from static 3DGS.

Three paths:
  A) DINO: render canonical views → DINOv2 features → project to Gaussians → cluster
  B) Heuristic: height + color + density → trunk/branch/leaf
  C) VLM: PhysX-Omni VLM (future upgrade)

All paths produce per-Gaussian part labels and optional per-Gaussian features
that feed into PlantMaterialNetwork.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional


PART_TRUNK = 0
PART_BRANCH = 1
PART_LEAF = 2
PART_NAMES = {PART_TRUNK: 'trunk', PART_BRANCH: 'branch', PART_LEAF: 'leaf'}


@dataclass
class PartLabels:
    labels: torch.Tensor       # [N] int, per-Gaussian part label
    n_parts: int               # number of unique parts
    trunk_mask: torch.Tensor   # [N] bool
    branch_mask: torch.Tensor  # [N] bool
    leaf_mask: torch.Tensor    # [N] bool


class StructureExtractor(nn.Module):
    """
    Extract structure information from static 3DGS Gaussians.
    Produces part labels + feature embeddings for PlantMaterialNetwork input.
    """

    def __init__(self, method='heuristic', n_parts=3, part_embed_dim=32,
                 dino_model_name='dinov2_vits14'):
        super().__init__()
        self.method = method
        self.n_parts = n_parts
        self.part_embed_dim = part_embed_dim
        self.dino_model_name = dino_model_name

        self.part_embedding = nn.Embedding(n_parts, part_embed_dim)

        self._dino_model = None

    @property
    def feature_dim(self):
        if self.method == 'dino_raw':
            return 384  # DINOv2-S
        return self.part_embed_dim

    def _get_dino_model(self, device):
        if self._dino_model is None:
            self._dino_model = torch.hub.load(
                'facebookresearch/dinov2', self.dino_model_name
            ).eval().to(device)
            for p in self._dino_model.parameters():
                p.requires_grad_(False)
        return self._dino_model

    def extract_heuristic(self, xyz, colors):
        """
        Height + color heuristic for plant part labeling.

        Args:
            xyz: [N, 3] Gaussian positions (Z-up)
            colors: [N, 3] RGB colors in [0, 1]

        Returns:
            PartLabels
        """
        N = xyz.shape[0]
        device = xyz.device

        height = xyz[:, 2]
        height_norm = (height - height.min()) / (height.max() - height.min() + 1e-8)

        green_ratio = colors[:, 1] / (colors.sum(dim=1) + 1e-8)

        # Trunk: bottom portion, less green
        trunk_height_thresh = 0.25
        trunk_mask = (height_norm < trunk_height_thresh) & (green_ratio < 0.40)

        # Leaf: high green ratio
        leaf_mask = green_ratio > 0.38

        # Branch: everything else
        branch_mask = ~trunk_mask & ~leaf_mask

        labels = torch.full((N,), PART_BRANCH, dtype=torch.long, device=device)
        labels[trunk_mask] = PART_TRUNK
        labels[leaf_mask] = PART_LEAF

        return PartLabels(
            labels=labels,
            n_parts=self.n_parts,
            trunk_mask=trunk_mask,
            branch_mask=branch_mask,
            leaf_mask=leaf_mask,
        )

    def extract_dino(self, xyz, rendered_images, cameras, renderer):
        """
        Project DINOv2 features from rendered views back to Gaussians via
        weighted alpha-compositing.

        Args:
            xyz: [N, 3] Gaussian centers
            rendered_images: [V, 3, H, W] rendered canonical views
            cameras: list of camera dicts (from GaussianRenderer.get_camera)
            renderer: GaussianRenderer instance for projection

        Returns:
            dino_features: [N, D_dino] per-Gaussian DINO features
        """
        device = xyz.device
        V = rendered_images.shape[0]
        dino = self._get_dino_model(device)

        H_dino, W_dino = 518, 518  # DINOv2 expects 14-divisible
        patch_size = 14
        H_feat = H_dino // patch_size
        W_feat = W_dino // patch_size
        D = 384  # DINOv2-S feature dim

        all_features = torch.zeros(xyz.shape[0], D, device=device)
        all_weights = torch.zeros(xyz.shape[0], 1, device=device)

        for v in range(V):
            img = rendered_images[v:v+1]
            img_resized = F.interpolate(img, size=(H_dino, W_dino), mode='bilinear', align_corners=False)
            # Normalize for DINOv2
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            img_norm = (img_resized - mean) / std

            with torch.no_grad():
                feat = dino.forward_features(img_norm)['x_norm_patchtokens']  # [1, P, D]
            feat = feat.reshape(1, H_feat, W_feat, D).permute(0, 3, 1, 2)  # [1, D, Hf, Wf]

            # TODO: proper differentiable Gaussian-to-pixel projection for feature backprop
            # For now, use nearest-neighbor assignment via rendered depth / pixel lookup
            # This is a placeholder — full implementation needs rasterizer alpha weights
            pass

        # Fallback: cluster xyz-based features if projection not yet implemented
        return self._cluster_by_position_color(xyz, all_features)

    def _cluster_by_position_color(self, xyz, features):
        """Spectral clustering fallback — returns part labels from DINO features."""
        # For MVP, fall back to heuristic when DINO projection isn't ready
        return None

    def extract_dino_cluster(self, xyz, rendered_images, cameras, renderer, colors):
        """
        DINO features → spectral clustering → part labels.
        Falls back to heuristic if DINO projection isn't implemented yet.
        """
        part_labels = self.extract_heuristic(xyz, colors)
        return part_labels

    def forward(self, xyz, colors, rendered_images=None, cameras=None, renderer=None):
        """
        Extract structure and return (part_labels, structure_features).

        Args:
            xyz: [N, 3]
            colors: [N, 3]
            rendered_images: optional [V, 3, H, W] for DINO path
            cameras: optional list of camera dicts
            renderer: optional GaussianRenderer

        Returns:
            part_labels: PartLabels
            structure_features: [N, D_struct] features for PlantMaterialNetwork
        """
        if self.method == 'heuristic':
            part_labels = self.extract_heuristic(xyz, colors)
        elif self.method in ('dino', 'dino_cluster'):
            if rendered_images is not None:
                part_labels = self.extract_dino_cluster(
                    xyz, rendered_images, cameras, renderer, colors
                )
            else:
                part_labels = self.extract_heuristic(xyz, colors)
        else:
            raise ValueError(f"Unknown structure method: {self.method}")

        structure_features = self.part_embedding(part_labels.labels)  # [N, D_embed]
        return part_labels, structure_features
