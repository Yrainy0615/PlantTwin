"""Video losses for per-scene plant physics optimization.

Inputs are PyTorch tensors of shape [T, 3, H, W] (rendered) and [T, 3, H, W]
(target). Masks and optical flow targets, when present, are also [T, 1, H, W]
and [T-1, 2, H, W] respectively. Targets are expected to be precomputed offline
(SAM masks, RAFT flow) since both detectors are non-differentiable.

Optical-flow on rendered output: a simple finite-difference proxy is used for v0
so the loss can be applied without integrating a differentiable flow network.
This is intentionally crude — it just captures pixel-velocity statistics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rgb_l2(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (rendered - target).pow(2).mean()


def rgb_charbonnier(rendered: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Smooth-L1-ish robust pixel loss."""
    d = rendered - target
    return torch.sqrt(d * d + eps * eps).mean()


def alpha_mask_loss(
    rendered_alpha: torch.Tensor,    # [T, 1, H, W] or [T, H, W]
    target_mask: torch.Tensor,        # same as above, in [0, 1]
) -> torch.Tensor:
    """Soft mask consistency: 1 - IoU + per-pixel BCE for stability."""
    if rendered_alpha.dim() == 3:
        rendered_alpha = rendered_alpha.unsqueeze(1)
    if target_mask.dim() == 3:
        target_mask = target_mask.unsqueeze(1)
    intersection = (rendered_alpha * target_mask).flatten(1).sum(-1)
    union = (rendered_alpha + target_mask - rendered_alpha * target_mask).flatten(1).sum(-1)
    iou = (intersection + 1e-6) / (union + 1e-6)
    bce = F.binary_cross_entropy(
        rendered_alpha.clamp(1e-6, 1 - 1e-6),
        target_mask.clamp(0.0, 1.0),
        reduction='mean',
    )
    return (1.0 - iou).mean() + 0.1 * bce


def temporal_smoothness(rendered: torch.Tensor) -> torch.Tensor:
    """Penalize jumpy rendered frames."""
    if rendered.shape[0] < 2:
        return rendered.new_zeros(())
    return (rendered[1:] - rendered[:-1]).pow(2).mean()


def flow_consistency(
    rendered: torch.Tensor,           # [T, 3, H, W]
    target_flow: torch.Tensor,        # [T-1, 2, H, W] precomputed (e.g. RAFT)
) -> torch.Tensor:
    """Coarse motion prior: match the magnitude of finite-difference pixel flow."""
    if rendered.shape[0] < 2 or target_flow is None:
        return rendered.new_zeros(())
    rendered_dt = (rendered[1:] - rendered[:-1]).mean(dim=1, keepdim=True)   # [T-1, 1, H, W]
    target_mag = target_flow.norm(dim=1, keepdim=True)                         # [T-1, 1, H, W]
    return (rendered_dt.abs() - target_mag).pow(2).mean()


class VideoLossBundle(torch.nn.Module):
    """Convenience aggregator: returns a dict and the weighted sum."""
    def __init__(
        self,
        w_rgb: float = 1.0,
        w_mask: float = 0.1,
        w_flow: float = 0.0,
        w_temporal: float = 0.01,
        robust_rgb: bool = False,
    ):
        super().__init__()
        self.w_rgb = w_rgb
        self.w_mask = w_mask
        self.w_flow = w_flow
        self.w_temporal = w_temporal
        self.robust_rgb = robust_rgb

    def forward(
        self,
        rendered: torch.Tensor,                          # [T, 3, H, W]
        target: torch.Tensor,                            # [T, 3, H, W]
        rendered_alpha: torch.Tensor | None = None,     # [T, 1, H, W]
        target_mask: torch.Tensor | None = None,         # [T, 1, H, W]
        target_flow: torch.Tensor | None = None,         # [T-1, 2, H, W]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        parts: dict[str, torch.Tensor] = {}
        if self.robust_rgb:
            parts['rgb'] = rgb_charbonnier(rendered, target)
        else:
            parts['rgb'] = rgb_l2(rendered, target)

        if rendered_alpha is not None and target_mask is not None and self.w_mask > 0:
            parts['mask'] = alpha_mask_loss(rendered_alpha, target_mask)

        if target_flow is not None and self.w_flow > 0:
            parts['flow'] = flow_consistency(rendered, target_flow)

        if self.w_temporal > 0:
            parts['temporal'] = temporal_smoothness(rendered)

        total = self.w_rgb * parts.get('rgb', rendered.new_zeros(()))
        if 'mask' in parts: total = total + self.w_mask * parts['mask']
        if 'flow' in parts: total = total + self.w_flow * parts['flow']
        if 'temporal' in parts: total = total + self.w_temporal * parts['temporal']

        return total, parts
