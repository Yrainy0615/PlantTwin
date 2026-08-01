"""Time-varying external contact force at specified nodes.

Use cases:
- During optimization: `ContactForceTrajectory` holds learnable per-frame 3D
  forces applied at fixed anchor node ids.
- For data generation / re-drive: pass a pre-computed force trajectory in.

The output is a `[T, N, 3]` tensor suitable for `ArticulatedChain.rollout`.

Two regularizers are provided that compose with the standard losses:
- temporal smoothness: penalizes |f_t - f_{t-1}|
- sparsity: penalizes |f_t| (encourages zero outside true contact frames)
"""

from __future__ import annotations

import torch
from torch import nn


class ContactForceTrajectory(nn.Module):
    """Learnable per-frame 3D forces applied at one or more fixed anchor nodes.

    Args:
        n_frames: T
        n_total_nodes: total number of branch nodes (so the output has shape [T, N, 3])
        anchor_node_ids: which node ids receive the force; the corresponding 3D
            force per frame is learned. Use `None` for the full per-frame force
            field (much larger — usually unnecessary).
    """
    def __init__(
        self,
        n_frames: int,
        n_total_nodes: int,
        anchor_node_ids: list[int] | torch.Tensor,
        init_scale: float = 0.0,
    ):
        super().__init__()
        self.n_frames = n_frames
        self.n_total_nodes = n_total_nodes
        anchor_node_ids = torch.as_tensor(anchor_node_ids, dtype=torch.long)
        self.register_buffer('anchor_node_ids', anchor_node_ids)
        n_anchors = anchor_node_ids.shape[0]
        # Learnable [T, N_anchors, 3]
        self.force = nn.Parameter(torch.randn(n_frames, n_anchors, 3) * init_scale)

    def forward(self) -> torch.Tensor:
        """Return a [T, N_total, 3] force trajectory (zero except at anchors)."""
        out = torch.zeros(
            self.n_frames, self.n_total_nodes, 3,
            device=self.force.device, dtype=self.force.dtype,
        )
        # scatter into the output at anchor node positions, per frame
        out[:, self.anchor_node_ids, :] = self.force
        return out

    def temporal_smoothness_loss(self) -> torch.Tensor:
        if self.n_frames < 2:
            return self.force.new_zeros(())
        return (self.force[1:] - self.force[:-1]).pow(2).mean()

    def sparsity_loss(self) -> torch.Tensor:
        return self.force.abs().mean()


def force_from_pixel_pin(
    pin_pixel_traj: torch.Tensor,        # [T, 2] (u, v) per frame, NaN where no contact
    pin_anchor_node: int,
    n_frames: int,
    n_total_nodes: int,
    init_magnitude: float = 0.1,
) -> ContactForceTrajectory:
    """Helper: build a ContactForceTrajectory whose initial force points along
    the per-frame contact direction inferred from a 2D pixel pin trajectory.

    For v0 we just initialize all forces to a small random vector; the pixel
    direction is used later by `L_contact_loc` (not here). Provided for API
    symmetry with the SAM+hand-pose detector.
    """
    return ContactForceTrajectory(
        n_frames=n_frames,
        n_total_nodes=n_total_nodes,
        anchor_node_ids=[pin_anchor_node],
        init_scale=init_magnitude,
    )


if __name__ == '__main__':
    T, N = 30, 50
    anchors = [10, 20]
    cft = ContactForceTrajectory(T, N, anchors, init_scale=0.1)
    f = cft()
    print(f'force tensor shape: {tuple(f.shape)}, nonzero rows per frame: '
          f'{(f.norm(dim=-1) > 0).sum(dim=-1).float().mean().item():.2f}')
    print(f'temporal smoothness loss: {cft.temporal_smoothness_loss().item():.4e}')
    print(f'sparsity loss:           {cft.sparsity_loss().item():.4e}')
    cft.temporal_smoothness_loss().backward()
    print(f'grad shape: {tuple(cft.force.grad.shape)}, grad norm: {cft.force.grad.norm().item():.4e}')
