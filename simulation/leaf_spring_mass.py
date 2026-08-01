"""Per-leaf spring-mass simulator scoped to a single leaf disk.

The full-plant `simulation.spring_mass.SpringMassSimulator` is kept untouched
(it is still the engine for the pretrain pipeline). This module is a thin,
leaf-specific variant: input is one leaf's points + a disk frame, simulation
runs in disk-local coordinates, and an output displacement field can be applied
in world frame via the leaf's current disk pose.

Topology: KNN within the leaf cluster. A future upgrade is Delaunay-on-disk
triangulation or an explicit radial+circumferential pattern.

This is wired up but not yet plugged into the v0 per-scene optimizer (leaves
move rigidly with their bone in v0). Provided for completeness so the API is
stable when leaf-internal deformation becomes part of the loss.
"""

from __future__ import annotations

import torch
from torch import nn


class LeafSpringMass(nn.Module):
    """KNN spring-mass over a single leaf cluster, simulated in disk-local 3D.

    Args:
        points_local: [N_k, 3] leaf points in the disk frame at rest
                      (e.g., world-frame points minus disk center; rotated if needed)
        k_neighbors: KNN connectivity
        dt, n_substeps: integration timestep + substeps per output frame
    """
    def __init__(
        self,
        points_local: torch.Tensor,
        k_neighbors: int = 6,
        dt: float = 0.01,
        n_substeps: int = 5,
    ):
        super().__init__()
        self.dt = dt
        self.n_substeps = n_substeps
        self.eps = 1e-14
        self.register_buffer('rest_local', points_local.detach().clone())

        rest = points_local
        d = torch.cdist(rest, rest)
        d.fill_diagonal_(float('inf'))
        rest_len, knn_idx = d.topk(k_neighbors, dim=1, largest=False)
        self.register_buffer('rest_len', rest_len)
        self.register_buffer('knn_idx', knn_idx)
        self.k_neighbors = k_neighbors

    def _force(self, x: torch.Tensor, v: torch.Tensor, k: torch.Tensor, c: torch.Tensor):
        """Strain-based spring + edge damping in disk-local frame."""
        nb = x[self.knn_idx]                         # [N, K, 3]
        delta = nb - x.unsqueeze(1)                  # [N, K, 3]
        L = delta.norm(dim=-1)                       # [N, K]
        dir_ = delta / (L.unsqueeze(-1) + self.eps)

        strain = (L / (self.rest_len + self.eps) - 1.0).clamp(-0.5, 0.5)
        f_spring = (strain * k).unsqueeze(-1) * dir_

        nb_v = v[self.knn_idx]
        delta_v = nb_v - v.unsqueeze(1)
        f_damp = (c * (delta_v * dir_).sum(-1)).unsqueeze(-1) * dir_

        return (f_spring + f_damp).sum(dim=1)

    def rollout(
        self,
        n_frames: int,
        k: torch.Tensor,             # [N, K] or scalar broadcast
        c: torch.Tensor,             # [N, K] or scalar broadcast
        m: torch.Tensor,             # [N] or scalar
        external_force_local: torch.Tensor | None = None,   # [T, N, 3] in disk frame
        pinned_mask: torch.Tensor | None = None,            # [N] bool — pinned points (zero velocity)
    ) -> torch.Tensor:
        """Returns trajectory [T, N, 3] in disk-local frame."""
        N = self.rest_local.shape[0]
        x = self.rest_local.clone()
        v = torch.zeros_like(x)

        k_b = k if k.dim() == 2 else k.expand(N, self.k_neighbors)
        c_b = c if c.dim() == 2 else c.expand(N, self.k_neighbors)
        m_b = m if m.dim() == 1 else m.expand(N)

        traj = []
        for t in range(n_frames):
            for _ in range(self.n_substeps):
                f = self._force(x, v, k_b, c_b)
                if external_force_local is not None:
                    f = f + external_force_local[t]
                v = v + self.dt * f / m_b.unsqueeze(-1).clamp_min(1e-6)
                if pinned_mask is not None:
                    v = v.clone()
                    v[pinned_mask] = 0.0
                x = x + self.dt * v
            traj.append(x)
        return torch.stack(traj, dim=0)


def apply_disk_pose(
    traj_local: torch.Tensor,        # [T, N, 3] in disk frame
    disk_center_t: torch.Tensor,     # [T, 3]
    disk_rot_t: torch.Tensor,        # [T, 3, 3]
) -> torch.Tensor:
    """Transform per-frame leaf trajectory from disk-local to world coords."""
    # world = disk_center + R @ local
    rotated = torch.einsum('tij,tnj->tni', disk_rot_t, traj_local)
    return disk_center_t[:, None, :] + rotated


if __name__ == '__main__':
    N = 30
    pts = torch.randn(N, 3) * 0.1
    sim = LeafSpringMass(pts, k_neighbors=6, dt=0.005, n_substeps=4)
    k = torch.full((N, 6), 100.0)
    c = torch.full((N, 6), 5.0)
    m = torch.full((N,), 0.001)
    fext = torch.zeros(10, N, 3)
    fext[:, 0, 0] = 0.5  # push one node in +x
    pinned = torch.zeros(N, dtype=torch.bool); pinned[N - 1] = True

    traj = sim.rollout(10, k, c, m, external_force_local=fext, pinned_mask=pinned)
    print(f'leaf rollout: T={traj.shape[0]}, N={traj.shape[1]}')
    disp = (traj[-1] - sim.rest_local).norm(dim=-1)
    print(f'final displacement: max={disp.max():.4f}, pinned (should be ~0)={disp[-1]:.4e}')
