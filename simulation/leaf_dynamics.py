"""Per-leaf spring-mass dynamics, batched across all leaves of the plant.

Runs each leaf as an independent KNN spring-mass system in its own *disk-local*
frame, with a single base-excitation force per leaf coming from the disk
center's world acceleration (so when the petiole / parent bone slings the disk
around, the leaf's points lag and wobble in disk-local coordinates).

Then maps the leaf node displacement field back onto the leaf-bound ApPs via
IDW weights computed once at rest.

Why batched: ~40 leaves on a typical plant. Stacking them into padded tensors
makes the rollout a handful of large ops instead of a Python loop.
"""

from __future__ import annotations

import torch
from torch import nn


class LeafDynamicsBank(nn.Module):
    """Batched per-leaf spring-mass with shared per-type parameters.

    Args:
        leaves_local_list: list of per-leaf disk-local rest points [N_l, 3].
        ap_local_leaf: [N_ap, 3] disk-local rest position of every ApP.
            For non-leaf-bound ApPs the entries are unused (but must exist).
        ap_leaf_idx: [N_ap] which leaf each ApP is bound to (-1 = bone/static).
        k_neighbors: KNN connectivity within a leaf cluster.
        k_neighbors_ap: KNN used for ApP->leaf-node IDW interpolation.
        dt, n_substeps: integration timestep and substeps per output frame.
    """
    def __init__(
        self,
        leaves_local_list: list[torch.Tensor],
        ap_local_leaf: torch.Tensor,
        ap_leaf_idx: torch.Tensor,
        pin_local_list: list[torch.Tensor] | None = None,
        k_neighbors: int = 6,
        k_neighbors_ap: int = 4,
        pin_top_k: int = 3,
        dt: float = 0.005,
        n_substeps: int = 4,
    ):
        """If `pin_local_list[l]` is given (a [3] tensor in disk-local frame), the
        `pin_top_k` cluster nodes closest to that point are pinned (rigidly follow
        the disk frame). This anchors the leaf to its petiole attachment so the
        only DOFs are spring deformations relative to the anchor."""
        super().__init__()
        self.dt = dt
        self.n_substeps = n_substeps
        self.k_neighbors = k_neighbors
        self.k_neighbors_ap = k_neighbors_ap
        self.eps = 1e-14

        L = len(leaves_local_list)
        M = max((p.shape[0] for p in leaves_local_list), default=0)

        rest_pad = torch.zeros(L, M, 3)
        node_mask = torch.zeros(L, M, dtype=torch.bool)
        knn_idx_pad = torch.zeros(L, M, k_neighbors, dtype=torch.long)
        knn_len_pad = torch.zeros(L, M, k_neighbors)
        edge_mask_pad = torch.zeros(L, M, k_neighbors, dtype=torch.bool)
        pin_mask_pad = torch.zeros(L, M, dtype=torch.bool)

        for l, pts in enumerate(leaves_local_list):
            N_l = pts.shape[0]
            rest_pad[l, :N_l] = pts
            node_mask[l, :N_l] = True
            if N_l > 1:
                K = min(k_neighbors, N_l - 1)
                d = torch.cdist(pts, pts)
                d.fill_diagonal_(float('inf'))
                top = d.topk(K, dim=1, largest=False)
                knn_idx_pad[l, :N_l, :K] = top.indices
                knn_len_pad[l, :N_l, :K] = top.values
                edge_mask_pad[l, :N_l, :K] = True

            if pin_local_list is not None and N_l > 0:
                pin_pt = pin_local_list[l]
                Kp = min(pin_top_k, N_l)
                d_pin = (pts - pin_pt).norm(dim=-1)
                pin_idx = d_pin.topk(Kp, largest=False).indices
                pin_mask_pad[l, pin_idx] = True

        self.register_buffer('rest_local', rest_pad)            # [L, M, 3]
        self.register_buffer('node_mask', node_mask)            # [L, M]
        self.register_buffer('knn_idx', knn_idx_pad)            # [L, M, K]
        self.register_buffer('rest_len', knn_len_pad)           # [L, M, K]
        self.register_buffer('edge_mask', edge_mask_pad)        # [L, M, K]
        self.register_buffer('pin_mask', pin_mask_pad)          # [L, M]  True = anchored to disk frame

        # ApP -> leaf-node IDW
        N_ap = ap_local_leaf.shape[0]
        ap_node_idx = torch.zeros(N_ap, k_neighbors_ap, dtype=torch.long)
        ap_weights = torch.zeros(N_ap, k_neighbors_ap)
        leaf_bound = ap_leaf_idx >= 0
        for ap_id in leaf_bound.nonzero(as_tuple=False).flatten().tolist():
            l = int(ap_leaf_idx[ap_id].item())
            pts = leaves_local_list[l]
            N_l = pts.shape[0]
            if N_l == 0:
                continue
            d = (pts - ap_local_leaf[ap_id]).norm(dim=-1)
            K = min(k_neighbors_ap, N_l)
            top = d.topk(K, largest=False)
            inv = 1.0 / (top.values + 1e-6)
            w = inv / inv.sum()
            ap_node_idx[ap_id, :K] = top.indices
            ap_weights[ap_id, :K] = w

        self.register_buffer('ap_leaf_idx', ap_leaf_idx.clone())   # [N_ap]
        self.register_buffer('ap_node_idx', ap_node_idx)           # [N_ap, K_ap]
        self.register_buffer('ap_weights', ap_weights)             # [N_ap, K_ap]
        self.register_buffer('leaf_bound_mask', leaf_bound)        # [N_ap]

    def _force(self, x: torch.Tensor, v: torch.Tensor, k: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Strain-based spring + edge damping, batched across leaves.

        x, v: [L, M, 3]; k, c: scalar tensors.
        Returns force [L, M, 3]; padded nodes/edges contribute zero.
        """
        L, M, K = self.knn_idx.shape
        l_idx = torch.arange(L, device=x.device)[:, None, None].expand(L, M, K)
        nb = x[l_idx, self.knn_idx]                       # [L, M, K, 3]
        nb_v = v[l_idx, self.knn_idx]                     # [L, M, K, 3]
        delta = nb - x.unsqueeze(2)
        L_len = delta.norm(dim=-1)
        dir_ = delta / (L_len.unsqueeze(-1) + self.eps)
        strain = (L_len / (self.rest_len + self.eps) - 1.0).clamp(-0.5, 0.5)
        f_spring = (strain * k).unsqueeze(-1) * dir_
        delta_v = nb_v - v.unsqueeze(2)
        f_damp = (c * (delta_v * dir_).sum(-1)).unsqueeze(-1) * dir_
        f = f_spring + f_damp
        f = f * self.edge_mask.unsqueeze(-1)              # zero out padded edges
        return f.sum(dim=2)

    def rollout(
        self,
        disk_center_t: torch.Tensor,   # [T, L, 3] world
        disk_rot_t: torch.Tensor,      # [T, L, 3, 3] world
        k_scalar: torch.Tensor,
        c_scalar: torch.Tensor,
        m_scalar: torch.Tensor,
    ) -> torch.Tensor:
        """Returns [T, L, M, 3] per-leaf node trajectories in disk-local frame."""
        T = disk_center_t.shape[0]
        L, M = self.rest_local.shape[:2]
        dt = self.dt
        device = disk_center_t.device

        # Disk linear acceleration in world frame -> disk-local (inertial force = -m * a_local).
        # disk_center_t samples are at OUTER-frame spacing (dt * n_substeps), not the
        # inner integration dt — divide by that, otherwise a is overestimated by n_substeps^2.
        outer_dt = self.dt * self.n_substeps
        c_w = disk_center_t
        c_prev = torch.cat([c_w[:1], c_w[:-1]], dim=0)
        c_next = torch.cat([c_w[1:], c_w[-1:]], dim=0)
        a_world = (c_next - 2 * c_w + c_prev) / (outer_dt * outer_dt)      # [T, L, 3]
        a_local = torch.einsum('tlij,tlj->tli', disk_rot_t.transpose(-1, -2), a_world)

        x = self.rest_local.clone()
        v = torch.zeros_like(x)
        node_mask_3 = self.node_mask.unsqueeze(-1).to(x.dtype)             # [L, M, 1]
        free_mask_3 = (self.node_mask & ~self.pin_mask).unsqueeze(-1).to(x.dtype)

        traj = []
        for t in range(T):
            f_inertial = -m_scalar * a_local[t][:, None, :]                # [L, 1, 3]
            for _ in range(self.n_substeps):
                f = self._force(x, v, k_scalar, c_scalar)                  # [L, M, 3]
                f = f + f_inertial
                f = f * free_mask_3                                         # pinned + padded -> zero
                v = v + dt * f / m_scalar.clamp_min(1e-6)
                v = v * free_mask_3                                         # zero pinned velocity
                x = x + dt * v
                # Re-clamp pinned nodes to rest (rigid follow of disk frame).
                x = torch.where(self.pin_mask.unsqueeze(-1), self.rest_local, x)
            traj.append(x)
        return torch.stack(traj, dim=0)                                     # [T, L, M, 3]

    def map_ap_world(
        self,
        traj_local: torch.Tensor,        # [T, L, M, 3]
        ap_local_leaf: torch.Tensor,     # [N_ap, 3]
        disk_center_t: torch.Tensor,     # [T, L, 3]
        disk_rot_t: torch.Tensor,        # [T, L, 3, 3]
    ) -> torch.Tensor:
        """Returns [T, N_ap_leaf_bound, 3] world positions for leaf-bound ApPs,
        keyed by the order in `leaf_bound_mask.nonzero()`. Caller should scatter
        these into the full ap buffer using `leaf_bound_mask`.
        """
        T = disk_center_t.shape[0]
        if not self.leaf_bound_mask.any():
            return torch.zeros(T, 0, 3, device=disk_center_t.device)

        ap_idx = self.leaf_bound_mask.nonzero(as_tuple=False).flatten()
        l_idx = self.ap_leaf_idx[ap_idx]                                    # [N_a]
        node_idx = self.ap_node_idx[ap_idx]                                 # [N_a, K]
        w = self.ap_weights[ap_idx]                                         # [N_a, K]
        local = ap_local_leaf[ap_idx]                                        # [N_a, 3]
        K = node_idx.shape[1]

        # Displacement field in disk-local frame
        disp_t = traj_local - self.rest_local[None]                          # [T, L, M, 3]
        # Gather per (n, k): disp_t[t, l_idx[n], node_idx[n, k], :]
        l_idx_exp = l_idx.unsqueeze(1).expand(-1, K)                         # [N_a, K]
        ap_disp = disp_t[:, l_idx_exp, node_idx, :]                          # [T, N_a, K, 3]
        ap_disp = (ap_disp * w[None, :, :, None]).sum(dim=2)                 # [T, N_a, 3]

        # World pose
        ap_world_local = local[None] + ap_disp                                # [T, N_a, 3]
        R_l = disk_rot_t[:, l_idx]                                            # [T, N_a, 3, 3]
        center_l = disk_center_t[:, l_idx]                                    # [T, N_a, 3]
        ap_world = center_l + torch.einsum('tnij,tnj->tni', R_l, ap_world_local)
        return ap_world


if __name__ == '__main__':
    # Smoke test: two tiny leaves with different node counts.
    L1 = torch.randn(20, 3) * 0.1
    L2 = torch.randn(35, 3) * 0.08
    # 5 ApPs: 3 in leaf 0, 2 in leaf 1
    ap_local = torch.cat([L1[:3], L2[:2]], dim=0)
    ap_leaf_idx = torch.tensor([0, 0, 0, 1, 1])
    bank = LeafDynamicsBank([L1, L2], ap_local, ap_leaf_idx, dt=0.005, n_substeps=4)

    T = 12
    centers = torch.zeros(T, 2, 3)
    centers[:, 0, 0] = torch.linspace(0, 0.05, T)
    centers[:, 1, 1] = torch.linspace(0, 0.03, T)
    rots = torch.eye(3).expand(T, 2, 3, 3).contiguous()
    k = torch.tensor(20.0); c = torch.tensor(2.0); m = torch.tensor(0.05)
    traj = bank.rollout(centers, rots, k, c, m)
    ap_world = bank.map_ap_world(traj, ap_local, centers, rots)
    print(f'traj {tuple(traj.shape)}, ap_world {tuple(ap_world.shape)}')
    # Canonical-pose check: zero center motion + identity rot -> ap_world = center + local = local
    centers0 = torch.zeros(T, 2, 3)
    traj0 = bank.rollout(centers0, rots, k, c, m)
    disp0 = (traj0 - bank.rest_local[None]).abs().max().item()
    ap0 = bank.map_ap_world(traj0, ap_local, centers0, rots)
    err = (ap0 - ap_local[None].expand(T, -1, -1)).abs().max().item()
    print(f'no-motion node disp max: {disp0:.3e}, ap world err vs rest: {err:.3e}')
