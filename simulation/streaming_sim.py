"""Stateful streaming simulator for interactive demo.

Mirrors the v11 `PerScenePhysicsModel` forward pass — articulated branch chain
+ per-leaf petiole oscillator — but steps one timestep at a time and holds
state (theta, omega, theta_pet, omega_pet) across calls. Used by the viser
interactive demo where the user drags on the plant and the simulation must
respond in real time.

Performance notes:
- Chain FK + reverse-BFS torque propagation use *level-batched* vectorization
  (process all nodes at one depth in a single op) and run on CPU because the
  branch tree is small (~400 nodes, ~95 depths) and CUDA kernel-launch latency
  dominates at that scale.
- Skinning runs on GPU because it touches 700k ApPs.
"""

from __future__ import annotations

import torch

from models.structure.graph_cleanup import STEM
from models.structure.skinning import BoneSkinning, apply_skinning
from simulation.articulated_chain import exp_so3


def _per_node_param(val_stem, val_branch, edges, etype, N, device):
    per_edge = torch.where(
        etype == STEM, val_stem.expand(edges.shape[0]), val_branch.expand(edges.shape[0])
    )
    per_node = torch.zeros(N, device=device, dtype=val_stem.dtype)
    per_node = per_node.scatter(0, edges[:, 1], per_edge)
    return per_node


class StreamingPlantSim:
    """One-step-at-a-time articulated-chain + petiole simulator with LBS.

    `step(dt, ext_force)` advances state by `dt` and returns the current per-ApP
    world-space positions. Designed to be called from a render thread at ~30 Hz.
    """

    def __init__(self, tree, attachments, skinning, params: dict, ap_xyz_rest, device):
        self.tree = tree
        self.attachments = attachments
        self.device = device
        self.cpu = torch.device('cpu')
        self.N = tree.nodes.shape[0]
        self.N_leaf = len(attachments)
        self.root_idx = tree.root_idx

        # Chain topology on CPU
        edges = tree.edges_oriented
        etype = tree.edge_type
        self.parent_cpu = tree.parent
        self.rest_pos_cpu = tree.nodes

        # Per-depth groupings for level-batched FK / reverse-BFS torque propagation
        depth = tree.depth
        D = int(depth.max().item()) + 1
        self.levels_fwd = [(depth == lvl).nonzero(as_tuple=False).flatten() for lvl in range(D)]
        self.levels_rev = list(reversed(self.levels_fwd))

        # Physics params (per-type scalars promoted to per-node) on CPU
        k_stem = params['log_k_stem'].exp().cpu()
        k_branch = params['log_k_branch'].exp().cpu()
        c_stem = params['log_c_stem'].exp().cpu()
        c_branch = params['log_c_branch'].exp().cpu()
        inertia = params['log_inertia'].exp().cpu()
        self.k_pn = _per_node_param(k_stem, k_branch, edges, etype, self.N, self.cpu)
        self.c_pn = _per_node_param(c_stem, c_branch, edges, etype, self.N, self.cpu)
        self.I_pn = inertia.expand(self.N).clone()

        # Petiole oscillator params (global scalars) on CPU
        self.k_pet = params['log_k_petiole'].exp().cpu()
        self.c_pet = params['log_c_petiole'].exp().cpu()

        # Leaf attachment buffers on CPU (small N_leaf=43)
        self.leaf_parent_cpu = torch.tensor(
            [int(edges[a.parent_edge_idx, 0].item()) for a in attachments], dtype=torch.long,
        )
        self.leaf_child_cpu = torch.tensor(
            [int(edges[a.parent_edge_idx, 1].item()) for a in attachments], dtype=torch.long,
        )
        self.petiole_offset_cpu = torch.stack(
            [a.surface_point - tree.nodes[self.leaf_parent_cpu[k]] for k, a in enumerate(attachments)]
        )
        self.disk_off_cpu = torch.stack([a.rest_direction * a.rest_length for a in attachments])

        # Skinning on GPU (where ApPs live)
        self.skinning = BoneSkinning(
            bone_idx=skinning.bone_idx.to(device),
            leaf_idx=skinning.leaf_idx.to(device),
            local_bone=skinning.local_bone.to(device),
            local_leaf=skinning.local_leaf.to(device),
        )
        self.ap_xyz_rest = ap_xyz_rest.to(device)
        self.edges_gpu = edges.to(device)

        # State (CPU)
        self.theta = torch.zeros(self.N, 3)
        self.omega = torch.zeros(self.N, 3)
        self.theta_pet = torch.zeros(self.N_leaf, 3)
        self.omega_pet = torch.zeros(self.N_leaf, 3)
        self.prev_child_rot = torch.eye(3).expand(self.N_leaf, 3, 3).contiguous()
        self.prev_omega_parent = torch.zeros(self.N_leaf, 3)

    def _fk_levelbatched(self, theta: torch.Tensor):
        """Level-batched forward kinematics on CPU. Returns (pos [N, 3], R [N, 3, 3])."""
        R_local = exp_so3(theta)
        R = torch.eye(3).expand(self.N, 3, 3).contiguous().clone()
        pos = self.rest_pos_cpu.clone()
        for nodes in self.levels_fwd[1:]:
            p = self.parent_cpu[nodes]
            Rn = R[p] @ R_local[nodes]
            R[nodes] = Rn
            offsets = self.rest_pos_cpu[nodes] - self.rest_pos_cpu[p]
            pos[nodes] = pos[p] + (Rn @ offsets.unsqueeze(-1)).squeeze(-1)
        return pos, R

    def _reverse_bfs_torques(self, ext_force: torch.Tensor, node_pos: torch.Tensor) -> torch.Tensor:
        """Reverse-BFS propagation of (force, torque-about-node) wrench to ancestors.

        Returns per-joint external torque [N, 3]. Root torque is left zero by the caller.
        """
        f_sum = ext_force.clone()
        tau = torch.zeros_like(ext_force)
        for nodes in self.levels_rev:
            if len(nodes) == 0:
                continue
            # Skip root level (no parent to propagate to)
            mask = self.parent_cpu[nodes] >= 0
            children = nodes[mask]
            if children.numel() == 0:
                continue
            parents = self.parent_cpu[children]
            r = node_pos[children] - node_pos[parents]
            tau_to_parent = tau[children] + torch.linalg.cross(r, f_sum[children])
            # scatter-add into parents (a single child per (depth, parent) is typical,
            # but multiple children at the same depth can share a parent — use index_add_)
            tau.index_add_(0, parents, tau_to_parent)
            f_sum.index_add_(0, parents, f_sum[children])
        return tau

    @torch.no_grad()
    def step(self, dt: float, ext_force_cpu: torch.Tensor):
        """Advance one substep. `ext_force_cpu` is [N, 3] CPU world-frame force per node.
        Returns (ap_xyz [N_ap, 3] on GPU, pos [N, 3] CPU, rot [N, 3, 3] CPU)."""
        # 1) Joint dynamics step (semi-implicit Euler) — needs current node_pos for tau_ext
        pos_cur, _ = self._fk_levelbatched(self.theta)
        tau_ext = self._reverse_bfs_torques(ext_force_cpu, pos_cur)
        tau = -self.k_pn.unsqueeze(-1) * self.theta - self.c_pn.unsqueeze(-1) * self.omega + tau_ext
        tau[self.root_idx] = 0.0
        self.omega = self.omega + dt * tau / self.I_pn.clamp_min(1e-6).unsqueeze(-1)
        self.theta = self.theta + dt * self.omega

        # 2) Forward kinematics after the joint update
        pos, rot = self._fk_levelbatched(self.theta)

        # 3) Petiole oscillator: base excitation from parent bone's angular accel
        child_rot = rot[self.leaf_child_cpu]                            # [L, 3, 3]
        R_delta = child_rot @ self.prev_child_rot.transpose(-1, -2)
        omega_parent = torch.stack([
            R_delta[..., 2, 1] - R_delta[..., 1, 2],
            R_delta[..., 0, 2] - R_delta[..., 2, 0],
            R_delta[..., 1, 0] - R_delta[..., 0, 1],
        ], dim=-1) * 0.5 / dt
        alpha_parent = (omega_parent - self.prev_omega_parent) / dt
        tau_pet = -self.k_pet * self.theta_pet - self.c_pet * self.omega_pet - alpha_parent
        self.omega_pet = self.omega_pet + dt * tau_pet
        self.theta_pet = self.theta_pet + dt * self.omega_pet
        self.prev_child_rot = child_rot
        self.prev_omega_parent = omega_parent

        R_pet = exp_so3(self.theta_pet)
        leaf_rot = torch.einsum('lij,ljk->lik', child_rot, R_pet)
        surface = pos[self.leaf_parent_cpu] + torch.einsum(
            'lij,lj->li', child_rot, self.petiole_offset_cpu
        )
        leaf_center = surface + torch.einsum('lij,lj->li', leaf_rot, self.disk_off_cpu)

        # 4) Move pos / rot to GPU and run LBS skinning
        pos_gpu = pos.to(self.device, non_blocking=True)
        rot_gpu = rot.to(self.device, non_blocking=True)
        leaf_center_gpu = leaf_center.to(self.device, non_blocking=True)
        leaf_rot_gpu = leaf_rot.to(self.device, non_blocking=True)
        ap_xyz = apply_skinning(
            self.skinning, pos_gpu, rot_gpu, self.edges_gpu,
            leaf_center_gpu, leaf_rot_gpu, self.ap_xyz_rest,
        )
        return ap_xyz, pos, rot

    def reset(self):
        self.theta.zero_(); self.omega.zero_()
        self.theta_pet.zero_(); self.omega_pet.zero_()
        self.prev_child_rot = torch.eye(3).expand(self.N_leaf, 3, 3).contiguous()
        self.prev_omega_parent.zero_()

    @torch.no_grad()
    def current_node_positions(self) -> torch.Tensor:
        pos, _ = self._fk_levelbatched(self.theta)
        return pos
