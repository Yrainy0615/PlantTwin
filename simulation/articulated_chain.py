"""Differentiable articulated kinematic-tree dynamics over a rooted branch tree.

State per non-root node `i`:
- theta[i] in R^3: axis-angle of the joint at the parent end of bone `i`
  (the rotation applied AFTER inheriting the parent's accumulated rotation)
- omega[i] in R^3: joint angular velocity in axis-angle space

Forward kinematics (per BFS order from root):
    R_local[i] = exp_so3(theta[i])
    R[i]       = R[parent[i]] @ R_local[i]
    pos[i]     = pos[parent[i]] + R[i] @ (rest_pos[i] - rest_pos[parent[i]])

Joint dynamics (semi-implicit Euler):
    tau[i]   = -k[i] * theta[i] - c[i] * omega[i] + tau_ext[i]
    omega[i] += dt * tau[i] / I[i]
    theta[i] += dt * omega[i]
    Root joint torques are zeroed (root is anchored to world).

External force at node `j` -> torque on every ancestor joint `i`:
    tau_ext[i] += (pos[j] - pos[i]) x f_world[j]

This is an approximate forward dynamics (no inertial coupling between joints,
no Coriolis); adequate for plant-scale motions in v0.
"""

from __future__ import annotations

import torch

from models.structure.graph_cleanup import RootedBranchTree


def exp_so3(omega: torch.Tensor) -> torch.Tensor:
    """Rodrigues exponential map from R^3 (axis-angle) to SO(3). Input [..., 3]."""
    theta = omega.norm(dim=-1, keepdim=True)
    safe_theta = theta.clamp_min(1e-12)
    axis = omega / safe_theta
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    K = torch.zeros(*omega.shape[:-1], 3, 3, device=omega.device, dtype=omega.dtype)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]

    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(K)
    cos_e = cos_t.unsqueeze(-1)
    sin_e = sin_t.unsqueeze(-1)
    R = I + sin_e * K + (1 - cos_e) * (K @ K)
    # Small-angle: just use I
    small = (theta.squeeze(-1) < 1e-6).unsqueeze(-1).unsqueeze(-1)
    return torch.where(small, I, R)


def _bfs_order(parent: torch.Tensor, root_idx: int) -> list[int]:
    N = parent.shape[0]
    children: dict[int, list[int]] = {i: [] for i in range(N)}
    for c in range(N):
        p = int(parent[c].item())
        if p >= 0:
            children[p].append(c)
    order = [root_idx]
    seen = [False] * N
    seen[root_idx] = True
    head = 0
    while head < len(order):
        u = order[head]; head += 1
        for v in children[u]:
            if not seen[v]:
                seen[v] = True
                order.append(v)
    return order


def forward_kinematics(
    theta: torch.Tensor,
    rest_pos: torch.Tensor,
    parent: torch.Tensor,
    bfs_order: list[int],
    depth_groups: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (node_pos [N, 3], node_rot [N, 3, 3]) from joint angles `theta`.

    If `depth_groups` (list[Tensor of long indices], grouped by tree depth) is
    given, FK is processed *one depth level at a time* in batched ops — the
    Python loop is over depth (~tree height) instead of over N. This is a big
    win for autograd: with `rest_pos` differentiable, the per-node loop creates
    O(N) graph nodes and is unusable on CPU at training time.
    """
    R_local = exp_so3(theta)
    if depth_groups is not None:
        N = rest_pos.shape[0]
        I3 = torch.eye(3, device=theta.device, dtype=theta.dtype)
        R = I3.expand(N, 3, 3).contiguous().clone()
        pos = rest_pos.clone()
        for nodes in depth_groups[1:]:
            if nodes.numel() == 0:
                continue
            p = parent[nodes]
            R_new = R[p] @ R_local[nodes]
            offsets = rest_pos[nodes] - rest_pos[p]
            pos_new = pos[p] + (R_new @ offsets.unsqueeze(-1)).squeeze(-1)
            R = R.clone(); R[nodes] = R_new
            pos = pos.clone(); pos[nodes] = pos_new
        return pos, R

    # Fallback: original per-node loop (kept for back-compat).
    N = rest_pos.shape[0]
    I3 = torch.eye(3, device=theta.device, dtype=theta.dtype)
    node_rot_list: list[torch.Tensor | None] = [None] * N
    node_pos_list: list[torch.Tensor | None] = [None] * N
    for i in bfs_order:
        if parent[i] < 0:
            node_rot_list[i] = I3
            node_pos_list[i] = rest_pos[i]
        else:
            p = int(parent[i].item())
            R_p = node_rot_list[p]
            R_i = R_p @ R_local[i]
            offset = rest_pos[i] - rest_pos[p]
            pos_i = node_pos_list[p] + (R_i @ offset.unsqueeze(-1)).squeeze(-1)
            node_rot_list[i] = R_i
            node_pos_list[i] = pos_i
    return torch.stack(node_pos_list, dim=0), torch.stack(node_rot_list, dim=0)


def external_force_to_joint_torques(
    external_force_world: torch.Tensor,  # [N, 3]
    node_pos: torch.Tensor,              # [N, 3]
    parent: torch.Tensor,                # [N] long
    bfs_order: list[int],
    depth_groups: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """For each non-root joint i, sum (pos[j] - pos[i]) x f[j] over all descendants j of i.

    Implemented bottom-up: each node propagates the (force, torque-about-node) wrench
    to its parent. The root receives the full wrench but its torque is discarded.

    If `depth_groups` is given, propagation is batched per depth level (out-of-place
    index_add, autograd-safe). Same speedup as the level-batched FK.
    """
    f_sum = external_force_world
    tau = torch.zeros_like(external_force_world)

    if depth_groups is not None:
        # Walk depths from deepest to shallowest. At each level, push (tau + r x f_sum)
        # of every node at that depth into its parent's accumulator.
        for nodes in reversed(depth_groups):
            if nodes.numel() == 0:
                continue
            mask = parent[nodes] >= 0
            children = nodes[mask]
            if children.numel() == 0:
                continue
            parents = parent[children]
            r = node_pos[children] - node_pos[parents]
            tau_to_parent = tau[children] + torch.linalg.cross(r, f_sum[children])
            tau = tau.index_add(0, parents, tau_to_parent)
            f_sum = f_sum.index_add(0, parents, f_sum[children])
        return tau

    # Fallback: per-node reverse-BFS loop.
    f_sum = f_sum.clone()
    for i in reversed(bfs_order):
        p = int(parent[i].item())
        if p >= 0:
            r = node_pos[i] - node_pos[p]
            tau_to_parent = tau[i] + torch.linalg.cross(r, f_sum[i])
            tau = tau.clone()
            f_sum = f_sum.clone()
            tau[p] = tau[p] + tau_to_parent
            f_sum[p] = f_sum[p] + f_sum[i]
    return tau


class ArticulatedChain(torch.nn.Module):
    """Stateless helper bundling tree topology + an FK / dynamics step.

    Per-edge parameters (k, c, I) are indexed by the child-node index of the edge.
    """
    def __init__(self, tree: RootedBranchTree):
        super().__init__()
        self.N = tree.nodes.shape[0]
        self.root_idx = tree.root_idx
        self.register_buffer('parent', tree.parent)
        self.register_buffer('rest_pos', tree.nodes)
        self.bfs_order = _bfs_order(tree.parent, tree.root_idx)
        # Depth groupings for level-batched FK (massive speedup with autograd
        # on rest_pos: ~depth iters instead of N).
        D = int(tree.depth.max().item()) + 1
        self.depth_groups = [
            (tree.depth == lvl).nonzero(as_tuple=False).flatten() for lvl in range(D)
        ]

    def _depth_groups_on(self, device):
        """Lazy-move depth groups to the requested device (cached)."""
        if not self.depth_groups or self.depth_groups[0].device == device:
            return self.depth_groups
        self.depth_groups = [g.to(device) for g in self.depth_groups]
        return self.depth_groups

    def fk(self, theta: torch.Tensor, rest_pos: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """If `rest_pos` is None, use the buffer baked in at __init__. Pass a
        differentiable tensor when learning structure deltas (rest_pos_eff)."""
        rp = self.rest_pos if rest_pos is None else rest_pos
        groups = self._depth_groups_on(rp.device)
        return forward_kinematics(theta, rp, self.parent, self.bfs_order, groups)

    def step(
        self,
        theta: torch.Tensor,           # [N, 3]
        omega: torch.Tensor,           # [N, 3]
        k_joint: torch.Tensor,         # [N]
        c_joint: torch.Tensor,         # [N]
        I_joint: torch.Tensor,         # [N]
        external_force_world: torch.Tensor,  # [N, 3]
        dt: float,
        rest_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_pos, _ = self.fk(theta, rest_pos=rest_pos)
        tau_ext = external_force_to_joint_torques(
            external_force_world, node_pos, self.parent, self.bfs_order,
            self._depth_groups_on(node_pos.device),
        )
        tau = -k_joint.unsqueeze(-1) * theta - c_joint.unsqueeze(-1) * omega + tau_ext
        tau = tau.clone()
        tau[self.root_idx] = 0.0
        omega_new = omega + dt * tau / I_joint.clamp_min(1e-6).unsqueeze(-1)
        theta_new = theta + dt * omega_new
        return theta_new, omega_new

    def rollout(
        self,
        theta0: torch.Tensor,
        omega0: torch.Tensor,
        k_joint: torch.Tensor,
        c_joint: torch.Tensor,
        I_joint: torch.Tensor,
        external_force_traj: torch.Tensor,  # [T, N, 3]
        dt: float,
        rest_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Roll out T steps. Returns:
            theta_traj [T, N, 3], omega_traj [T, N, 3],
            node_pos_traj [T, N, 3], node_rot_traj [T, N, 3, 3]
        """
        T = external_force_traj.shape[0]
        theta = theta0
        omega = omega0
        thetas, omegas, poss, rots = [], [], [], []
        for t in range(T):
            theta, omega = self.step(
                theta, omega, k_joint, c_joint, I_joint, external_force_traj[t], dt,
                rest_pos=rest_pos,
            )
            pos, rot = self.fk(theta, rest_pos=rest_pos)
            thetas.append(theta)
            omegas.append(omega)
            poss.append(pos)
            rots.append(rot)
        return (
            torch.stack(thetas, dim=0),
            torch.stack(omegas, dim=0),
            torch.stack(poss, dim=0),
            torch.stack(rots, dim=0),
        )


if __name__ == '__main__':
    import argparse
    from data.gaussian_plant_loader import load_gaussian_plant_scene
    from models.structure.graph_cleanup import root_branch_graph

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    parser.add_argument('--output', default='outputs/gsplant_output/newplant9')
    parser.add_argument('--frames', type=int, default=20)
    args = parser.parse_args()

    scene = load_gaussian_plant_scene(args.source, args.output)
    tree = root_branch_graph(scene.branch, scene.tube)
    chain = ArticulatedChain(tree)

    N = chain.N
    theta = torch.zeros(N, 3, requires_grad=True)
    omega = torch.zeros(N, 3, requires_grad=True)
    k_joint = torch.full((N,), 5.0)
    c_joint = torch.full((N,), 0.5)
    I_joint = torch.full((N,), 0.01)

    # Apply a constant lateral force to a leaf-bearing terminal
    terminal = int(((tree.subtree_size == 1).nonzero().flatten()[0]).item())
    f_traj = torch.zeros(args.frames, N, 3)
    f_traj[:, terminal, 0] = 0.5  # push in +x

    theta_t, omega_t, pos_t, rot_t = chain.rollout(theta, omega, k_joint, c_joint, I_joint, f_traj, dt=0.02)

    print(f'rollout: T={pos_t.shape[0]}, N={pos_t.shape[1]}')
    disp = (pos_t[-1] - tree.nodes).norm(dim=-1)
    print(f'final-frame node displacement: max={disp.max():.4f}, terminal={disp[terminal]:.4f}')

    # Gradient flow sanity check: loss through the whole rollout, expect non-zero grads on theta.
    loss = (pos_t[-1, terminal] - tree.nodes[terminal]).pow(2).sum()
    loss.backward()
    print(f'grad on theta exists: {theta.grad is not None}, norm={theta.grad.norm().item():.4e}')
