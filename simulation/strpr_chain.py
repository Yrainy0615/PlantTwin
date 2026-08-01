"""Kinematic chain over StPr motion nodes (Stage-3 dynamic refinement).

Turns the loose set of branch StPr into a rooted tree and runs forward kinematics,
so a small set of joint angles drives every StPr's world transform — which in turn
drives the AppGas through Gaussian-distance LBS (models/structure/strpr_motion).

Tree: kNN graph over StPr centers -> MST -> rooted at the lowest (base) node.
FK per node i (BFS from root):
    R[i]   = R[parent] @ exp_so3(theta[i]) @ R0[i]      (R0[i] = StPr rest rotation)
    c[i]   = c[parent] + R_joint[i] @ (c0[i] - c0[parent])
where R_joint[i] = R[parent] @ exp_so3(theta[i]) @ R[parent]^T is the rigid rotation
of bone i about its parent. At theta=0, R[i]=R0[i] and c[i]=c0[i] (rest reproduced).

`node_rot` returned here feeds models.structure.strpr_motion.apply_strpr_lbs directly
(it expects node_rot such that world = c + node_rot @ (R0^T (x-c0)) = c + R_fk @ (x-c0)).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from simulation.articulated_chain import exp_so3
from models.structure.strpr_motion import StprBranch


@dataclass
class StprTree:
    parent: torch.Tensor        # [S] long, -1 for root
    order: torch.Tensor         # [S] long, BFS order (root first)
    root: int
    c0: torch.Tensor            # [S,3] rest centers
    R0: torch.Tensor            # [S,3,3] rest rotations
    edges: torch.Tensor         # [S-1,2] (parent,child) undirected tree edges

    def to(self, device):
        return StprTree(self.parent.to(device), self.order.to(device), self.root,
                        self.c0.to(device), self.R0.to(device), self.edges.to(device))


def _mst_edges(pts: torch.Tensor, k: int = 8) -> torch.Tensor:
    """kNN graph -> minimum spanning tree (Prim). Returns [M,2] undirected edges."""
    S = pts.shape[0]
    d = torch.cdist(pts, pts)                       # [S,S]
    knn = torch.topk(d, min(k + 1, S), largest=False).indices  # includes self
    INF = float('inf')
    W = torch.full((S, S), INF)
    for i in range(S):
        for j in knn[i].tolist():
            if j != i:
                W[i, j] = d[i, j]; W[j, i] = d[i, j]
    # ensure connectivity: if graph disconnected, add global-nearest bridges by falling
    # back to the full distance matrix for unreached nodes during Prim.
    in_tree = torch.zeros(S, dtype=torch.bool)
    in_tree[0] = True
    best = W[0].clone()
    best_from = torch.zeros(S, dtype=torch.long)
    edges = []
    for _ in range(S - 1):
        best_masked = best.clone(); best_masked[in_tree] = INF
        j = int(torch.argmin(best_masked).item())
        if not torch.isfinite(best_masked[j]):      # disconnected -> bridge via full dist
            dd = d.clone(); dd[:, in_tree] = INF; dd[~in_tree.unsqueeze(1).expand(S, S)] = INF
            src = torch.where(in_tree)[0]
            sub = d[src][:, ~in_tree]
            a = int(src[(sub.min(1).values).argmin()].item())
            rem = torch.where(~in_tree)[0]
            j = int(rem[(d[a][rem]).argmin()].item())
            best_from[j] = a; best[j] = d[a, j]
        edges.append((best_from[j].item(), j))
        in_tree[j] = True
        upd = W[j] < best
        best = torch.where(upd, W[j], best)
        best_from = torch.where(upd, torch.full_like(best_from, j), best_from)
    return torch.tensor(edges, dtype=torch.long)


def build_strpr_tree(strpr: StprBranch, k: int = 8, root: int | None = None) -> StprTree:
    c0 = strpr.xyz
    edges = _mst_edges(c0, k=k)
    S = c0.shape[0]
    if root is None:
        root = int(torch.argmin(c0[:, 1]).item())   # lowest node = base (y up)
    # root the tree: BFS to assign parents
    adj = [[] for _ in range(S)]
    for a, b in edges.tolist():
        adj[a].append(b); adj[b].append(a)
    parent = torch.full((S,), -1, dtype=torch.long)
    order = []
    seen = torch.zeros(S, dtype=torch.bool)
    stack = [root]; seen[root] = True
    while stack:
        u = stack.pop(0)
        order.append(u)
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True; parent[v] = u; stack.append(v)
    return StprTree(parent=parent, order=torch.tensor(order, dtype=torch.long), root=root,
                    c0=c0, R0=strpr.R, edges=edges)


def fk(tree: StprTree, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward kinematics. theta [S,3] axis-angle joint at each node.

    Handles a FOREST: every node with parent<0 is a root, anchored at its rest pose.
    Returns node_pos [S,3], node_rot [S,3,3] with node_rot[i]=R_fk[i]@R0[i]. theta=0 -> (c0,R0).
    """
    pos, node_rot = fk_batch(tree, theta[None])
    return pos[0], node_rot[0]


def fk_batch(tree: StprTree, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched, differentiable FK over T frames. theta [T,S,3] -> pos [T,S,3], rot [T,S,3,3].

    Forest-aware: nodes with parent<0 are roots anchored at (I, c0). Backpropagates to theta.
    """
    T, S, _ = theta.shape
    dev = tree.c0.device
    Rjoint = exp_so3(theta.reshape(-1, 3)).reshape(T, S, 3, 3)      # [T,S,3,3]
    I = torch.eye(3, device=dev).expand(T, 3, 3)
    Rfk = [None] * S
    pos = [None] * S
    parent = tree.parent
    c0 = tree.c0
    for i in tree.order.tolist():
        p = int(parent[i].item())
        if p < 0:                                                   # root of a fragment
            Rfk[i] = I
            pos[i] = c0[i].expand(T, 3)
        else:
            Rfk[i] = Rfk[p] @ Rjoint[:, i]
            pos[i] = pos[p] + torch.einsum('tij,j->ti', Rfk[i], (c0[i] - c0[p]))
    Rfk = torch.stack(Rfk, 1)                                       # [T,S,3,3]
    pos = torch.stack(pos, 1)                                       # [T,S,3]
    node_rot = Rfk @ tree.R0[None]
    return pos, node_rot


def _shortest_arc(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Batched shortest-arc rotation mapping unit vectors u -> v. [T,3]x[T,3] -> [T,3,3]."""
    c = (u * v).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)              # cos
    axis = torch.cross(u, v, dim=-1)
    s = axis.norm(dim=-1).clamp_min(1e-8)                        # sin
    axis = axis / s[:, None]
    K = torch.zeros(u.shape[0], 3, 3, device=u.device, dtype=u.dtype)
    K[:, 0, 1] = -axis[:, 2]; K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2];  K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]; K[:, 2, 1] = axis[:, 0]
    I = torch.eye(3, device=u.device, dtype=u.dtype).expand_as(K)
    return I + s[:, None, None] * K + (1 - c)[:, None, None] * (K @ K)


def fit_fk_from_nodes(tree: StprTree, obs_node_pos: torch.Tensor):
    """Fit the FK chain to OBSERVED node trajectories, closed-form and guaranteed-connected.

    Per BFS node i with parent p: choose the child's accumulated rotation R_fk[i] as the
    shortest-arc rotation aligning the rest bone direction to the observed direction
    (obs[i] - pos_fk[p]), then set pos_fk[i] = pos_fk[p] + R_fk[i] @ (c0[i]-c0[p]).
    Bone lengths are preserved EXACTLY and children stay anchored to parents, so branches
    cannot break apart — the structural constraint the per-StPr independent rigid fit lacks.
    Roots (parent<0) stay at rest: an unlinked fragment is static, its motion unexplained.

    obs_node_pos: [T,S,3]. Returns pos [T,S,3], node_rot [T,S,3,3] (= R_fk @ R0).
    """
    T, S, _ = obs_node_pos.shape
    dev = tree.c0.device
    c0 = tree.c0
    I = torch.eye(3, device=dev).expand(T, 3, 3)
    Rfk = [None] * S
    pos = [None] * S
    for i in tree.order.tolist():
        p = int(tree.parent[i].item())
        if p < 0:
            Rfk[i] = I
            pos[i] = c0[i].expand(T, 3)
            continue
        d0 = c0[i] - c0[p]
        L = d0.norm().clamp_min(1e-9)
        u = (d0 / L).expand(T, 3)
        tgt = obs_node_pos[:, i] - pos[p]
        v = tgt / tgt.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        Rfk[i] = _shortest_arc(u, v)                             # maps rest dir -> observed dir
        pos[i] = pos[p] + torch.einsum('tij,j->ti', Rfk[i], d0)
    pos = torch.stack(pos, 1)
    Rfk = torch.stack(Rfk, 1)
    node_rot = Rfk @ tree.R0[None]
    return pos, node_rot


def fk_from_node_rotations(tree: StprTree, R_obs: torch.Tensor):
    """Connected reconstruction from OBSERVED per-node rotations.

    Under an FK motion the neighborhood-Kabsch rotation of bone i equals its accumulated
    FK rotation R_fk[i]. So we take rotations from observation (robust — averaged over
    hundreds of AppGas, no short-bone angular blowup) and chain POSITIONS through the
    tree:  pos[i] = pos[parent] + R_obs[i] @ (c0[i]-c0[parent]).  Bone lengths are exact
    and children stay anchored to parents — branches cannot break. Roots (parent<0) are
    static at rest; in a broken topology every orphan node is its own root -> frozen.

    R_obs: [T,S,3,3]. Returns pos [T,S,3], node_rot [T,S,3,3] (= R @ R0 for LBS).
    """
    T, S = R_obs.shape[0], R_obs.shape[1]
    dev = tree.c0.device
    c0 = tree.c0
    I = torch.eye(3, device=dev).expand(T, 3, 3)
    Rout = [None] * S
    pos = [None] * S
    for i in tree.order.tolist():
        p = int(tree.parent[i].item())
        if p < 0:
            Rout[i] = I
            pos[i] = c0[i].expand(T, 3)
            continue
        Rout[i] = R_obs[:, i]
        pos[i] = pos[p] + torch.einsum('tij,j->ti', R_obs[:, i], c0[i] - c0[p])
    pos = torch.stack(pos, 1)
    Rout = torch.stack(Rout, 1)
    node_rot = Rout @ tree.R0[None]
    return pos, node_rot


def fit_theta(tree: StprTree, target_pos: torch.Tensor, iters: int = 400, lr: float = 0.05,
              smooth: float = 1e-3, verbose: bool = False):
    """Fit joint angles theta[T,S,3] so FK node positions match an OBSERVED trajectory
    target_pos [T,S,3] (e.g. tracked branch motion). This is the actual Stage-3 motion
    fit: under a broken topology the orphan (rootless) nodes cannot move, so their motion
    stays unexplained; under the repaired topology they can be fit.

    Returns (theta [T,S,3], pos [T,S,3], rot [T,S,3,3], history[list]).
    """
    T, S = target_pos.shape[0], target_pos.shape[1]
    dev = tree.c0.device
    # init off the exp_so3 axis-angle singularity at 0 (0/0 kills gradients there)
    theta = (1e-3 * torch.randn(T, S, 3, device=dev)).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.02)
    hist = []
    for it in range(iters):
        opt.zero_grad()
        pos, _ = fk_batch(tree, theta)
        data = (pos - target_pos).pow(2).sum(-1).mean()
        # temporal smoothness + tiny L2 (L2 removes the unobservable twist-about-bone DOF
        # that otherwise makes the deep-chain IK ill-conditioned)
        reg = (smooth * theta.diff(dim=0).pow(2).mean() if T > 1 else theta.new_zeros(())) \
            + 1e-4 * theta.pow(2).mean()
        loss = data + reg
        loss.backward()
        opt.step(); sched.step()
        if verbose and (it % 100 == 0 or it == iters - 1):
            hist.append((it, float(data.detach().sqrt())))
            print(f'   fit it{it:4d}  rms={float(data.detach().sqrt()):.5f}')
    with torch.no_grad():
        pos, rot = fk_batch(tree, theta.detach())
    return theta.detach(), pos, rot, hist


if __name__ == '__main__':
    import argparse
    from models.structure.strpr_motion import load_strpr_branch

    ap = argparse.ArgumentParser()
    ap.add_argument('--strpr', default='/mnt/data/gaussianplant_data/gsplant_results/strpr_branch.ply')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    dev = torch.device(args.device)
    strpr = load_strpr_branch(args.strpr).to(dev)
    tree = build_strpr_tree(strpr).to(dev)
    print(f'tree: {len(tree.edges)} edges, root={tree.root}, '
          f'max depth reachable={len(tree.order)}/{len(strpr)}')
    theta = torch.zeros(len(strpr), 3, device=dev)
    pos, rot = fk(tree, theta)
    print(f'rest FK: max |pos-c0|={ (pos-strpr.xyz).norm(dim=-1).max():.3e}  '
          f'max |rot-R0|={ (rot-strpr.R).abs().max():.3e}')
