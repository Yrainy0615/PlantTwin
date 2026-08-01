"""Per-edge motion non-rigidity residual — the densification signal for fusion.

A true branch segment is locally rigid: under bending, all the AppGas bound to one
edge move as a single rigid body. An edge that spans a *bend* the current (coarse)
topology cannot articulate has bound points whose observed motion is NOT explained by
any single rigid transform — its non-rigidity residual is high. That residual is
exactly "this edge needs another joint," and it needs no joint angles (theta): it is
computed purely from the GT-observed AppGas trajectory + the edge binding.

After splitting such an edge at its midpoint, each half spans a shorter, straighter
arc, so the per-edge residual drops — the measurable densification benefit.
"""

from __future__ import annotations

import torch

from models.structure.graph_cleanup import RootedBranchTree


def _kabsch_residual(P: torch.Tensor, Q_t: torch.Tensor) -> torch.Tensor:
    """Residual of the best rigid fit aligning ref points P [n,3] to each frame Q_t [T,n,3].

    Returns the per-frame mean point error after the optimal rigid transform, averaged
    over frames (scalar). Zero iff every frame is a rigid motion of P.
    """
    Pc = P - P.mean(0, keepdim=True)                       # [n,3]
    Qc = Q_t - Q_t.mean(1, keepdim=True)                   # [T,n,3]
    H = torch.einsum('ni,tnj->tij', Pc, Qc)                # [T,3,3]
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.transpose(-1, -2) @ U.transpose(-1, -2)))
    D = torch.eye(3, device=P.device).expand(H.shape[0], 3, 3).clone()
    D[:, 2, 2] = d
    R = Vt.transpose(-1, -2) @ D @ U.transpose(-1, -2)     # [T,3,3], maps Pc -> Qc
    pred = torch.einsum('tij,nj->tni', R, Pc)              # [T,n,3]
    # Per-point error, aggregated by a quantile rather than the mean: edges near
    # branchings grab a few gaussians from SIBLING branches whose motion differs
    # (binding bleed) — a minority that would otherwise fake non-rigidity forever.
    # A true bend displaces the majority of points, so the q60 survives it.
    per_pt = (Qc - pred).norm(dim=-1).mean(0)              # [n]
    return per_pt.quantile(0.6)


def edge_nonrigidity(tree: RootedBranchTree, binding: dict, gt_ap_world_traj: torch.Tensor,
                     max_pts: int = 200, min_pts: int = 6) -> torch.Tensor:
    """Per-edge non-rigidity residual [E] from GT-observed branch AppGas motion.

    gt_ap_world_traj: [T, M, 3] world trajectory of all AppGas (from the FULL tree).
    Only branch-bound AppGas (binding['is_branch']) of each coarse edge are used.
    Edges with < min_pts bound branch points get residual 0 (cannot/should not split).
    """
    device = gt_ap_world_traj.device
    E = tree.edges_oriented.shape[0]
    edge = binding['edge'].to(device)
    is_branch = binding['is_branch'].to(device)
    # Drop junction-ambiguous points: near a shared node the cylinders of the two
    # incident edges overlap, assignment is a coin flip on row order / float noise,
    # and a regrouped chunk carries the NEIGHBOR edge's rigid motion — reading as
    # several cm of fake non-rigidity. Clamped-t points are exactly those.
    t_in = (binding['t'].to(device) > 0.05) & (binding['t'].to(device) < 0.95)
    res = torch.zeros(E, device=device)
    g = torch.Generator(device='cpu').manual_seed(0)
    for e in range(E):
        sel = ((edge == e) & is_branch & t_in).nonzero(as_tuple=False).flatten()
        if sel.numel() < min_pts:
            continue
        if sel.numel() > max_pts:
            idx = torch.randperm(sel.numel(), generator=g)[:max_pts]
            sel = sel[idx.to(device)]
        traj = gt_ap_world_traj[:, sel]                    # [T, n, 3]
        res[e] = _kabsch_residual(traj[0], traj)
    return res


def edge_fk_residual(tree: RootedBranchTree, binding: dict, gt_ap_world_traj: torch.Tensor,
                     pred_ap_world_traj: torch.Tensor, min_pts: int = 6) -> torch.Tensor:
    """Per-edge FK-unexplained motion [E]: onset of |FK prediction - GT| along the tree.

    Local Kabsch (edge_nonrigidity) only sees an edge's own bound points, so a
    hidden TWIST joint (rotation axis ~ edge direction) leaves a residual of only
    radius*theta there while displacing its whole subtree by leverarm*theta. With
    the structural tree in GT coordinates and theta prescribed, the FK-predicted
    trajectory is computable, and |pred - GT| per gaussian measures exactly the
    motion the current articulation cannot explain — twists and lever arms
    included. A missing joint poisons all DESCENDANT edges equally, so the raw
    per-edge mean is high for the whole subtree; subtracting the parent edge's
    mean (onset differencing) localizes the responsible edge.
    """
    device = gt_ap_world_traj.device
    E = tree.edges_oriented.shape[0]
    edge = binding['edge'].to(device)
    is_branch = binding['is_branch'].to(device)
    err = pred_ap_world_traj - gt_ap_world_traj                            # [T,M,3]
    score = torch.zeros(E, device=device)
    for e in range(E):
        sel = ((edge == e) & is_branch).nonzero(as_tuple=False).flatten()
        if sel.numel() < min_pts:
            continue
        d = err[:, sel]                                                    # [T,n,3]
        # Within-edge spread of the error field. The edge HIDING the joint has a
        # bimodal error (correct below the joint, displaced above) -> large spread;
        # a descendant edge rides the displacement near-rigidly -> small spread.
        score[e] = (d - d.mean(1, keepdim=True)).norm(dim=-1).pow(2).mean(1).sqrt().mean()
    return score


def pick_split_edges_by_motion(tree: RootedBranchTree, residual: torch.Tensor, top_k: int,
                               min_edge_length: float | None = None) -> list[int]:
    """Top-K edges by non-rigidity residual, skipping edges shorter than min_edge_length."""
    score = residual.clone()
    if min_edge_length is not None:
        score[tree.edge_length.to(score.device) < min_edge_length] = -1.0
    valid = (score > 0).nonzero(as_tuple=False).flatten()
    if valid.numel() == 0:
        return []
    k = min(top_k, valid.numel())
    top = torch.topk(score[valid], k, largest=True).indices
    return valid[top].tolist()
