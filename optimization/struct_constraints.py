"""Shape + topology constraints for structure-in-the-loop optimization.

When the branch tree's rest geometry becomes a learnable variable (`delta_rest_pos`)
and densification inserts new midpoint nodes, two failure modes appear:

  1. **Nodes drift out of the branch volume** — `delta` is free to push a node into
     empty space because the motion-fit residual only weakly localizes it. The
     densified midpoints are especially prone to this.
  2. **Neighboring nodes move incoherently** — a node (often a freshly inserted one)
     can introduce a local "kink", bending opposite to its neighbors, which is not
     physically plausible for a continuous branch.

This module supplies two differentiable regularizers driven by the *dense* branch
geometry (the GaussianPlant tube-surface point cloud), which the skeleton nodes
alone do not provide:

  - `geometric_containment_loss` — keep every (perturbed / new) node inside the
    branch material: a one-sided hinge on the distance from each node to the nearest
    dense branch point, active only beyond a tolerance band `r_tol`. Nodes are free
    to *slide along* the branch (zero penalty) but are pulled back once they exit.
  - `motion_direction_consistency_loss` — adjacent nodes along the tree should move
    in consistent directions: a magnitude-gated cosine penalty on the per-frame
    displacement vectors of each (parent, child) pair.

Both are pure tensor ops with no dependency on the big optimizer model, so they can
be unit-tested in isolation.
"""

from __future__ import annotations

import torch


def subsample_points(points: torch.Tensor, max_points: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Uniformly subsample a point cloud to at most `max_points` rows (no replacement)."""
    V = points.shape[0]
    if V <= max_points:
        return points
    idx = torch.randperm(V, generator=generator, device=points.device)[:max_points]
    return points[idx]


def estimate_geom_margin(
    orig_nodes: torch.Tensor,
    branch_points: torch.Tensor,
    quantile: float = 0.95,
    scale: float = 2.0,
) -> float:
    """Adaptive tolerance band `r_tol` from the original (valid) skeleton.

    The original nodes already sit inside the branch, so their nearest-branch-point
    distances define what "inside" looks like for this scene. We take a high quantile
    (robust to a few outliers like the pot/root) and inflate by `scale` for slack.
    """
    d = torch.cdist(orig_nodes, branch_points).min(dim=1).values
    return float(d.quantile(quantile).item() * scale)


def rest_relative_tolerance(
    rest_nodes: torch.Tensor,
    branch_points: torch.Tensor,
    global_r_tol: float,
    margin: float = 0.02,
    chunk: int = 4096,
) -> torch.Tensor:
    """Per-node tolerance band for the containment hinge.

    A node already far from the (subsampled / incomplete) tube cloud at rest — e.g.
    the pot/root base or a leaf-region node the tube mesh doesn't cover — should not
    be forced *into* the branch; we only want to stop `delta` from pushing it *farther
    out* than it started. So the per-node tolerance is

        tol_i = max(global_r_tol, d_nn_rest(i) + margin).

    Inside nodes (d_rest << r_tol) are held to the global band; structurally-outside
    nodes get a band around their own rest distance plus a small slack.
    """
    d_rest = nearest_branch_distance(rest_nodes, branch_points, chunk=chunk)
    return torch.clamp(d_rest + margin, min=global_r_tol)


def nearest_branch_distance(node_pos: torch.Tensor, branch_points: torch.Tensor,
                            chunk: int = 4096) -> torch.Tensor:
    """Distance from each node to the nearest dense branch point. [N] (differentiable).

    Chunked over branch points to bound peak memory for large clouds.
    """
    N = node_pos.shape[0]
    best = node_pos.new_full((N,), float('inf'))
    for s in range(0, branch_points.shape[0], chunk):
        bp = branch_points[s:s + chunk]                       # [C, 3]
        d = torch.cdist(node_pos, bp)                         # [N, C]
        best = torch.minimum(best, d.min(dim=1).values)
    return best


def geometric_containment_loss(
    node_pos: torch.Tensor,
    branch_points: torch.Tensor,
    r_tol: float | torch.Tensor,
    chunk: int = 4096,
) -> tuple[torch.Tensor, dict]:
    """One-sided hinge keeping nodes within `r_tol` of the dense branch cloud.

    L_geom = mean_i ( max(0, d_nn(i) - r_tol) )^2

    `r_tol` may be a scalar (global band) or a per-node tensor (e.g. from
    `rest_relative_tolerance`). Returns (loss, info) with `d_nn`, `n_outside`,
    `max_excess` for logging.
    """
    d_nn = nearest_branch_distance(node_pos, branch_points, chunk=chunk)   # [N]
    excess = (d_nn - r_tol).clamp_min(0.0)
    # Active-set mean: normalize by the number of violators, not by N. Averaging over
    # all nodes would dilute each violator's gradient by the (large) inside population,
    # leaving the constraint inert. Dividing by the active count keeps per-violator
    # gradient undiluted and self-reinforcing — as violators are pulled in, the few
    # remaining ones feel a stronger restoring force.
    n_out = (excess > 0).sum()
    loss = (excess ** 2).sum() / n_out.clamp(min=1)
    info = {
        'd_nn': d_nn.detach(),
        'n_outside': int(n_out.item()),
        'max_excess': float(excess.max().item()) if excess.numel() else 0.0,
    }
    return loss, info


def motion_direction_consistency_loss(
    node_pos_traj: torch.Tensor,
    rest_pos: torch.Tensor,
    edges: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Magnitude-gated cosine penalty on neighbor displacement directions.

    For each oriented edge (p, c) and frame t, with displacement
        m_i(t) = pos(t, i) - rest_pos(i),
    penalize 1 - cos(m_p, m_c), weighted by ||m_p|| * ||m_c|| so that frames/edges
    with negligible motion (undefined direction) do not contribute.

        L_motion = sum_{t,(p,c)} w * (1 - cos) / sum_{t,(p,c)} w
    """
    if edges.numel() == 0:
        return node_pos_traj.new_zeros(())
    m = node_pos_traj - rest_pos.unsqueeze(0)                 # [T, N, 3]
    p = edges[:, 0]
    c = edges[:, 1]
    a = m[:, p]                                               # [T, E, 3]
    b = m[:, c]
    na = a.norm(dim=-1)                                       # [T, E]
    nb = b.norm(dim=-1)
    w = na * nb
    cos = (a * b).sum(dim=-1) / (w + eps)
    num = (w * (1.0 - cos)).sum()
    den = w.sum() + eps
    return num / den
