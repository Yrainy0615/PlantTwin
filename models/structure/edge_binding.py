"""Hybrid geometric binding of AppGas to a branch skeleton.

Branch-near AppGas are bound to a branch *edge* (segment a->b) and parameterized in the
segment's local frame; far AppGas (leaves) fall back to rigid attachment to the nearest
*node*. Both reproduce the rest pose exactly, and both move when their node(s) move, so
skeleton geometry is observable in the render (multi-view / video RGB can constrain it).

Edge binding (branch AppGas):
    d = P[b]-P[a];  u = d/|d|;  e1 = norm(u x ref);  e2 = u x e1
    world = P[a] + t*d + ou*u + p1*e1 + p2*e2          (ou,p1,p2 fixed offsets)
  -> moving/rotating the edge stretches & rotates the AppGas about the branch axis.

Node binding (leaf AppGas):
    world = P[n] + local                                (local fixed)
  -> the whole leaf cluster rides its parent node rigidly.

The binding is computed once from a reference skeleton and held fixed; node positions
(rest or per-frame from FK) are the variable. `reconstruct` / `reconstruct_traj` rebuild
AppGas world positions.
"""

from __future__ import annotations

import torch


def _edge_frame(d: torch.Tensor, eps: float = 1e-8):
    L = d.norm(dim=-1, keepdim=True).clamp_min(eps)
    u = d / L
    ref = torch.zeros_like(u); ref[..., 2] = 1.0
    parallel = (u[..., 2].abs() > 0.99)
    ref2 = torch.zeros_like(u); ref2[..., 0] = 1.0
    ref = torch.where(parallel.unsqueeze(-1), ref2, ref)
    e1 = torch.cross(u, ref, dim=-1)
    e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(eps)
    e2 = torch.cross(u, e1, dim=-1)
    return u, e1, e2, L.squeeze(-1)


def build_edge_binding(ap_xyz: torch.Tensor, nodes: torch.Tensor, edges: torch.Tensor,
                       branch_thresh: float = 0.3, chunk: int = 100000,
                       leaf_centroids: torch.Tensor | None = None):
    """Assign AppGas to nearest edge (branch) or to a rigid leaf instance (leaf).

    `branch_thresh`: perpendicular distance below which an AppGas is treated as a
    branch point bound to its edge; otherwise it is a leaf point.

    Leaf binding (the key to letting abundant leaf motion constrain branch structure):
    if `leaf_centroids` [K,3] is given, each leaf AppGas is assigned to its nearest leaf
    instance, and the whole instance is attached to the single nearest branch node. Under
    motion the instance rotates *rigidly* with that node's FK frame, so leaf swing is
    driven by — and therefore constrains — the parent branch node. Without
    `leaf_centroids`, leaves fall back to per-point rigid attachment to the nearest node.

    Returns a dict of fixed tensors on ap_xyz.device.
    """
    device = ap_xyz.device
    a = nodes[edges[:, 0]]; b = nodes[edges[:, 1]]
    d = b - a
    L2 = (d * d).sum(-1).clamp_min(1e-12)
    M = ap_xyz.shape[0]
    best_edge = torch.empty(M, dtype=torch.long, device=device)
    best_t = torch.empty(M, device=device)
    best_d2 = torch.full((M,), float('inf'), device=device)
    for s in range(0, M, chunk):
        p = ap_xyz[s:s + chunk]
        ap_ = p.unsqueeze(1) - a.unsqueeze(0)
        t = (ap_ * d.unsqueeze(0)).sum(-1) / L2.unsqueeze(0)
        t = t.clamp(0.0, 1.0)
        proj = a.unsqueeze(0) + t.unsqueeze(-1) * d.unsqueeze(0)
        dist2 = ((p.unsqueeze(1) - proj) ** 2).sum(-1)
        mind2, mine = dist2.min(dim=1)
        best_edge[s:s + chunk] = mine
        best_t[s:s + chunk] = t[torch.arange(t.shape[0], device=device), mine]
        best_d2[s:s + chunk] = mind2

    res = best_d2.sqrt()
    is_branch = res < branch_thresh

    # edge-frame offsets for branch AppGas (full 3D so reconstruction is exact)
    ea = nodes[edges[best_edge, 0]]; eb = nodes[edges[best_edge, 1]]
    u, e1, e2, L = _edge_frame(eb - ea)
    axis_pt = ea + best_t.unsqueeze(-1) * (eb - ea)
    off = ap_xyz - axis_pt
    ou = (off * u).sum(-1)
    p1 = (off * e1).sum(-1)
    p2 = (off * e2).sum(-1)

    # --- leaf binding: group leaf AppGas into instances, attach instance to a node ---
    with torch.no_grad():
        inst = torch.full((M,), -1, dtype=torch.long, device=device)
        if leaf_centroids is not None and leaf_centroids.shape[0] > 0:
            lc = leaf_centroids.to(device)
            for s in range(0, M, chunk):
                dd = torch.cdist(ap_xyz[s:s + chunk], lc)
                inst[s:s + chunk] = dd.argmin(1)
            K = lc.shape[0]
            # each instance attaches to the branch node nearest its centroid
            attach_of_inst = torch.cdist(lc, nodes).argmin(1)            # [K]
            attach_node = attach_of_inst[inst]                           # [M]
        else:
            # fallback: per-point nearest node
            attach_node = torch.empty(M, dtype=torch.long, device=device)
            for s in range(0, M, chunk):
                dd = torch.cdist(ap_xyz[s:s + chunk], nodes)
                attach_node[s:s + chunk] = dd.argmin(1)
    local = ap_xyz - nodes[attach_node]                                  # rest-world offset

    return {'edge': best_edge, 't': best_t, 'ou': ou, 'p1': p1, 'p2': p2,
            'is_branch': is_branch, 'node': attach_node, 'local': local,
            'inst': inst, 'res': res}


def reconstruct(nodes: torch.Tensor, edges: torch.Tensor, binding: dict) -> torch.Tensor:
    """World positions of all bound AppGas given current node positions [N,3]."""
    e = binding['edge']
    a = nodes[edges[e, 0]]; b = nodes[edges[e, 1]]
    d = b - a
    u, e1, e2, L = _edge_frame(d)
    world_edge = (a + binding['t'].unsqueeze(-1) * d + binding['ou'].unsqueeze(-1) * u
                  + binding['p1'].unsqueeze(-1) * e1 + binding['p2'].unsqueeze(-1) * e2)
    world_node = nodes[binding['node']] + binding['local']
    return torch.where(binding['is_branch'].unsqueeze(-1), world_edge, world_node)


def reconstruct_traj(pos_t: torch.Tensor, edges: torch.Tensor, binding: dict,
                     rot_t: torch.Tensor | None = None) -> torch.Tensor:
    """Per-frame AppGas world positions. pos_t [T,N,3] -> [T,M,3].

    `rot_t` [T,N,3,3]: if given, leaf instances rotate rigidly with their attachment
    node's frame (leaf swing). Without it, leaves only translate with the node.
    """
    e = binding['edge']
    a = pos_t[:, edges[e, 0]]; b = pos_t[:, edges[e, 1]]
    d = b - a
    u, e1, e2, L = _edge_frame(d)
    world_edge = (a + binding['t'].view(1, -1, 1) * d + binding['ou'].view(1, -1, 1) * u
                  + binding['p1'].view(1, -1, 1) * e1 + binding['p2'].view(1, -1, 1) * e2)
    attach = binding['node']
    if rot_t is not None:
        R = rot_t[:, attach]                                   # [T,M,3,3]
        local = binding['local'].unsqueeze(0).unsqueeze(-1)    # [1,M,3,1]
        world_node = pos_t[:, attach] + (R @ local).squeeze(-1)
    else:
        world_node = pos_t[:, attach] + binding['local'].unsqueeze(0)
    return torch.where(binding['is_branch'].view(1, -1, 1), world_edge, world_node)
