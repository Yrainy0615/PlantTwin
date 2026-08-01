"""Use GaussianPlant Stage-2 structural primitives (StPr) directly as the motion
basis for dynamic refinement (Stage 3), and bind appearance Gaussians (AppGas)
to them with *soft* Gaussian-distance LBS weights.

Why this module exists
----------------------
Earlier binding (models/structure/skinning.py, edge_binding.py) drives AppGas from
the *derived* MST skeleton (branch_mst.ply) with a **hard** nearest-bone assignment.
Stage 3 instead treats each branch StPr — an oriented anisotropic Gaussian written by
GaussianPlant as `strpr_branch.ply` — as a motion node, and binds each AppGas to its
k nearest StPr with weights from the StPr's own Gaussian (Mahalanobis) distance. This
gives a smooth deformation field whose parameters (per-node rigid transforms) are the
only unknowns optimized against the monocular video.

strpr_branch.ply format (confirmed against gsplant_results)
-----------------------------------------------------------
A standard 3DGS PLY `vertex` element, all float32, one row per branch StPr:
    x, y, z                     center (world)
    nx, ny, nz                  normals (unused, zeros)
    f_dc_0..2                   DC SH  -> RGB
    f_rest_0..44                rest SH (deg-3: 15*3)
    opacity                     raw (pre-sigmoid)
    scale_0, scale_1, scale_2   raw log-scale (pre-exp); anisotropic:
                                axis-0 = axial (branch length), axes 1,2 = cross-section
    rot_0..3                    quaternion wxyz (raw, NOT normalized)
    semantic_0..127             DINOv3 semantic feature (128-d)
    label_0                     raw; sigmoid -> branch-vs-leaf confidence
Cylinder geometry (utils/gs_utils.strpr_to_cylinder): R = quat2mat(rot);
axis = R[:,:,0]; radius = exp(scale)[:,2]; height = 3*exp(scale)[:,0].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
@dataclass
class StprBranch:
    xyz: torch.Tensor       # [S, 3] centers (world)
    quat: torch.Tensor      # [S, 4] normalized wxyz
    scale: torch.Tensor     # [S, 3] post-exp (std-dev per local axis)
    R: torch.Tensor         # [S, 3, 3] rotation (columns = local axes)
    label: torch.Tensor     # [S] sigmoid branch confidence
    colors: torch.Tensor    # [S, 3] RGB (from DC SH)
    opacity: torch.Tensor   # [S, 1] post-sigmoid

    @property
    def axis(self) -> torch.Tensor:          # cylinder axis (local x-axis)
        return self.R[:, :, 0]

    @property
    def radius(self) -> torch.Tensor:        # branch thickness
        return self.scale[:, 2]

    @property
    def half_len(self) -> torch.Tensor:      # axial half-length (matches 3*scale0 height/2 ~= 1.5*scale0)
        return 1.5 * self.scale[:, 0]

    def cov(self) -> torch.Tensor:           # [S,3,3] Sigma = R diag(s^2) R^T
        S2 = (self.scale ** 2)
        return self.R @ torch.diag_embed(S2) @ self.R.transpose(-1, -2)

    def to(self, device) -> "StprBranch":
        return StprBranch(*(t.to(device) for t in
                            (self.xyz, self.quat, self.scale, self.R, self.label,
                             self.colors, self.opacity)))

    def __len__(self):
        return self.xyz.shape[0]


_SH_C0 = 0.28209479177387814


def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion (assumed normalized) -> [.,3,3]. Columns are local axes."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.empty(q.shape[0], 3, 3, dtype=q.dtype, device=q.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def load_strpr_branch(path: str | Path) -> StprBranch:
    ply = PlyData.read(str(path))
    v = ply['vertex']
    xyz = torch.tensor(np.stack([v['x'], v['y'], v['z']], -1), dtype=torch.float32)
    quat = torch.tensor(np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], -1),
                        dtype=torch.float32)
    quat = quat / quat.norm(dim=1, keepdim=True).clamp_min(1e-8)
    scale = torch.exp(torch.tensor(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], -1),
                                   dtype=torch.float32))
    R = _quat_to_mat(quat)
    if 'label_0' in v.data.dtype.names:
        label = torch.sigmoid(torch.tensor(np.asarray(v['label_0']), dtype=torch.float32))
    else:
        label = torch.ones(xyz.shape[0])
    dc = torch.tensor(np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], -1), dtype=torch.float32)
    colors = (_SH_C0 * dc + 0.5).clamp(0, 1)
    if 'opacity' in v.data.dtype.names:
        opacity = torch.sigmoid(torch.tensor(np.asarray(v['opacity'])[:, None], dtype=torch.float32))
    else:
        opacity = torch.ones(xyz.shape[0], 1)
    return StprBranch(xyz=xyz, quat=quat, scale=scale, R=R, label=label,
                      colors=colors, opacity=opacity)


# --------------------------------------------------------------------------- #
#  Gaussian-distance LBS binding
# --------------------------------------------------------------------------- #
@dataclass
class StprLBS:
    """Soft binding of AppGas to StPr motion nodes.

    For each AppGas i, `idx[i]` are its k bound StPr and `weight[i]` the normalized
    LBS weights (sum to 1). `local[i,j]` is the AppGas position expressed in bound
    StPr j's *rest* frame:  local = R_k0^T (x_i - c_k0). Deforming a StPr to
    (c_k(t), R_k(t)) moves the AppGas to  c_k(t) + R_k(t) @ local.
    """
    idx: torch.Tensor       # [N, k] long
    weight: torch.Tensor    # [N, k]
    local: torch.Tensor     # [N, k, 3]
    n_nodes: int


def gaussian_lbs_binding(
    ap_xyz: torch.Tensor,
    strpr: StprBranch,
    k: int = 4,
    temperature: float = 20.0,
    max_maha: float | None = None,
    chunk: int = 8192,
) -> StprLBS:
    """Bind each AppGas to its k nearest StPr by *Gaussian (Mahalanobis) distance*.

    d2_ik = (x_i - c_k)^T Sigma_k^{-1} (x_i - c_k),  Sigma_k = R_k diag(s_k^2) R_k^T
    Weights over the k nearest:  w_ik = softmax_k(-d2_ik / (2*temperature)).

    `temperature` sets blend softness. Branch StPr are thin (cross-section std ~0.014),
    so raw Mahalanobis d2 is large and temperature=1 is nearly one-hot. temperature~20
    yields smooth-yet-local blending (~20% of AppGas blend >1 node, top-1 weight ~0.93)
    while the partition-of-unity keeps rest reconstruction exact regardless of value.

    Because a branch StPr is a long thin ellipsoid, the Mahalanobis metric stretches
    influence *along* the branch axis and shrinks it across — exactly the anisotropy
    wanted for skinning AppGas to a branch. Softmax over the top-k keeps the binding
    well-posed even where absolute distances are large (e.g. leaf AppGas far from any
    branch): each AppGas is always partitioned across its k nearest nodes.

    `max_maha`: if set, AppGas whose nearest StPr Mahalanobis distance exceeds this are
    marked static (weight 0, idx=-1) so background survives unmoved.
    """
    device = ap_xyz.device
    strpr = strpr.to(device)
    N = ap_xyz.shape[0]
    S = len(strpr)
    k = min(k, S)

    c = strpr.xyz                                   # [S,3]
    Rk = strpr.R                                    # [S,3,3]
    inv_s2 = 1.0 / (strpr.scale ** 2).clamp_min(1e-12)   # [S,3]

    idx = torch.empty(N, k, dtype=torch.long, device=device)
    d2k = torch.empty(N, k, device=device)
    for s0 in range(0, N, chunk):
        x = ap_xyz[s0:s0 + chunk]                   # [c,3]
        delta = x[:, None, :] - c[None, :, :]       # [c,S,3]
        # local = R_k^T delta  -> project onto each StPr's local axes
        local = torch.einsum('skd,csd->csk', Rk, delta)   # [c,S,3]
        d2 = (local ** 2 * inv_s2[None, :, :]).sum(-1)     # [c,S] Mahalanobis^2
        dk, ik = torch.topk(d2, k, dim=1, largest=False)
        idx[s0:s0 + chunk] = ik
        d2k[s0:s0 + chunk] = dk

    weight = torch.softmax(-d2k / (2.0 * temperature), dim=1)   # [N,k]

    # rest-frame local coords for the k bound nodes
    c_sel = c[idx]                                  # [N,k,3]
    R_sel = Rk[idx]                                 # [N,k,3,3]
    delta_sel = ap_xyz[:, None, :] - c_sel          # [N,k,3]
    local = torch.einsum('nkij,nkj->nki', R_sel.transpose(-1, -2), delta_sel)  # R^T delta

    if max_maha is not None:
        static = d2k[:, 0] > (max_maha ** 2)
        weight = torch.where(static[:, None], torch.zeros_like(weight), weight)
        idx = torch.where(static[:, None], torch.full_like(idx, -1), idx)

    return StprLBS(idx=idx, weight=weight, local=local, n_nodes=S)


def apply_strpr_lbs(
    binding: StprLBS,
    node_pos: torch.Tensor,     # [S,3] current StPr centers
    node_rot: torch.Tensor,     # [S,3,3] current StPr rotations
    ap_xyz_rest: torch.Tensor,  # [N,3] rest positions (fallback for static AppGas)
) -> torch.Tensor:
    """Linear-blend-skin the AppGas by the current StPr motion nodes."""
    idx = binding.idx                               # [N,k]
    w = binding.weight                              # [N,k]
    safe = idx.clamp_min(0)
    c_t = node_pos[safe]                            # [N,k,3]
    R_t = node_rot[safe]                            # [N,k,3,3]
    contrib = c_t + torch.einsum('nkij,nkj->nki', R_t, binding.local)  # [N,k,3]
    out = (w[..., None] * contrib).sum(1)          # [N,3]
    # AppGas with no valid binding (all idx==-1 -> weight 0) stay at rest.
    bound = (idx >= 0).any(1) & (w.sum(1) > 1e-6)
    return torch.where(bound[:, None], out, ap_xyz_rest)


def _mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """[.,3,3] rotation matrix -> wxyz quaternion (normalized). Robust branch-free-ish."""
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    t = 1.0 + m00 + m11 + m22
    q = torch.empty(R.shape[0], 4, dtype=R.dtype, device=R.device)
    # generic case (t>eps); fall back per-element for the rest
    s = torch.sqrt(t.clamp_min(1e-8)) * 2
    q[:, 0] = 0.25 * s
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / s
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / s
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / s
    bad = t <= 1e-6
    if bad.any():                                   # rare: near-pi rotations
        Rb = R[bad]
        diag = torch.stack([Rb[:, 0, 0], Rb[:, 1, 1], Rb[:, 2, 2]], -1)
        i = diag.argmax(-1)
        qb = torch.empty(Rb.shape[0], 4, dtype=R.dtype, device=R.device)
        for n in range(Rb.shape[0]):
            ii = int(i[n]); jj = (ii + 1) % 3; kk = (ii + 2) % 3
            M = Rb[n]
            s_ = torch.sqrt((M[ii, ii] - M[jj, jj] - M[kk, kk] + 1.0).clamp_min(1e-8)) * 2
            qb[n, 0] = (M[kk, jj] - M[jj, kk]) / s_
            qb[n, 1 + ii] = 0.25 * s_
            qb[n, 1 + jj] = (M[jj, ii] + M[ii, jj]) / s_
            qb[n, 1 + kk] = (M[kk, ii] + M[ii, kk]) / s_
        q[bad] = qb
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of wxyz quaternions, [.,4] x [.,4] -> [.,4]."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], -1)


def fit_strpr_rigid(
    binding: StprLBS,
    strpr: StprBranch,          # rest StPr (provides rest center c0, rot R0 per node)
    ap_rest: torch.Tensor,      # [N,3] rest AppGas
    ap_obs: torch.Tensor,       # [T,N,3] OBSERVED AppGas trajectory (from tracked video)
    include: torch.Tensor,      # [S] bool: StPr that are in the articulated model
    min_pts: int = 6,
):
    """Fit each StPr's per-frame rigid transform by Procrustes on the OBSERVED motion of the
    AppGas bound to it (idx[:,0]==i). This is the actual dynamic fit: the per-StPr rigid
    handles are solved from the video's tracked point motion — no ground-truth parameters.
    StPr not in `include` (a missed / unlinked branch) OR with too few bound AppGas stay
    static, so their observed motion is left unexplained.

    Output (node_pos, node_rot) is in the convention apply_strpr_lbs expects:
        world = node_pos + node_rot @ (R0^T (x_rest - c0_strpr)).
    For a cluster rigid motion R(t) about the observed centroid ct(t):
        node_rot = R @ R0 ,  node_pos = ct + R @ (c0_strpr - c0_cluster).
    Excluded/static StPr -> (c0_strpr, R0) => reproduce rest exactly.
    """
    dev = ap_rest.device
    T = ap_obs.shape[0]
    S = binding.n_nodes
    top1 = binding.idx[:, 0]
    node_pos = strpr.xyz[None].expand(T, S, 3).clone()
    node_rot = strpr.R[None].expand(T, S, 3, 3).clone()
    I = torch.eye(3, device=dev)
    for i in range(S):
        m = (top1 == i)
        n = int(m.sum().item())
        if (not bool(include[i])) or n < min_pts:
            continue                                 # keep rest (static)
        P0 = ap_rest[m]                              # [n,3]
        Pt = ap_obs[:, m]                            # [T,n,3]
        c0c = P0.mean(0)                             # cluster rest centroid
        ct = Pt.mean(1)                             # [T,3] cluster observed centroid
        Pc = P0 - c0c
        Qc = Pt - ct[:, None, :]
        H = torch.einsum('ni,tnj->tij', Pc, Qc)
        U, _, Vt = torch.linalg.svd(H)
        d = torch.sign(torch.linalg.det(Vt.transpose(-1, -2) @ U.transpose(-1, -2)))
        D = I.expand(T, 3, 3).clone(); D[:, 2, 2] = d
        R = Vt.transpose(-1, -2) @ D @ U.transpose(-1, -2)   # [T,3,3] rest->frame
        node_rot[:, i] = R @ strpr.R[i]
        node_pos[:, i] = ct + torch.einsum('tij,j->ti', R, strpr.xyz[i] - c0c)
    return node_pos, node_rot


def estimate_node_motion_excl(
    strpr: StprBranch,
    binding: StprLBS,
    ap_rest: torch.Tensor,      # [N,3]
    ap_obs: torch.Tensor,       # [T,N,3] observed (tracked) AppGas trajectory
    edges: torch.Tensor,        # [E,2] graph edges for propagating to sparse nodes
    min_pts: int = 6,
):
    """Per-node observed rotation/trajectory from each StPr's EXCLUSIVE cluster (its top-1
    bound AppGas), which is unbiased — no motion bleed from neighboring branches. Nodes
    with < min_pts bound AppGas (thin branches hidden under foliage) inherit the rotation
    of the graph-nearest observed node (adjacent segments rotate almost identically), so
    every node is covered without cross-branch contamination.

    Returns obs_pos [T,S,3], R_obs [T,S,3,3], observed [S] bool, node_var [S]
    (node_var = estimated variance of the node-trajectory estimate from tracking noise:
    per-point Kabsch residual variance / n_pts — the noise floor any bone-length-variance
    test against this node must be normalized by).
    """
    dev = ap_rest.device
    T = ap_obs.shape[0]
    S = len(strpr)
    top1 = binding.idx[:, 0]
    I = torch.eye(3, device=dev)
    R_obs = I.expand(T, S, 3, 3).clone().reshape(T, S, 3, 3).clone()
    obs_pos = strpr.xyz[None].expand(T, S, 3).clone()
    observed = torch.zeros(S, dtype=torch.bool, device=dev)
    node_var = torch.full((S,), float('inf'), device=dev)
    for i in range(S):
        m = top1 == i
        n = int(m.sum())
        if n < min_pts:
            continue
        P0 = ap_rest[m]
        Pt = ap_obs[:, m]
        c0c = P0.mean(0)
        ct = Pt.mean(1)
        Pc = P0 - c0c
        Qc = Pt - ct[:, None, :]
        H = torch.einsum('ni,tnj->tij', Pc, Qc)
        U, _, Vt = torch.linalg.svd(H)
        dsign = torch.sign(torch.linalg.det(Vt.transpose(-1, -2) @ U.transpose(-1, -2)))
        D = I.expand(T, 3, 3).clone(); D[:, 2, 2] = dsign
        R = Vt.transpose(-1, -2) @ D @ U.transpose(-1, -2)
        R_obs[:, i] = R
        obs_pos[:, i] = ct + torch.einsum('tij,j->ti', R, strpr.xyz[i] - c0c)
        observed[i] = True
        resid = Qc - torch.einsum('tij,nj->tni', R, Pc)       # [T,n,3] tracking noise
        # noise floor of the NODE estimate = centroid noise + rotation noise amplified by
        # the lever arm from cluster centroid to node center (leaf clusters hang far off
        # their branch node, so this term dominates):
        #   var ~ sigma^2/n * (1 + |c_node - c_cluster|^2 / r_gyration^2)
        r_gyr2 = Pc.pow(2).sum(-1).mean().clamp_min(1e-12)
        lever = 1.0 + (strpr.xyz[i] - c0c).pow(2).sum() / r_gyr2
        node_var[i] = resid.pow(2).sum(-1).mean() / (3 * n) * lever
    # propagate to sparse nodes along graph edges (BFS from observed set)
    adj = [[] for _ in range(S)]
    for a, b in edges.tolist():
        adj[a].append(b); adj[b].append(a)
    frontier = [i for i in range(S) if bool(observed[i])]
    seen = observed.clone()
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if not bool(seen[v]):
                    seen[v] = True
                    R_obs[:, v] = R_obs[:, u]
                    obs_pos[:, v] = obs_pos[:, u] + torch.einsum(
                        'tij,j->ti', R_obs[:, u], strpr.xyz[v] - strpr.xyz[u])
                    nxt.append(v)
        frontier = nxt
    return obs_pos, R_obs, observed, node_var


def estimate_node_motion(
    strpr: StprBranch,
    ap_rest: torch.Tensor,      # [N,3]
    ap_obs: torch.Tensor,       # [T,N,3] observed (tracked) AppGas trajectory
    n_nbr: int = 200,
    chunk: int = 8192,
):
    """Estimate every StPr node's observed trajectory + rotation from its NEIGHBORHOOD.

    Exclusive top-1 clusters fail here: leaf AppGas pile onto a few StPr and ~half the
    branch StPr get zero points (unobserved -> frozen -> visibly broken chains). Instead,
    each StPr takes its n_nbr nearest AppGas by Mahalanobis distance (non-exclusive), and
    a per-frame Kabsch on that neighborhood gives a well-averaged rigid (R(t), t(t)).

    Returns obs_pos [T,S,3] (node c0 carried by the local rigid motion) and
    R_obs [T,S,3,3] (accumulated rotation vs rest, ~= R_fk under an FK motion).
    """
    dev = ap_rest.device
    T = ap_obs.shape[0]
    S = len(strpr)
    N = ap_rest.shape[0]
    n_nbr = min(n_nbr, N)
    c = strpr.xyz
    Rk = strpr.R
    inv_s2 = 1.0 / (strpr.scale ** 2).clamp_min(1e-12)

    # per-StPr n_nbr nearest AppGas by Mahalanobis distance (running top-k over chunks)
    best_d = torch.full((S, n_nbr), float('inf'), device=dev)
    best_i = torch.zeros(S, n_nbr, dtype=torch.long, device=dev)
    for s0 in range(0, N, chunk):
        x = ap_rest[s0:s0 + chunk]
        delta = x[:, None, :] - c[None, :, :]
        local = torch.einsum('skd,csd->csk', Rk, delta)
        d2 = (local ** 2 * inv_s2[None, :, :]).sum(-1).T          # [S,chunk]
        cat_d = torch.cat([best_d, d2], dim=1)
        cat_i = torch.cat([best_i, torch.arange(s0, s0 + x.shape[0], device=dev)
                           .expand(S, -1)], dim=1)
        best_d, sel = torch.topk(cat_d, n_nbr, dim=1, largest=False)
        best_i = torch.gather(cat_i, 1, sel)

    obs_pos = torch.zeros(T, S, 3, device=dev)
    R_obs = torch.zeros(T, S, 3, 3, device=dev)
    I = torch.eye(3, device=dev)
    for i in range(S):
        nbr = best_i[i]
        P0 = ap_rest[nbr]                            # [n,3]
        Pt = ap_obs[:, nbr]                          # [T,n,3]
        c0c = P0.mean(0)
        ct = Pt.mean(1)
        Pc = P0 - c0c
        Qc = Pt - ct[:, None, :]
        H = torch.einsum('ni,tnj->tij', Pc, Qc)
        U, _, Vt = torch.linalg.svd(H)
        dsign = torch.sign(torch.linalg.det(Vt.transpose(-1, -2) @ U.transpose(-1, -2)))
        D = I.expand(T, 3, 3).clone(); D[:, 2, 2] = dsign
        R = Vt.transpose(-1, -2) @ D @ U.transpose(-1, -2)
        R_obs[:, i] = R
        obs_pos[:, i] = ct + torch.einsum('tij,j->ti', R, c[i] - c0c)
    return obs_pos, R_obs


def apply_strpr_lbs_full(
    binding: StprLBS,
    strpr_rest: StprBranch,     # rest StPr (for delta rotations)
    node_pos: torch.Tensor,     # [S,3] current StPr centers
    node_rot: torch.Tensor,     # [S,3,3] current StPr rotations
    ap_xyz_rest: torch.Tensor,  # [N,3]
    ap_quat_rest: torch.Tensor, # [N,4] rest Gaussian orientations (wxyz, normalized)
) -> tuple[torch.Tensor, torch.Tensor]:
    """LBS positions AND rotate Gaussian orientations by the dominant (top-1) node's
    delta rotation R_delta = node_rot @ R0^T. Without this, anisotropic Gaussians keep
    their world orientation while their centers move — leaves visually shatter.
    (Rotation uses top-1 only: blending rotations needs quat averaging and top-1 is
    correct for rigid leaf/branch attachment.)
    """
    xyz = apply_strpr_lbs(binding, node_pos, node_rot, ap_xyz_rest)
    top1 = binding.idx[:, 0].clamp_min(0)
    R_delta = node_rot[top1] @ strpr_rest.R[top1].transpose(-1, -2)   # [N,3,3]
    q_delta = _mat_to_quat(R_delta)
    quat = _quat_mul(q_delta, ap_quat_rest)
    bound = (binding.idx >= 0).any(1) & (binding.weight.sum(1) > 1e-6)
    quat = torch.where(bound[:, None], quat, ap_quat_rest)
    return xyz, quat


# --------------------------------------------------------------------------- #
#  Self-test: rest reconstruction must be exact
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--strpr', default='/mnt/data/gaussianplant_data/gsplant_results/strpr_branch.ply')
    ap.add_argument('--appgas', default='/mnt/data/gaussianplant_data/gsplant_results/point_cloud.ply')
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    dev = torch.device(args.device)
    strpr = load_strpr_branch(args.strpr).to(dev)
    print(f'[strpr] {len(strpr)} branch StPr | radius mean={strpr.radius.mean():.4f} '
          f'half_len mean={strpr.half_len.mean():.4f} label mean={strpr.label.mean():.3f}')

    from data.gaussian_plant_loader import _load_app_gaussians
    app = _load_app_gaussians(Path(args.appgas))
    ap_xyz = app.xyz.to(dev)
    print(f'[appgas] {ap_xyz.shape[0]} AppGas')

    binding = gaussian_lbs_binding(ap_xyz, strpr, k=args.k)
    w = binding.weight
    print(f'[bind] k={args.k}  top-1 weight mean={w[:,0].mean():.3f}  '
          f'entropy mean={(-(w.clamp_min(1e-9)*w.clamp_min(1e-9).log()).sum(1)).mean():.3f}')

    # rest reconstruction: identity motion -> must reproduce ap_xyz
    recon = apply_strpr_lbs(binding, strpr.xyz, strpr.R, ap_xyz)
    err = (recon - ap_xyz).norm(dim=-1)
    print(f'[rest] max recon err={err.max():.3e}  mean={err.mean():.3e}')
