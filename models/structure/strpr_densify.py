"""Motion-driven StPr densification, constrained by GaussianPlant's appearance + binding.

Where the observed motion cannot be explained by a StPr's single rigid transform (its
bound AppGas bend within it -> high non-rigidity residual), the StPr is UNDER-articulated
and is split along its axis into two child cylinders. Splitting alone would drift the
children off the geometry, so each split is followed by a short re-optimization under the
SAME constraints GaussianPlant trains StPr with:
  - binding  : each child cylinder must fit the AppGas bound to it (Gaussian NLL — pulls
               the ellipsoid onto its points; the logdet term stops it collapsing).
  - appearance: the StPr set, rendered as Gaussians, must still match the AppGas render
               (GaussianPlant's photometric teacher), so the split preserves appearance.
"""

from __future__ import annotations

import torch

from models.structure.strpr_motion import StprBranch, _quat_to_mat


def _mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    from models.structure.strpr_motion import _mat_to_quat as m2q
    return m2q(R)


def split_strpr(strpr: StprBranch, idx: list[int]):
    """Split each StPr in `idx` along its axis into two half-length children.

    Child k of parent p: center = c_p +/- (h_p/2) * axis_p, axial scale halved, cross
    section kept; rot/color/label/opacity inherited. Returns (new_strpr, info) where
    info['children'][j] = (base_new_idx, tip_new_idx) for the j-th split parent, and
    info['kept'] maps old->new index for un-split StPr.
    """
    dev = strpr.xyz.device
    S = len(strpr)
    idx_set = set(idx)
    keep = [i for i in range(S) if i not in idx_set]
    kept_map = {old: new for new, old in enumerate(keep)}

    xyz, quat, scale, R = strpr.xyz, strpr.quat, strpr.scale, strpr.R
    axis = strpr.axis                                    # [S,3]
    half = strpr.half_len                                # [S]

    new_xyz = [xyz[keep]]
    new_quat = [quat[keep]]
    new_scale = [scale[keep]]
    new_label = [strpr.label[keep]]
    new_col = [strpr.colors[keep]]
    new_op = [strpr.opacity[keep]]
    children = []
    n = len(keep)
    for p in idx:
        a = axis[p]
        off = 0.5 * half[p] * a
        c_base = (xyz[p] - off)[None]
        c_tip = (xyz[p] + off)[None]
        s = scale[p].clone(); s[0] = s[0] * 0.5
        s = s[None].expand(2, 3)
        new_xyz.append(torch.cat([c_base, c_tip], 0))
        new_quat.append(quat[p][None].expand(2, 4))
        new_scale.append(s)
        new_label.append(strpr.label[p][None].expand(2))
        new_col.append(strpr.colors[p][None].expand(2, 3))
        new_op.append(strpr.opacity[p][None].expand(2, 1))
        children.append((n, n + 1)); n += 2

    q = torch.cat(new_quat, 0)
    ns = StprBranch(
        xyz=torch.cat(new_xyz, 0), quat=q, scale=torch.cat(new_scale, 0),
        R=_quat_to_mat(q / q.norm(dim=1, keepdim=True).clamp_min(1e-8)),
        label=torch.cat(new_label, 0), colors=torch.cat(new_col, 0),
        opacity=torch.cat(new_op, 0))
    return ns, {'children': children, 'kept': kept_map}


def gaussian_nll(ap_xyz: torch.Tensor, center: torch.Tensor, R: torch.Tensor,
                 scale: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood of points under an anisotropic Gaussian (up to const).

    0.5 * maha^2 + 0.5 * sum(log scale^2). Minimizing over (center,R,scale) is max-likelihood
    ellipsoid fit: the Mahalanobis term pulls the cylinder onto the points; the logdet term
    keeps it from collapsing to a sheet/line to cheat the Mahalanobis distance.
    """
    delta = ap_xyz - center                              # [n,3]
    local = delta @ R                                    # project onto local axes (R cols)
    inv_s2 = 1.0 / (scale ** 2).clamp_min(1e-10)
    maha2 = (local ** 2 * inv_s2).sum(-1)                # [n]
    logdet = torch.log((scale ** 2).clamp_min(1e-10)).sum()
    return 0.5 * maha2.mean() + 0.5 * logdet


def render_strpr_gaussians(rnd, strpr: StprBranch, cam):
    """Render the StPr themselves as Gaussians (for the appearance/photometric constraint)."""
    q = strpr.quat / strpr.quat.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return rnd.render_frame(strpr.xyz, strpr.scale, q, strpr.opacity, strpr.colors,
                            cam, shs=None).clamp(0, 1)
