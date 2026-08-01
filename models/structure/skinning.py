"""Bind appearance Gaussians (ApP) to either a bone or a leaf, then apply LBS.

Binding rule (per ApP, computed once at canonical pose):
- Distance to each bone's surface = perpendicular dist to bone centerline minus bone radius
- Distance to each leaf's surface ~= dist to disk center minus disk radius
- ApP is bound to whichever (bone vs. leaf) wins on signed surface distance
- The bone vs. leaf assignment is mutually exclusive; rendering at canonical pose
  with these bindings must reproduce the original world position exactly

LBS at runtime:
- Bone-bound ApP world pos = node_pos[parent_of_bone] + R_bone @ local_coord
  where R_bone is the propagated rotation of the bone (== node_rot[child_of_bone])
- Leaf-bound ApP world pos = leaf_center + R_leaf @ local_coord
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.structure.graph_cleanup import RootedBranchTree
from models.structure.leaf_attachment import LeafAttachment


@dataclass
class BoneSkinning:
    bone_idx: torch.Tensor   # [N_ap] long; -1 if ApP is leaf-bound
    leaf_idx: torch.Tensor   # [N_ap] long; -1 if ApP is bone-bound
    local_bone: torch.Tensor # [N_ap, 3] in bone's parent-node-anchored frame at rest
    local_leaf: torch.Tensor # [N_ap, 3] in leaf disk frame at rest


def _point_to_segment_distance(points: torch.Tensor, p: torch.Tensor, q: torch.Tensor):
    """For each point, distance to each segment p->q (clamped to endpoints).

    Returns (dist [N, E], t [N, E]).
    """
    seg = q - p
    seg_len2 = (seg * seg).sum(-1).clamp_min(1e-12)
    diff = points[:, None, :] - p[None, :, :]
    t = (diff * seg[None, :, :]).sum(-1) / seg_len2[None, :]
    t_clamped = t.clamp(0.0, 1.0)
    closest = p[None, :, :] + t_clamped[..., None] * seg[None, :, :]
    d = (points[:, None, :] - closest).norm(dim=-1)
    return d, t_clamped


def compute_skinning(
    ap_xyz: torch.Tensor,
    tree: RootedBranchTree,
    attachments: list[LeafAttachment],
    chunk: int = 4096,
    max_dist: float | None = None,
) -> BoneSkinning:
    """If `max_dist` is given, any ApP whose nearest bone-surface AND nearest
    leaf-surface distance both exceed `max_dist` is marked static (bone_idx=-1,
    leaf_idx=-1) — `apply_skinning` leaves these at their rest position. Use
    this to keep background / pot Gaussians from being dragged with the plant.
    """
    N_ap = ap_xyz.shape[0]
    p = tree.nodes[tree.edges_oriented[:, 0]]
    q = tree.nodes[tree.edges_oriented[:, 1]]
    edge_radius = tree.edge_radius

    nearest_bone_surf = torch.full((N_ap,), float('inf'))
    nearest_bone = torch.zeros(N_ap, dtype=torch.long)

    leaf_centers = torch.stack([a.disk_center for a in attachments]) if attachments else None
    leaf_radii = torch.tensor([a.disk_radius for a in attachments]) if attachments else None
    nearest_leaf_surf = torch.full((N_ap,), float('inf'))
    nearest_leaf = torch.zeros(N_ap, dtype=torch.long)

    for s in range(0, N_ap, chunk):
        pts = ap_xyz[s:s + chunk]
        d, _ = _point_to_segment_distance(pts, p, q)
        d_surf = (d - edge_radius[None, :]).clamp_min(0.0)
        best = d_surf.min(dim=1)
        nearest_bone_surf[s:s + chunk] = best.values
        nearest_bone[s:s + chunk] = best.indices

        if leaf_centers is not None:
            dl = (pts[:, None, :] - leaf_centers[None, :, :]).norm(dim=-1)
            dl_surf = (dl - leaf_radii[None, :]).clamp_min(0.0)
            best_l = dl_surf.min(dim=1)
            nearest_leaf_surf[s:s + chunk] = best_l.values
            nearest_leaf[s:s + chunk] = best_l.indices

    use_leaf = nearest_leaf_surf < nearest_bone_surf
    bone_idx = torch.where(use_leaf, torch.full_like(nearest_bone, -1), nearest_bone)
    leaf_idx = torch.where(use_leaf, nearest_leaf, torch.full_like(nearest_leaf, -1))

    if max_dist is not None:
        static = (nearest_bone_surf > max_dist) & (nearest_leaf_surf > max_dist)
        bone_idx = torch.where(static, torch.full_like(bone_idx, -1), bone_idx)
        leaf_idx = torch.where(static, torch.full_like(leaf_idx, -1), leaf_idx)
        print(f'[skinning] static (background) ApPs: {static.sum().item()}/{static.numel()} '
              f'(max_dist={max_dist})')

    parent_pos = tree.nodes[tree.edges_oriented[:, 0]]
    safe_bone_idx = bone_idx.clamp_min(0)
    local_bone = ap_xyz - parent_pos[safe_bone_idx]
    local_bone = torch.where(use_leaf[:, None], torch.zeros_like(local_bone), local_bone)

    if leaf_centers is not None:
        safe_leaf_idx = leaf_idx.clamp_min(0)
        local_leaf = ap_xyz - leaf_centers[safe_leaf_idx]
        local_leaf = torch.where(use_leaf[:, None], local_leaf, torch.zeros_like(local_leaf))
    else:
        local_leaf = torch.zeros_like(ap_xyz)

    return BoneSkinning(
        bone_idx=bone_idx,
        leaf_idx=leaf_idx,
        local_bone=local_bone,
        local_leaf=local_leaf,
    )


def apply_skinning(
    skinning: BoneSkinning,
    node_pos: torch.Tensor,         # [N_b, 3]
    node_rot: torch.Tensor,         # [N_b, 3, 3]
    edges_oriented: torch.Tensor,   # [N_e, 2]
    leaf_pos: torch.Tensor | None,  # [N_l, 3]
    leaf_rot: torch.Tensor | None,  # [N_l, 3, 3]
    ap_xyz_rest: torch.Tensor,
) -> torch.Tensor:
    """Return current ApP world positions given the current bone & leaf frames."""
    out = ap_xyz_rest.clone()

    bone_mask = skinning.bone_idx >= 0
    if bone_mask.any():
        bidx = skinning.bone_idx[bone_mask]
        parent_idx = edges_oriented[bidx, 0]
        child_idx = edges_oriented[bidx, 1]
        R = node_rot[child_idx]
        parent_p = node_pos[parent_idx]
        local = skinning.local_bone[bone_mask]
        out = out.clone()
        out[bone_mask] = parent_p + (R @ local.unsqueeze(-1)).squeeze(-1)

    leaf_mask = skinning.leaf_idx >= 0
    if leaf_mask.any() and leaf_pos is not None and leaf_rot is not None:
        lidx = skinning.leaf_idx[leaf_mask]
        R = leaf_rot[lidx]
        center = leaf_pos[lidx]
        local = skinning.local_leaf[leaf_mask]
        out = out.clone()
        out[leaf_mask] = center + (R @ local.unsqueeze(-1)).squeeze(-1)

    return out


def skinning_summary(skinning: BoneSkinning) -> str:
    n_bone = int((skinning.bone_idx >= 0).sum().item())
    n_leaf = int((skinning.leaf_idx >= 0).sum().item())
    return f'BoneSkinning: {n_bone} bone-bound ApPs, {n_leaf} leaf-bound ApPs (total {n_bone + n_leaf})'


if __name__ == '__main__':
    import argparse
    from data.gaussian_plant_loader import load_gaussian_plant_scene
    from models.structure.graph_cleanup import root_branch_graph
    from models.structure.leaf_attachment import infer_leaf_attachments

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    parser.add_argument('--output', default='outputs/gsplant_output/newplant9')
    args = parser.parse_args()

    scene = load_gaussian_plant_scene(args.source, args.output)
    tree = root_branch_graph(scene.branch, scene.tube)
    attachments = infer_leaf_attachments(scene.leaves, tree)
    skinning = compute_skinning(scene.app.xyz, tree, attachments)
    print(skinning_summary(skinning))

    # Canonical-pose sanity check: applying skinning at rest must reproduce ap_xyz.
    N_b = tree.nodes.shape[0]
    I_b = torch.eye(3).expand(N_b, 3, 3).contiguous()
    N_l = len(attachments)
    I_l = torch.eye(3).expand(N_l, 3, 3).contiguous()
    leaf_centers = torch.stack([a.disk_center for a in attachments])
    recovered = apply_skinning(
        skinning,
        node_pos=tree.nodes,
        node_rot=I_b,
        edges_oriented=tree.edges_oriented,
        leaf_pos=leaf_centers,
        leaf_rot=I_l,
        ap_xyz_rest=scene.app.xyz,
    )
    err = (recovered - scene.app.xyz).norm(dim=-1).max().item()
    print(f'canonical-pose max ApP reconstruction error: {err:.6e}')
