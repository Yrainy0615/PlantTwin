"""Infer how each leaf cluster attaches to the rooted branch tree.

For every leaf cluster $L_k$:
1. Fit a disk (PCA): centroid $c_k$, normal $n_k$, in-plane radius $r_k$.
2. Identify the leaf-root candidate $\ell_k$ — the leaf-cluster point closest to
   any branch node (typical petiole side of the leaf).
3. Score every branch edge by distance from $\ell_k$ to the edge's cylinder
   surface, with a small terminality bonus for `subtree_size == 1` children.
4. Pick the argmin edge, snap the attachment point onto the cylinder surface,
   and read out (parent edge, along-edge u, rest length, rest direction).

Hungarian global assignment is intentionally skipped for v0 — argmin-per-leaf is
usually fine on a plant scene; revisit if duplicates pile up.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from data.gaussian_plant_loader import LeafCluster
from models.structure.graph_cleanup import RootedBranchTree


@dataclass
class LeafAttachment:
    leaf_idx: int
    leaf_name: str
    parent_edge_idx: int          # index into RootedBranchTree.edges_oriented
    u: float                       # along-edge parameter in [0, 1]
    surface_point: torch.Tensor    # [3] world coords on cylinder surface
    rest_length: float             # || disk_center - surface_point ||
    rest_direction: torch.Tensor   # [3] unit vector from surface_point toward disk_center
    disk_center: torch.Tensor      # [3]
    disk_normal: torch.Tensor      # [3] unit
    disk_radius: float             # max in-plane distance from centroid
    leaf_root: torch.Tensor        # [3] candidate petiole-side point
    score: float                   # final score (lower = better)


def _fit_disk(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """PCA: returns (centroid, normal, in-plane radius)."""
    c = points.mean(dim=0)
    centered = points - c
    # SVD on covariance — small matrix, exact
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]                                        # smallest variance direction
    normal = normal / (normal.norm() + 1e-12)
    # In-plane radius: max distance from centroid in the disk plane
    in_plane = centered - (centered @ normal)[:, None] * normal
    radius = float(in_plane.norm(dim=-1).max().item())
    return c, normal, radius


def _leaf_root_candidate(points: torch.Tensor, branch_nodes: torch.Tensor) -> tuple[torch.Tensor, float]:
    """The leaf-cluster point closest to any branch node — likely the petiole side."""
    # [N_k, N_b] pairwise distances
    d = torch.cdist(points, branch_nodes)
    pt_idx = int(d.min(dim=1).values.argmin().item())
    nearest_branch_d = float(d.min(dim=1).values[pt_idx].item())
    return points[pt_idx], nearest_branch_d


def _score_edges(
    leaf_root: torch.Tensor,
    tree: RootedBranchTree,
    terminality_bonus: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each edge, return (score, u, surface_point).

    score = surface_distance - terminality_bonus * is_terminal_child
    surface_distance = max(0, centerline_distance - edge_radius)
    """
    p = tree.nodes[tree.edges_oriented[:, 0]]              # [E, 3]
    q = tree.nodes[tree.edges_oriented[:, 1]]              # [E, 3]
    seg = q - p                                             # [E, 3]
    seg_len2 = (seg * seg).sum(-1).clamp_min(1e-12)         # [E]

    t = ((leaf_root - p) * seg).sum(-1) / seg_len2          # [E]
    t = t.clamp(0.0, 1.0)
    closest = p + t[:, None] * seg                          # [E, 3]
    centerline_d = (leaf_root - closest).norm(dim=-1)       # [E]
    surface_d = (centerline_d - tree.edge_radius).clamp_min(0.0)

    is_terminal = (tree.subtree_size[tree.edges_oriented[:, 1]] == 1).float()
    score = surface_d - terminality_bonus * is_terminal

    return score, t, closest


def infer_leaf_attachments(
    leaves: list[LeafCluster],
    tree: RootedBranchTree,
    terminality_bonus: float = 0.005,
    orphan_warn_factor: float = 30.0,
) -> list[LeafAttachment]:
    """Attach each leaf cluster to the best branch edge.

    Args:
        leaves: leaf clusters from the loader
        tree: rooted branch tree from `root_branch_graph`
        terminality_bonus: subtract this from the score for edges whose child is
            a terminal node (subtree_size == 1). Tune to the scene scale.
        orphan_warn_factor: print a warning if a leaf's nearest-branch distance
            exceeds (median edge length) * this factor.
    """
    median_edge_len = float(tree.edge_length.median().item())
    attachments: list[LeafAttachment] = []

    for k, cluster in enumerate(leaves):
        points = cluster.points
        c, normal, r = _fit_disk(points)
        leaf_root, nearest_branch_d = _leaf_root_candidate(points, tree.nodes)

        if nearest_branch_d > median_edge_len * orphan_warn_factor:
            print(
                f'[leaf_attachment] WARNING: leaf {k} ({cluster.name}) is far from any branch '
                f'(nearest branch dist {nearest_branch_d:.3f} vs median edge {median_edge_len:.3f})'
            )

        score, t, closest = _score_edges(leaf_root, tree, terminality_bonus)
        edge_idx = int(score.argmin().item())
        u = float(t[edge_idx].item())
        centerline_pt = closest[edge_idx]

        # Snap to cylinder surface along (leaf_root - centerline_pt) direction.
        offset = leaf_root - centerline_pt
        offset_norm = offset.norm()
        if offset_norm > 1e-8:
            surface_pt = centerline_pt + (tree.edge_radius[edge_idx] * offset / offset_norm)
        else:
            # Leaf root coincides with the centerline — fall back to the disk normal direction.
            surface_pt = centerline_pt + tree.edge_radius[edge_idx] * normal

        rest_vec = c - surface_pt
        rest_len = float(rest_vec.norm().item())
        rest_dir = rest_vec / (rest_vec.norm() + 1e-8)

        attachments.append(LeafAttachment(
            leaf_idx=k,
            leaf_name=cluster.name,
            parent_edge_idx=edge_idx,
            u=u,
            surface_point=surface_pt,
            rest_length=rest_len,
            rest_direction=rest_dir,
            disk_center=c,
            disk_normal=normal,
            disk_radius=r,
            leaf_root=leaf_root,
            score=float(score[edge_idx].item()),
        ))

    return attachments


def attachments_summary(attachments: list[LeafAttachment], tree: RootedBranchTree) -> str:
    if not attachments:
        return '(no leaves)'
    rest = torch.tensor([a.rest_length for a in attachments])
    disk_r = torch.tensor([a.disk_radius for a in attachments])
    # Edges that received at least one leaf
    parent_edges = [a.parent_edge_idx for a in attachments]
    n_unique_edges = len(set(parent_edges))
    n_terminal = sum(
        1 for a in attachments
        if int(tree.subtree_size[tree.edges_oriented[a.parent_edge_idx, 1]].item()) == 1
    )
    return (
        f'leaves attached: {len(attachments)}; unique parent edges: {n_unique_edges}; '
        f'attached to terminal: {n_terminal}/{len(attachments)}\n'
        f'  rest_length  min/median/max: {rest.min():.4f} / {rest.median():.4f} / {rest.max():.4f}\n'
        f'  disk_radius  min/median/max: {disk_r.min():.4f} / {disk_r.median():.4f} / {disk_r.max():.4f}'
    )


if __name__ == '__main__':
    import argparse
    from data.gaussian_plant_loader import load_gaussian_plant_scene
    from models.structure.graph_cleanup import root_branch_graph

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    parser.add_argument('--output', default='outputs/gsplant_output/newplant9')
    parser.add_argument('--root', type=int, default=None)
    parser.add_argument('--terminality-bonus', type=float, default=0.005)
    args = parser.parse_args()

    scene = load_gaussian_plant_scene(args.source, args.output)
    tree = root_branch_graph(scene.branch, scene.tube, root_idx=args.root)
    attachments = infer_leaf_attachments(scene.leaves, tree, terminality_bonus=args.terminality_bonus)
    print(attachments_summary(attachments, tree))
