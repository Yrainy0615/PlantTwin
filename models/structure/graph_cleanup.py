"""Root the branch MST, derive parent/child relations, and fit per-segment radii.

Input: `BranchGraph` (nodes + undirected edges, tree) and an optional `BranchTube`
mesh used to estimate the cylinder radius along each branch segment.

Output: `RootedBranchTree` with:
- root index
- parent array (per node, -1 for root)
- per-edge attributes (oriented parent→child, length, fitted radius, depth, type)

Tree typing is intentionally simple for v0: the path from the root that always
follows the largest-subtree child is labeled `stem`; everything else is `branch`.
This is refinable later (e.g. via radius thresholds or DINO part labels).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from data.gaussian_plant_loader import BranchGraph, BranchTube


STEM = 0
BRANCH = 1


@dataclass
class RootedBranchTree:
    nodes: torch.Tensor          # [N, 3]
    root_idx: int
    parent: torch.Tensor         # [N] long, -1 for root
    depth: torch.Tensor          # [N] long, 0 for root
    edges_oriented: torch.Tensor # [N-1, 2] long, (parent, child)
    edge_length: torch.Tensor    # [N-1] float
    edge_radius: torch.Tensor    # [N-1] float
    edge_type: torch.Tensor      # [N-1] long (STEM or BRANCH)
    subtree_size: torch.Tensor   # [N] long, |descendants|+1


def _select_root(nodes: torch.Tensor, root_idx: int | None) -> int:
    if root_idx is not None:
        return int(root_idx)
    # GaussianPlant's exported world frame has +Y pointing UP (visualized as
    # image top after the COLMAP camera transform we verified). The plant's
    # base is therefore the node with the smallest Y value.
    return int(torch.argmin(nodes[:, 1]).item())


def _build_adjacency(edges: torch.Tensor, n: int) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges.tolist():
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _bfs_root(adj: list[list[int]], root: int, n: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    parent = torch.full((n,), -1, dtype=torch.long)
    depth = torch.zeros(n, dtype=torch.long)
    order = [root]
    seen = [False] * n
    seen[root] = True
    head = 0
    while head < len(order):
        u = order[head]
        head += 1
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                parent[v] = u
                depth[v] = depth[u] + 1
                order.append(v)
    if not all(seen):
        n_unreached = sum(1 for s in seen if not s)
        raise ValueError(f'BranchGraph is not connected from root {root}: {n_unreached} unreached nodes')
    return parent, depth, order


def _subtree_sizes(parent: torch.Tensor, traversal_order: list[int]) -> torch.Tensor:
    n = parent.shape[0]
    size = torch.ones(n, dtype=torch.long)
    # Process in reverse BFS order so children are summed before parents.
    for u in reversed(traversal_order):
        p = int(parent[u].item())
        if p >= 0:
            size[p] += size[u]
    return size


def _fit_radii_per_edge(
    edges_oriented: torch.Tensor,
    nodes: torch.Tensor,
    tube_vertices: torch.Tensor | None,
    default_radius: float = 0.01,
) -> torch.Tensor:
    """For each parent→child edge, fit the cylinder radius as the mean distance
    from nearby tube-mesh vertices to the segment axis.

    Strategy: for every tube vertex, assign it to the closest edge segment, then
    average the perpendicular distance.
    """
    n_edges = edges_oriented.shape[0]
    if tube_vertices is None or tube_vertices.shape[0] == 0:
        return torch.full((n_edges,), default_radius)

    p = nodes[edges_oriented[:, 0]]                       # [E, 3]
    q = nodes[edges_oriented[:, 1]]                       # [E, 3]
    seg = q - p                                            # [E, 3]
    seg_len2 = (seg * seg).sum(-1).clamp_min(1e-12)        # [E]

    sums = torch.zeros(n_edges, dtype=torch.float32)
    counts = torch.zeros(n_edges, dtype=torch.long)

    chunk = 4096
    V = tube_vertices.shape[0]
    for start in range(0, V, chunk):
        verts = tube_vertices[start:start + chunk]         # [C, 3]
        diff = verts[:, None, :] - p[None, :, :]           # [C, E, 3]
        t = (diff * seg[None, :, :]).sum(-1) / seg_len2[None, :]
        t = t.clamp(0.0, 1.0)                              # [C, E]
        closest = p[None, :, :] + t[..., None] * seg[None, :, :]  # [C, E, 3]
        d = torch.linalg.norm(verts[:, None, :] - closest, dim=-1) # [C, E]
        edge_idx = torch.argmin(d, dim=-1)                 # [C]
        edge_dist = d.gather(1, edge_idx[:, None]).squeeze(-1)
        sums.scatter_add_(0, edge_idx, edge_dist)
        counts.scatter_add_(0, edge_idx, torch.ones_like(edge_idx, dtype=torch.long))

    radii = torch.where(counts > 0, sums / counts.clamp_min(1).float(), torch.full_like(sums, default_radius))
    return radii


def _label_stem_branch(
    edges_oriented: torch.Tensor,
    parent: torch.Tensor,
    subtree_size: torch.Tensor,
    root_idx: int,
) -> torch.Tensor:
    """Walk from the root, always following the largest-subtree child — that path
    is the stem. Every edge off that path is a branch.
    """
    n_edges = edges_oriented.shape[0]
    edge_type = torch.full((n_edges,), BRANCH, dtype=torch.long)

    # children-of-node lookup
    children: dict[int, list[int]] = {}
    for ei in range(n_edges):
        u, v = int(edges_oriented[ei, 0].item()), int(edges_oriented[ei, 1].item())
        children.setdefault(u, []).append((v, ei))

    cur = root_idx
    while cur in children and children[cur]:
        # pick child with largest subtree size
        best_child, best_edge = max(children[cur], key=lambda cv: int(subtree_size[cv[0]].item()))
        edge_type[best_edge] = STEM
        cur = best_child

    return edge_type


def root_branch_graph(
    branch: BranchGraph,
    tube: BranchTube | None,
    root_idx: int | None = None,
) -> RootedBranchTree:
    n = branch.nodes.shape[0]
    if branch.edges.shape[0] != n - 1:
        raise ValueError(
            f'Branch graph is not a tree: {n} nodes, {branch.edges.shape[0]} edges (expected {n - 1}).'
        )

    root = _select_root(branch.nodes, root_idx)
    adj = _build_adjacency(branch.edges, n)
    parent, depth, traversal = _bfs_root(adj, root, n)

    # Oriented edges parent→child (drop the root since parent[root] = -1)
    child_ids = torch.tensor([i for i in range(n) if i != root], dtype=torch.long)
    parent_ids = parent[child_ids]
    edges_oriented = torch.stack([parent_ids, child_ids], dim=-1)

    edge_length = torch.linalg.norm(
        branch.nodes[edges_oriented[:, 1]] - branch.nodes[edges_oriented[:, 0]], dim=-1
    )

    tube_v = tube.vertices if tube is not None else None
    edge_radius = _fit_radii_per_edge(edges_oriented, branch.nodes, tube_v)

    subtree_size = _subtree_sizes(parent, traversal)
    edge_type = _label_stem_branch(edges_oriented, parent, subtree_size, root)

    return RootedBranchTree(
        nodes=branch.nodes,
        root_idx=root,
        parent=parent,
        depth=depth,
        edges_oriented=edges_oriented,
        edge_length=edge_length,
        edge_radius=edge_radius,
        edge_type=edge_type,
        subtree_size=subtree_size,
    )


def tree_summary(tree: RootedBranchTree) -> str:
    n_stem = int((tree.edge_type == STEM).sum().item())
    n_branch = int((tree.edge_type == BRANCH).sum().item())
    n_leaves_in_tree = int(((tree.subtree_size == 1)).sum().item())
    return (
        f'RootedBranchTree: nodes={tree.nodes.shape[0]}, root_idx={tree.root_idx}, '
        f'max_depth={int(tree.depth.max().item())}, terminals={n_leaves_in_tree}\n'
        f'  edges: stem={n_stem}, branch={n_branch}\n'
        f'  edge length min/median/max: '
        f'{tree.edge_length.min():.4f} / {tree.edge_length.median():.4f} / {tree.edge_length.max():.4f}\n'
        f'  edge radius min/median/max: '
        f'{tree.edge_radius.min():.4f} / {tree.edge_radius.median():.4f} / {tree.edge_radius.max():.4f}'
    )


if __name__ == '__main__':
    import argparse
    from data.gaussian_plant_loader import load_gaussian_plant_scene

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    parser.add_argument('--output', default='outputs/gsplant_output/newplant9')
    parser.add_argument('--root', type=int, default=None, help='Force root node index')
    args = parser.parse_args()

    scene = load_gaussian_plant_scene(args.source, args.output)
    tree = root_branch_graph(scene.branch, scene.tube, root_idx=args.root)
    print(tree_summary(tree))
