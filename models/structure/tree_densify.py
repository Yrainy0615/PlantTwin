"""Topology densification for the branch tree.

Given an existing `RootedBranchTree` and a set of edges to split, this module
inserts a midpoint node on each chosen edge, rebuilds parent / depth / oriented
edges / edge_type / edge_radius / subtree_size, and returns a new tree.

The intended driver is `delta_rest_pos`-magnitude scoring from a Phase-1
structure-in-the-loop optimization: edges whose child has a large learned
delta are candidates for densification, since the delta signals that motion
fit wanted more articulation there than the original topology provided.

Constraints preserved by `split_edges`:
- Each split converts (p, c) -> (p, M) + (M, c). M = midpoint(p, c) at rest.
- Subtree of c is preserved (no re-rooting). All descendant depths shift by +1.
- Stem-vs-branch label is inherited from the original edge for both halves.
- Edge radius is inherited (we lack a finer estimate at the midpoint).
"""

from __future__ import annotations

import torch

from models.structure.graph_cleanup import RootedBranchTree, STEM, BRANCH


def _descendants_of(node: int, children_of: dict[int, list[int]]) -> list[int]:
    out = []
    stack = [node]
    while stack:
        u = stack.pop()
        for v in children_of.get(u, []):
            out.append(v)
            stack.append(v)
    return out


def split_edges(tree: RootedBranchTree, edge_indices: list[int]) -> tuple[RootedBranchTree, dict]:
    """Insert a midpoint node on each given edge.

    Returns the densified tree and an `info` dict with:
        - `new_node_indices`: the [K] new node indices in the densified tree
        - `parent_edge_per_new_node`: original edge index that each new node was
          inserted on (useful when remapping leaf attachments)
        - `n_new`: K
    """
    if len(edge_indices) == 0:
        return tree, {'new_node_indices': torch.empty(0, dtype=torch.long),
                      'parent_edge_per_new_node': torch.empty(0, dtype=torch.long), 'n_new': 0}

    N = tree.nodes.shape[0]
    K = len(edge_indices)
    edge_indices_t = torch.as_tensor(edge_indices, dtype=torch.long)

    # Midpoint positions of split edges -> append as new nodes
    p_old = tree.edges_oriented[edge_indices_t, 0]
    c_old = tree.edges_oriented[edge_indices_t, 1]
    midpoints = 0.5 * (tree.nodes[p_old] + tree.nodes[c_old])
    new_nodes_pos = torch.cat([tree.nodes, midpoints], dim=0)                   # [N + K, 3]
    new_M = torch.arange(N, N + K, dtype=torch.long)

    # Build new edge list: keep all non-split edges, add (p, M) and (M, c) for each split.
    keep_mask = torch.ones(tree.edges_oriented.shape[0], dtype=torch.bool)
    keep_mask[edge_indices_t] = False
    kept_edges = tree.edges_oriented[keep_mask]
    kept_etype = tree.edge_type[keep_mask]
    kept_radius = tree.edge_radius[keep_mask]
    kept_length = tree.edge_length[keep_mask]

    pM_edges = torch.stack([p_old, new_M], dim=1)
    Mc_edges = torch.stack([new_M, c_old], dim=1)
    new_edges = torch.cat([kept_edges, pM_edges, Mc_edges], dim=0)
    split_etype = tree.edge_type[edge_indices_t]
    split_radius = tree.edge_radius[edge_indices_t]
    new_etype = torch.cat([kept_etype, split_etype, split_etype], dim=0)
    new_radius = torch.cat([kept_radius, split_radius, split_radius], dim=0)
    # Length: each half is half of the original.
    split_length = 0.5 * tree.edge_length[edge_indices_t]
    new_length = torch.cat([kept_length, split_length, split_length], dim=0)

    # Parent array. Start from the original; update:
    #   - parent[M] = p_old
    #   - parent[c_old] = M (was p_old)
    new_parent = torch.cat([tree.parent, torch.full((K,), -1, dtype=torch.long)], dim=0)
    new_parent[new_M] = p_old
    new_parent[c_old] = new_M

    # Depth: rebuild via BFS from the original root index (unchanged).
    N_new = N + K
    children_of: dict[int, list[int]] = {i: [] for i in range(N_new)}
    for c in range(N_new):
        p = int(new_parent[c].item())
        if p >= 0:
            children_of[p].append(c)

    depth = torch.full((N_new,), -1, dtype=torch.long)
    depth[tree.root_idx] = 0
    order = [tree.root_idx]
    head = 0
    while head < len(order):
        u = order[head]; head += 1
        for v in children_of[u]:
            if depth[v] < 0:
                depth[v] = depth[u] + 1
                order.append(v)

    # Subtree sizes (reverse BFS order accumulation)
    subtree_size = torch.ones(N_new, dtype=torch.long)
    for u in reversed(order):
        p = int(new_parent[u].item())
        if p >= 0:
            subtree_size[p] += subtree_size[u]

    # Reorder new_edges into BFS (parent->child) order so downstream code that
    # iterates by row index lines up with depth-major traversal.
    edge_order = torch.argsort(depth[new_edges[:, 1]])
    new_edges = new_edges[edge_order]
    new_etype = new_etype[edge_order]
    new_radius = new_radius[edge_order]
    new_length = new_length[edge_order]

    new_tree = RootedBranchTree(
        nodes=new_nodes_pos,
        root_idx=tree.root_idx,
        parent=new_parent,
        depth=depth,
        edges_oriented=new_edges,
        edge_length=new_length,
        edge_radius=new_radius,
        edge_type=new_etype,
        subtree_size=subtree_size,
    )
    info = {
        'new_node_indices': new_M,
        'parent_edge_per_new_node': edge_indices_t,
        'n_new': K,
    }
    return new_tree, info


def pick_split_edges(
    tree: RootedBranchTree,
    delta_rest_pos: torch.Tensor,
    top_k: int,
    min_edge_length: float | None = None,
    min_delta_norm: float | None = None,
) -> list[int]:
    """Score each edge by the |delta_rest_pos| of its child node and pick top-K.

    Skip edges shorter than `min_edge_length` (don't split already-tiny ones)
    and edges whose child delta-norm is below `min_delta_norm`.
    """
    delta_norm = delta_rest_pos.norm(dim=-1)
    edge_score = delta_norm[tree.edges_oriented[:, 1]]

    mask = torch.ones(edge_score.shape[0], dtype=torch.bool)
    if min_edge_length is not None:
        mask &= tree.edge_length >= min_edge_length
    if min_delta_norm is not None:
        mask &= edge_score >= min_delta_norm
    valid = mask.nonzero(as_tuple=False).flatten()
    if valid.numel() == 0:
        return []
    scores = edge_score[valid]
    k = min(top_k, valid.numel())
    top = torch.topk(scores, k, largest=True).indices
    return valid[top].tolist()


if __name__ == '__main__':
    import argparse
    from data.gaussian_plant_loader import load_gaussian_plant_scene
    from models.structure.graph_cleanup import root_branch_graph

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    parser.add_argument('--output', default='outputs/gsplant_output/newplant9')
    parser.add_argument('--params', required=True)
    parser.add_argument('--top-k', type=int, default=10)
    args = parser.parse_args()

    scene = load_gaussian_plant_scene(args.source, args.output)
    tree = root_branch_graph(scene.branch, scene.tube)
    p = torch.load(args.params, map_location='cpu', weights_only=False)
    if 'delta_rest_pos' not in p:
        raise SystemExit('params must include delta_rest_pos (train with --learn-struct)')
    idx = pick_split_edges(tree, p['delta_rest_pos'], args.top_k,
                            min_edge_length=float(tree.edge_length.median()) * 0.3)
    print(f'picked {len(idx)} edges to split: {idx}')
    new_tree, info = split_edges(tree, idx)
    print(f'tree before: N={tree.nodes.shape[0]}, after: N={new_tree.nodes.shape[0]}')
    print(f'max depth before: {int(tree.depth.max())}, after: {int(new_tree.depth.max())}')
