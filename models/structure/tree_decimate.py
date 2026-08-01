"""Decimate a RootedBranchTree by collapsing degree-2 interior nodes.

Inverse operation to `tree_densify.split_edges`. Used to manufacture an
*under-articulated* coarse init for the motion-fusion densification experiment:
collapse a random fraction of degree-2 nodes (one child, not root), merging each
node's two incident edges into one straight edge. The full tree is kept as the
faithful motion source; the coarse tree is what the optimizer starts from, and
densification must re-discover the collapsed joints from motion residual.

Collapsing only degree-2 nodes preserves all junctions, so the result is a valid
tree with the same root. `info` carries the coarse->GT node map and which coarse
edge each collapsed GT node now lies on, for recovery precision/recall eval.
"""

from __future__ import annotations

import torch

from models.structure.graph_cleanup import RootedBranchTree


def _children_of(parent: torch.Tensor) -> dict[int, list[int]]:
    ch: dict[int, list[int]] = {i: [] for i in range(parent.shape[0])}
    for c in range(parent.shape[0]):
        p = int(parent[c].item())
        if p >= 0:
            ch[p].append(c)
    return ch


def decimate_tree(tree: RootedBranchTree, frac: float, seed: int = 0
                  ) -> tuple[RootedBranchTree, dict]:
    """Collapse `frac` of the eligible degree-2 (single-child, non-root) nodes."""
    N = tree.nodes.shape[0]
    parent = tree.parent
    children = _children_of(parent)
    edge_of_child = {int(c): i for i, c in enumerate(tree.edges_oriented[:, 1].tolist())}

    eligible = [i for i in range(N)
                if i != tree.root_idx and parent[i] >= 0 and len(children[i]) == 1]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(eligible), generator=g).tolist()
    n_collapse = int(round(frac * len(eligible)))
    collapsed = set(eligible[perm[i]] for i in range(n_collapse))

    survivors = [i for i in range(N) if i not in collapsed]
    old2new = {old: new for new, old in enumerate(survivors)}

    def eff_parent(c: int) -> int:
        a = int(parent[c].item())
        while a in collapsed:
            a = int(parent[a].item())
        return a

    # Build merged edges for every surviving non-root node by walking up the
    # collapsed chain, accumulating length and length-weighted radius.
    new_edges, new_len, new_rad, new_type = [], [], [], []
    for c in survivors:
        if c == tree.root_idx:
            continue
        p = eff_parent(c)
        L = 0.0; rw = 0.0; cur = c; last_type = None
        while cur != p:
            e = edge_of_child[cur]
            el = float(tree.edge_length[e]); L += el
            rw += el * float(tree.edge_radius[e])
            if last_type is None:
                last_type = int(tree.edge_type[e])           # child-side edge type
            cur = int(parent[cur].item())
        new_edges.append((old2new[p], old2new[c]))
        new_len.append(L)
        new_rad.append(rw / max(L, 1e-9))
        new_type.append(last_type)

    Nc = len(survivors)
    new_nodes = tree.nodes[torch.tensor(survivors, dtype=torch.long)]
    new_parent = torch.full((Nc,), -1, dtype=torch.long)
    edges_t = torch.tensor(new_edges, dtype=torch.long) if new_edges else torch.empty(0, 2, dtype=torch.long)
    for a, b in new_edges:
        new_parent[b] = a
    root_new = old2new[tree.root_idx]

    # depth / subtree via BFS from the (unchanged) root
    ch_new = _children_of(new_parent)
    depth = torch.full((Nc,), -1, dtype=torch.long)
    depth[root_new] = 0
    order = [root_new]; head = 0
    while head < len(order):
        u = order[head]; head += 1
        for v in ch_new[u]:
            if depth[v] < 0:
                depth[v] = depth[u] + 1; order.append(v)
    subtree = torch.ones(Nc, dtype=torch.long)
    for u in reversed(order):
        p = int(new_parent[u].item())
        if p >= 0:
            subtree[p] += subtree[u]

    edge_len = torch.tensor(new_len, dtype=torch.float32)
    edge_rad = torch.tensor(new_rad, dtype=torch.float32)
    edge_type = torch.tensor(new_type, dtype=torch.long)

    # BFS (parent->child by depth) reorder so row index lines up with traversal
    if edges_t.shape[0] > 0:
        eo = torch.argsort(depth[edges_t[:, 1]])
        edges_t = edges_t[eo]; edge_len = edge_len[eo]; edge_rad = edge_rad[eo]; edge_type = edge_type[eo]

    coarse = RootedBranchTree(
        nodes=new_nodes, root_idx=root_new, parent=new_parent, depth=depth,
        edges_oriented=edges_t, edge_length=edge_len, edge_radius=edge_rad,
        edge_type=edge_type, subtree_size=subtree,
    )

    # --- info for recovery eval ---
    # row index of each coarse edge keyed by its child (child uniquely identifies edge)
    row_of_child = {int(c): i for i, c in enumerate(edges_t[:, 1].tolist())}

    def eff_child(m: int) -> int:
        cur = m
        while cur in collapsed:
            cur = children[cur][0]               # degree-2 => single child
        return cur

    collapsed_sorted = sorted(collapsed)
    collapsed_onto_edge = []
    for m in collapsed_sorted:
        c_surv = eff_child(m)
        collapsed_onto_edge.append(row_of_child[old2new[c_surv]])
    edge_has_collapsed = torch.zeros(edges_t.shape[0], dtype=torch.bool)
    if collapsed_onto_edge:
        edge_has_collapsed[torch.tensor(collapsed_onto_edge, dtype=torch.long)] = True

    info = {
        'kept_gt_idx': torch.tensor(survivors, dtype=torch.long),       # coarse i -> GT node
        'collapsed_gt_idx': torch.tensor(collapsed_sorted, dtype=torch.long),
        'collapsed_onto_edge': torch.tensor(collapsed_onto_edge, dtype=torch.long)
        if collapsed_onto_edge else torch.empty(0, dtype=torch.long),
        'edge_has_collapsed': edge_has_collapsed,                       # [E_coarse] bool
        'n_collapsed': len(collapsed),
        'n_eligible': len(eligible),
    }
    return coarse, info
