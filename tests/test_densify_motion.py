"""Unit tests: tree decimation validity + motion non-rigidity residual behavior."""

from __future__ import annotations

import numpy as np
import torch

from data.gaussian_plant_loader import load_gaussian_plant_scene
from models.structure.graph_cleanup import root_branch_graph
from models.structure.tree_decimate import decimate_tree
from models.structure.tree_densify import split_edges
from models.structure.motion_residual import _kabsch_residual, edge_nonrigidity


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f'  ok: {msg}')


def _valid_tree(t, name):
    N = t.nodes.shape[0]
    _assert(t.edges_oriented.shape[0] == N - 1, f'{name}: edges == N-1 ({N-1})')
    _assert(int((t.parent == -1).sum()) == 1, f'{name}: exactly one root')
    _assert(t.parent[t.root_idx] == -1, f'{name}: root parent is -1')
    # every non-root child appears once; depth = parent depth + 1
    for a, b in t.edges_oriented.tolist():
        _assert(int(t.parent[b]) == a, f'{name}: parent[{b}]==edge parent {a}')
        _assert(int(t.depth[b]) == int(t.depth[a]) + 1, f'{name}: depth monotone on edge ({a},{b})')
    _assert(int(t.depth.min()) == 0 and int(t.depth[t.root_idx]) == 0, f'{name}: root depth 0')


def test_decimate():
    print('[test_decimate]')
    scene = load_gaussian_plant_scene('/mnt/data/gaussianplant_data/newplant9',
                                      'outputs/gsplant_output/newplant9')
    tree = root_branch_graph(scene.branch, scene.tube)
    _valid_tree(tree, 'full')
    N0 = tree.nodes.shape[0]
    coarse, info = decimate_tree(tree, frac=0.4, seed=0)
    _valid_tree(coarse, 'coarse')
    _assert(coarse.nodes.shape[0] == N0 - info['n_collapsed'],
            f'coarse N = {N0} - {info["n_collapsed"]} = {coarse.nodes.shape[0]}')
    _assert(info['n_collapsed'] > 0, f'collapsed some nodes ({info["n_collapsed"]}/{info["n_eligible"]})')
    _assert(info['kept_gt_idx'].shape[0] == coarse.nodes.shape[0], 'kept_gt_idx covers coarse nodes')
    # coarse node positions == the GT positions of kept nodes
    _assert(torch.allclose(coarse.nodes, tree.nodes[info['kept_gt_idx']]),
            'coarse nodes are exactly the kept GT nodes')
    _assert(info['edge_has_collapsed'].shape[0] == coarse.edges_oriented.shape[0],
            'edge_has_collapsed aligned to coarse edges')
    _assert(int(info['edge_has_collapsed'].sum()) > 0, 'some coarse edges carry a collapsed joint')
    print(f'  full N={N0} -> coarse N={coarse.nodes.shape[0]} '
          f'({info["n_collapsed"]}/{info["n_eligible"]} eligible collapsed), '
          f'{int(info["edge_has_collapsed"].sum())} edges bear a bend')


def test_kabsch():
    print('[test_kabsch] rigid -> ~0, bend -> >0')
    torch.manual_seed(0)
    n, T = 30, 8
    P = torch.randn(n, 3)
    # rigid: random rotation+translation per frame
    Q = []
    for _ in range(T):
        w = torch.randn(3) * 0.5
        from simulation.articulated_chain import exp_so3
        R = exp_so3(w)
        Q.append(P @ R.T + torch.randn(3))
    Q = torch.stack(Q)
    r_rigid = float(_kabsch_residual(P, Q))
    _assert(r_rigid < 1e-4, f'rigid residual ~0 (got {r_rigid:.2e})')

    # bend: points along x-axis, rotate each by angle proportional to x (non-rigid)
    line = torch.zeros(n, 3); line[:, 0] = torch.linspace(0, 1, n)
    Qb = []
    for t in range(T):
        amp = 0.8 * (t + 1) / T
        ang = amp * line[:, 0]                       # per-point angle -> bend
        c, s = torch.cos(ang), torch.sin(ang)
        bent = torch.stack([line[:, 0] * c, line[:, 0] * s, torch.zeros(n)], -1)
        Qb.append(bent)
    Qb = torch.stack(Qb)
    r_bend = float(_kabsch_residual(line, Qb))
    _assert(r_bend > 10 * r_rigid and r_bend > 1e-3, f'bend residual >> rigid (got {r_bend:.3e})')
    print(f'  rigid={r_rigid:.2e}  bend={r_bend:.3e}')


def test_edge_nonrigidity_grouping():
    print('[test_edge_nonrigidity] grouping + split lowers residual')
    # tiny chain tree: 0(root) - 1 - 2 along x
    from models.structure.graph_cleanup import RootedBranchTree, BRANCH
    nodes = torch.tensor([[0., 0, 0], [1, 0, 0], [2, 0, 0]])
    tree = RootedBranchTree(
        nodes=nodes, root_idx=0, parent=torch.tensor([-1, 0, 1]),
        depth=torch.tensor([0, 1, 2]), edges_oriented=torch.tensor([[0, 1], [1, 2]]),
        edge_length=torch.tensor([1., 1.]), edge_radius=torch.tensor([0.1, 0.1]),
        edge_type=torch.tensor([BRANCH, BRANCH]), subtree_size=torch.tensor([3, 2, 1]))
    # branch points: 20 along edge (1,2) [the long span], bound to edge index 1
    m = 20
    pts = torch.zeros(m, 3); pts[:, 0] = torch.linspace(1, 2, m)
    binding = {'edge': torch.ones(m, dtype=torch.long), 'is_branch': torch.ones(m, dtype=torch.bool)}
    # bend trajectory of those points (non-rigid on edge 1)
    T = 6
    traj = []
    for t in range(T):
        amp = 0.7 * (t + 1) / T
        x = pts[:, 0] - 1.0
        ang = amp * x
        traj.append(torch.stack([1 + x * torch.cos(ang), x * torch.sin(ang), torch.zeros(m)], -1))
    traj = torch.stack(traj)                              # [T, m, 3]
    res = edge_nonrigidity(tree, binding, traj, min_pts=4)
    _assert(res[1] > 1e-3, f'bending edge has residual (got {float(res[1]):.3e})')
    _assert(res[0] == 0, 'edge with no bound points has 0 residual')

    # split edge 1 -> residual of each half should be lower than the whole
    new_tree, _ = split_edges(tree, [1])
    # rebind the 20 points to the two new sub-edges by x midpoint (x<1.5 -> first half)
    # find the two edges incident to the midpoint node (index 3)
    eo = new_tree.edges_oriented
    half_edge = torch.zeros(m, dtype=torch.long)
    mid_x = 1.5
    for j, (a, b) in enumerate(eo.tolist()):
        ca, cb = new_tree.nodes[a], new_tree.nodes[b]
        lo, hi = sorted([float(ca[0]), float(cb[0])])
        for i in range(m):
            if lo - 1e-6 <= float(pts[i, 0]) <= hi + 1e-6:
                half_edge[i] = j
    binding2 = {'edge': half_edge, 'is_branch': torch.ones(m, dtype=torch.bool)}
    res2 = edge_nonrigidity(new_tree, binding2, traj, min_pts=4)
    _assert(float(res2.max()) < float(res[1]),
            f'split lowers max sub-edge residual ({float(res2.max()):.3e} < {float(res[1]):.3e})')
    print(f'  whole={float(res[1]):.3e}  split halves max={float(res2.max()):.3e}')


if __name__ == '__main__':
    test_kabsch()
    test_edge_nonrigidity_grouping()
    test_decimate()
    print('\nALL TESTS PASSED')
