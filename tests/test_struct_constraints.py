"""Unit tests for optimization/struct_constraints.py.

Run: PYTHONPATH=. python scripts/test_struct_constraints.py
(plain asserts; no pytest dependency required.)

Covers:
  - geometric_containment_loss: zero when nodes are inside the band, positive +
    gradient pulls an escaped node back toward the branch cloud.
  - motion_direction_consistency_loss: ~zero for aligned neighbor motion, positive
    for anti-aligned, and a gradient step reduces it.
  - estimate_geom_margin / subsample_points sanity.
"""

from __future__ import annotations

import torch

from optimization.struct_constraints import (
    geometric_containment_loss,
    motion_direction_consistency_loss,
    estimate_geom_margin,
    subsample_points,
    nearest_branch_distance,
    rest_relative_tolerance,
)


def _line_cloud(n=2000):
    """A dense 'branch': points along the x-axis from 0..1, tiny radial jitter."""
    t = torch.linspace(0, 1, n).unsqueeze(1)
    base = torch.cat([t, torch.zeros(n, 1), torch.zeros(n, 1)], dim=1)
    jitter = 0.01 * torch.randn(n, 3)
    return base + jitter


def test_geom_zero_inside():
    torch.manual_seed(0)
    cloud = _line_cloud()
    # Nodes sitting right on the line (inside) -> nn dist ~ jitter scale << r_tol.
    nodes = torch.stack([torch.linspace(0.1, 0.9, 5),
                         torch.zeros(5), torch.zeros(5)], dim=1)
    loss, info = geometric_containment_loss(nodes, cloud, r_tol=0.05)
    assert loss.item() < 1e-4, f'inside nodes should have ~0 geom loss, got {loss.item()}'
    assert info['n_outside'] == 0, info
    print(f'[ok] geom inside: loss={loss.item():.2e}, n_outside=0')


def test_geom_penalizes_and_pulls_back_outside():
    torch.manual_seed(0)
    cloud = _line_cloud()
    # One node far off the line in +y (escaped the branch).
    escaped = torch.tensor([[0.5, 0.5, 0.0]], requires_grad=True)
    r_tol = 0.05
    loss, info = geometric_containment_loss(escaped, cloud, r_tol=r_tol)
    assert loss.item() > 0, 'escaped node must incur positive geom loss'
    assert info['n_outside'] == 1
    loss.backward()
    g = escaped.grad[0]
    # Gradient descent (-g) should move the node back toward the line, i.e. -y.
    assert (-g)[1] < 0, f'restoring direction should point toward branch (-y), grad_y={g[1].item()}'
    # A small step must reduce the distance to the cloud.
    with torch.no_grad():
        moved = escaped - 0.1 * escaped.grad
    d0 = nearest_branch_distance(escaped.detach(), cloud).item()
    d1 = nearest_branch_distance(moved, cloud).item()
    assert d1 < d0, f'step should reduce nn-dist: {d0:.3f} -> {d1:.3f}'
    print(f'[ok] geom outside: loss={loss.item():.3e}, nn {d0:.3f}->{d1:.3f}, grad_y={g[1].item():.3f}')


def test_geom_slide_along_branch_is_free():
    """A node moved ALONG the branch (still inside) stays penalty-free — distinguishes
    this constraint from a plain L2-to-original-position prior."""
    torch.manual_seed(0)
    cloud = _line_cloud()
    a = torch.tensor([[0.2, 0.0, 0.0]])
    b = torch.tensor([[0.8, 0.0, 0.0]])   # slid far along x but still on the branch
    la, _ = geometric_containment_loss(a, cloud, r_tol=0.05)
    lb, _ = geometric_containment_loss(b, cloud, r_tol=0.05)
    assert la.item() < 1e-4 and lb.item() < 1e-4, (la.item(), lb.item())
    print(f'[ok] geom slide-along free: {la.item():.2e}, {lb.item():.2e}')


def test_motion_aligned_zero():
    T, N = 4, 3
    edges = torch.tensor([[0, 1], [1, 2]])
    rest = torch.zeros(N, 3)
    # All nodes move in the same +x direction (different magnitudes) -> aligned.
    traj = torch.zeros(T, N, 3)
    for t in range(T):
        traj[t, :, 0] = torch.tensor([0.1, 0.2, 0.3]) * (t + 1)
    loss = motion_direction_consistency_loss(traj, rest, edges)
    assert loss.item() < 1e-5, f'aligned motion should be ~0, got {loss.item()}'
    print(f'[ok] motion aligned: loss={loss.item():.2e}')


def test_motion_antialigned_positive_and_improves():
    T, N = 1, 2
    edges = torch.tensor([[0, 1]])
    rest = torch.zeros(N, 3)
    traj = torch.zeros(T, N, 3, requires_grad=True)
    with torch.no_grad():
        traj[0, 0] = torch.tensor([1.0, 0.0, 0.0])     # parent moves +x
        # Child nearly opposite but slightly off-axis so the gradient is non-degenerate
        # (perfectly anti-parallel cos=-1 is a symmetric saddle with no restoring dir).
        traj[0, 1] = torch.tensor([-1.0, 0.15, 0.0])
    loss = motion_direction_consistency_loss(traj, rest, edges)
    assert loss.item() > 1.5, f'near anti-aligned (1-cos)~2 expected, got {loss.item()}'
    loss.backward()
    with torch.no_grad():
        moved = traj - 0.2 * traj.grad
    new_loss = motion_direction_consistency_loss(moved, rest, edges)
    assert new_loss.item() < loss.item(), f'step should reduce: {loss.item():.3f}->{new_loss.item():.3f}'
    print(f'[ok] motion anti-aligned: {loss.item():.3f}->{new_loss.item():.3f}')


def test_motion_magnitude_gating():
    """A neighbor with ~zero motion should not dominate the direction penalty."""
    T, N = 1, 2
    edges = torch.tensor([[0, 1]])
    rest = torch.zeros(N, 3)
    traj = torch.zeros(T, N, 3)
    traj[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    traj[0, 1] = torch.tensor([1e-9, 0.0, 0.0])   # essentially still
    loss = motion_direction_consistency_loss(traj, rest, edges)
    assert loss.item() < 1e-3, f'gated by tiny magnitude, got {loss.item()}'
    print(f'[ok] motion gating: loss={loss.item():.2e}')


def test_margin_and_subsample():
    torch.manual_seed(0)
    cloud = _line_cloud(5000)
    nodes = torch.stack([torch.linspace(0.1, 0.9, 20),
                         0.01 * torch.randn(20), 0.01 * torch.randn(20)], dim=1)
    r_tol = estimate_geom_margin(nodes, cloud, quantile=0.95, scale=2.0)
    assert 0.0 < r_tol < 0.3, f'margin out of expected range: {r_tol}'
    sub = subsample_points(cloud, 1000)
    assert sub.shape[0] == 1000
    assert subsample_points(cloud, 999999).shape[0] == cloud.shape[0]
    print(f'[ok] margin={r_tol:.4f}, subsample ok')


def test_rest_relative_tolerance():
    """A node already off the cloud at rest gets its own band (not forced inward);
    delta is only penalized for pushing it FARTHER out."""
    torch.manual_seed(0)
    cloud = _line_cloud()
    # node A on the branch (inside), node B far off in +y (tube-uncovered at rest)
    rest = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.4, 0.0]])
    tol = rest_relative_tolerance(rest, cloud, global_r_tol=0.05, margin=0.02)
    assert tol[0] < 0.06, f'inside node should keep global band, got {tol[0].item()}'
    assert tol[1] > 0.38, f'tube-uncovered node should get own band ~0.4, got {tol[1].item()}'
    # B staying put -> no penalty; B pushed farther out -> penalty.
    l_stay, info_stay = geometric_containment_loss(rest, cloud, tol)
    assert info_stay['n_outside'] == 0, info_stay
    pushed = rest.clone(); pushed[1, 1] = 0.6   # B drifts farther from branch
    l_push, info_push = geometric_containment_loss(pushed, cloud, tol)
    assert info_push['n_outside'] == 1 and l_push.item() > l_stay.item()
    print(f'[ok] rest-relative tol: inside={tol[0].item():.3f} outside={tol[1].item():.3f}; '
          f'stay nout=0, push nout=1')


def main():
    tests = [
        test_geom_zero_inside,
        test_geom_penalizes_and_pulls_back_outside,
        test_geom_slide_along_branch_is_free,
        test_rest_relative_tolerance,
        test_motion_aligned_zero,
        test_motion_antialigned_positive_and_improves,
        test_motion_magnitude_gating,
        test_margin_and_subsample,
    ]
    for t in tests:
        t()
    print(f'\nAll {len(tests)} struct-constraint tests passed.')


if __name__ == '__main__':
    main()
