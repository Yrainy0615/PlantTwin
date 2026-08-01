"""StPr cylinder-mesh before/after comparison (geometry / depth recovery).

Renders the branch tree as variable-radius tubes (the StPr cylinder representation),
with node positions taken from the fused optimizer's result:

  before = motion_out  (multi-view static optimization, NO motion)
  after  = motion_in   (+ faithful video RGB + 3D rigidity)

Both panels share one camera. The view is rotated ~90 deg off the video camera's
forward axis so the recovered *depth* (which the perturbation corrupts along that
forward axis) is actually visible. Tubes are colored by per-node 3D error to the GT
structure (Pstar) on a shared scale, so the depth cleanup reads at a glance without
adding a third panel.

Input: outputs/per_scene_optim/fuse/joint.pt  (must contain results[*]['P'],
edges_true, edge_radius, Pstar).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def node_radius_from_edges(N, edges, edge_radius, cap_pct=95.0):
    """Assign each node the radius of its thickest incident edge.

    The root/trunk disk radius is a large outlier (~0.6 m vs median ~0.016 m) and
    would render as a giant ball that hides the branches, so radii are clamped at the
    cap_pct percentile of the edge radii."""
    er = edge_radius.numpy() if torch.is_tensor(edge_radius) else np.asarray(edge_radius)
    cap = float(np.percentile(er, cap_pct))
    er = np.minimum(er, cap)
    r = np.full(N, float(er.min()), dtype=np.float32)
    e = edges.numpy() if torch.is_tensor(edges) else np.asarray(edges)
    for (a, b), rad in zip(e, er):
        r[a] = max(r[a], rad); r[b] = max(r[b], rad)
    return r


def build_polyline_graph(points, edges, node_radius):
    """Inlined from GaussianPlant.utils.gs_utils (avoids its pytorch3d import)."""
    import pyvista as pv
    points = np.asarray(points, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.int64)
    node_radius = np.asarray(node_radius, dtype=np.float32)
    lines = np.empty(edges.shape[0] * 3, dtype=np.int64)
    lines[0::3] = 2
    lines[1::3] = edges[:, 0]
    lines[2::3] = edges[:, 1]
    poly = pv.PolyData(points)
    poly.lines = lines
    poly['radius'] = node_radius
    return poly


def build_tube(P, edges, node_radius, err):
    poly = build_polyline_graph(P, edges, node_radius)
    poly['err'] = err.astype(np.float32)
    # vary tube radius by the per-node 'radius' scalar (absolute values)
    tube = poly.tube(scalars='radius', absolute=True, n_sides=14, capping=True)
    return tube


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pt', default='outputs/per_scene_optim/fuse/joint.pt')
    ap.add_argument('--out', default='outputs/per_scene_optim/fuse/cylinder_before_after.png')
    ap.add_argument('--azimuth', type=float, default=90.0,
                    help='rotate camera off the auto view to expose depth')
    ap.add_argument('--elevation', type=float, default=15.0)
    ap.add_argument('--window', type=int, nargs=2, default=[1500, 1100])
    args = ap.parse_args()

    import pyvista as pv
    pv.OFF_SCREEN = True
    try:
        pv.start_xvfb()
    except Exception:
        pass

    d = torch.load(args.pt, map_location='cpu', weights_only=False)
    edges = d['edges_true']
    edge_radius = d['edge_radius']
    Pstar = d['Pstar'].numpy()
    N = Pstar.shape[0]
    nr = node_radius_from_edges(N, edges, edge_radius)

    P_before = d['results']['motion_out']['P'].numpy()
    P_after = d['results']['motion_in']['P'].numpy()
    err_before = np.linalg.norm(P_before - Pstar, axis=-1) * 100.0  # cm
    err_after = np.linalg.norm(P_after - Pstar, axis=-1) * 100.0
    clim = (0.0, float(np.percentile(np.concatenate([err_before, err_after]), 95)))

    rmse_b = d['results']['motion_out']['depth'] * 100
    rmse_a = d['results']['motion_in']['depth'] * 100

    pl = pv.Plotter(shape=(1, 2), off_screen=True, window_size=args.window, border=False)
    edges_np = edges.numpy()

    for col, (P, err, title) in enumerate([
        (P_before, err_before, f'BEFORE (motion_out)\ndepth RMSE = {rmse_b:.2f} cm'),
        (P_after, err_after, f'AFTER (motion_in)\ndepth RMSE = {rmse_a:.2f} cm'),
    ]):
        pl.subplot(0, col)
        tube = build_tube(P, edges_np, nr, err)
        pl.add_mesh(tube, scalars='err', cmap='RdYlGn_r', clim=clim,
                    scalar_bar_args={'title': 'node err to GT (cm)', 'fmt': '%.1f'},
                    smooth_shading=True)
        pl.add_text(title, font_size=11, position='upper_edge')
        pl.set_background('white')

    pl.link_views()
    pl.subplot(0, 0)
    pl.camera_position = 'yz'           # look along +x; depth (video cam fwd ~ x/z) becomes lateral
    pl.reset_camera()
    pl.camera.azimuth = args.azimuth
    pl.camera.elevation = args.elevation
    pl.reset_camera()
    pl.camera.zoom(1.6)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(args.out)
    pl.close()
    print(f'wrote {args.out}  (before depthRMSE {rmse_b:.2f}cm -> after {rmse_a:.2f}cm)')


if __name__ == '__main__':
    main()
