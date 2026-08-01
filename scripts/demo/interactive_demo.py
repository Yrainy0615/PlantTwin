"""Interactive viser demo: drag the plant with the mouse, see the v11-trained
physics respond in real time. Backend is `simulation.streaming_sim`; frontend
is a viser server that renders the appearance Gaussians via its native 3DGS
viewer and forwards mouse-drag events.

Usage:
    python -m scripts.interactive_demo \\
        --source /mnt/data/gaussianplant_data/newplant9 \\
        --output-dir outputs/gsplant_output/newplant9 \\
        --params outputs/per_scene_optim/newplant9_v11/final_params.pt \\
        --port 8080

Then open http://<host>:8080 in a browser. Left-click + drag on the plant to
pull; release to let it settle.
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import torch
import viser

from data.colmap_loader import (
    read_cameras_bin, read_images_bin, find_image_by_name, quat_to_R,
)
from data.gaussian_plant_loader import load_gaussian_plant_scene
from models.structure.graph_cleanup import root_branch_graph
from models.structure.leaf_attachment import infer_leaf_attachments
from models.structure.skinning import compute_skinning
from simulation.streaming_sim import StreamingPlantSim


def quat_wxyz_to_R(q):
    """Quaternion (wxyz) -> 3x3 rotation matrix, vectorized over [N, 4]."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = (q * q).sum(-1).clamp_min(1e-12).sqrt()
    w, x, y, z = w / n, x / n, y / n, z / n
    R = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)
    return R


def build_covariances(scales: torch.Tensor, rotations_wxyz: torch.Tensor) -> torch.Tensor:
    """Per-Gaussian world-space covariance: cov = R diag(s)^2 R^T. Shape [N, 3, 3]."""
    R = quat_wxyz_to_R(rotations_wxyz)                          # [N, 3, 3]
    S2 = scales.pow(2)                                          # [N, 3]
    # cov = R * diag(s^2) * R^T = (R * s^2[None, :]) @ R^T
    cov = (R * S2.unsqueeze(-2)) @ R.transpose(-1, -2)
    return cov


def gs_world_y_up_to_viser_z_up(centers: np.ndarray) -> np.ndarray:
    """3DGS world: +Y up. viser default: +Z up. Permute axes for nicer default view."""
    # (x, y, z) -> (x, z, -y): rotate -90° about +X (so old +Y becomes new +Z)
    out = np.empty_like(centers)
    out[..., 0] = centers[..., 0]
    out[..., 1] = centers[..., 2]
    out[..., 2] = -centers[..., 1]
    return out


def world_to_gs(viser_pos: np.ndarray) -> np.ndarray:
    """Inverse of gs_world_y_up_to_viser_z_up for a single position."""
    x, y, z = viser_pos[..., 0], viser_pos[..., 1], viser_pos[..., 2]
    out = np.stack([x, -z, y], axis=-1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    parser.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    parser.add_argument('--params', required=True,
                        help='final_params.pt from optimize_per_scene (v11+).')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--dt', type=float, default=0.02,
                        help='Display frame dt; substeps run at dt/substeps.')
    parser.add_argument('--substeps', type=int, default=4)
    parser.add_argument('--skin-max-dist', type=float, default=0.3)
    parser.add_argument('--force-scale', type=float, default=20.0,
                        help='Drag-vector (m) -> force (N) multiplier. Tune to taste.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    print('Loading scene...')
    scene = load_gaussian_plant_scene(args.source, args.output_dir)
    tree = root_branch_graph(scene.branch, scene.tube)
    attachments = infer_leaf_attachments(scene.leaves, tree)
    skinning = compute_skinning(scene.app.xyz, tree, attachments, max_dist=args.skin_max_dist)

    params = torch.load(args.params, map_location=device, weights_only=False)
    sim = StreamingPlantSim(tree, attachments, skinning, params, scene.app.xyz, device)

    # Precompute static GS attributes
    scales = scene.app.scales.to(device)
    rotations = scene.app.rots.to(device)                       # wxyz
    opacities = scene.app.opacities.to(device)                  # [N, 1]
    colors = scene.app.colors.to(device)                        # [N, 3] in [0, 1]
    covariances = build_covariances(scales, rotations)          # [N, 3, 3]

    rgb_u8 = (colors.clamp(0, 1) * 255).byte().cpu().numpy()
    cov_np = covariances.cpu().numpy().astype(np.float32)
    opac_np = opacities.cpu().numpy().astype(np.float32)        # [N, 1]

    print(f'Starting viser on :{args.port} ...')
    server = viser.ViserServer(port=args.port)

    # ----- GUI -----
    with server.gui.add_folder('Interaction'):
        gui_status = server.gui.add_text('Status', initial_value='idle', disabled=True)
        gui_anchor = server.gui.add_text('Anchor node', initial_value='-', disabled=True)
        gui_force_scale = server.gui.add_slider(
            'Force scale', min=1.0, max=200.0, step=1.0, initial_value=args.force_scale,
        )
        gui_reset = server.gui.add_button('Reset plant')
        gui_pause = server.gui.add_checkbox('Pause sim', initial_value=False)
    with server.gui.add_folder('Physics'):
        # Read-only displays of the loaded params
        for key in ('log_k_stem', 'log_k_branch', 'log_c_stem', 'log_c_branch',
                    'log_inertia', 'log_k_petiole', 'log_c_petiole'):
            if key in params:
                server.gui.add_text(
                    key.replace('log_', ''), initial_value=f'{params[key].exp().item():.3f}',
                    disabled=True,
                )

    @gui_reset.on_click
    def _(_):
        with state_lock:
            sim.reset()
            state['force_world'] = torch.zeros(sim.N, 3)
            state['anchor'] = None
        gui_status.value = 'reset'
        gui_anchor.value = '-'

    # ----- Splat -----
    init_centers_viser = gs_world_y_up_to_viser_z_up(scene.app.xyz.cpu().numpy()).astype(np.float32)
    splat = server.scene.add_gaussian_splats(
        name='/plant',
        centers=init_centers_viser,
        covariances=cov_np,
        rgbs=rgb_u8,
        opacities=opac_np,
    )

    # Tree-node markers (small spheres at non-root nodes; helps drag pickup).
    # Hidden by default to keep visuals clean — can toggle via gui_show_nodes.
    gui_show_nodes = server.gui.add_checkbox('Show tree nodes', initial_value=False)
    # We add one icosphere per node — branch tree is small (~424 nodes).
    node_centers_viser = gs_world_y_up_to_viser_z_up(tree.nodes.cpu().numpy())
    node_handles = []
    bbox_size = float((tree.nodes.max(0).values - tree.nodes.min(0).values).max().item())
    marker_radius = max(bbox_size * 0.005, 0.005)
    for i, pos in enumerate(node_centers_viser):
        h = server.scene.add_icosphere(
            name=f'/nodes/n{i}',
            radius=marker_radius,
            color=(80, 180, 255),
            position=tuple(pos.tolist()),
            visible=False,
        )
        node_handles.append(h)

    @gui_show_nodes.on_update
    def _(_):
        for h in node_handles:
            h.visible = gui_show_nodes.value

    # ----- Shared state for the sim thread -----
    state_lock = threading.Lock()
    state = {
        'force_world': torch.zeros(sim.N, 3),    # CPU — chain runs on CPU
        'anchor': None,                          # int or None
        'drag_start_world': None,                # np.ndarray (GS coords) or None
    }
    tree_nodes_world_cpu = tree.nodes  # for nearest-node lookup at drag start

    # ----- Drag handler on the splat -----
    @splat.on_drag(button='left')
    def on_drag(event: viser.SceneNodeDragEvent):
        # Phases: 'start' (mouse down on object), 'drag' (continuous), 'end' (release).
        phase = event.phase
        start_viser = np.asarray(event.start_position)
        end_viser = np.asarray(event.end_position)
        start_gs = world_to_gs(start_viser)
        end_gs = world_to_gs(end_viser)

        if phase == 'start':
            # Find nearest tree node to the grab point (in GS world coords, CPU).
            with torch.no_grad():
                d = (tree_nodes_world_cpu - torch.as_tensor(start_gs, dtype=torch.float32)).norm(dim=-1)
                anchor = int(d.argmin().item())
            with state_lock:
                state['anchor'] = anchor
                state['drag_start_world'] = start_gs
                state['force_world'] = torch.zeros(sim.N, 3)
            gui_status.value = 'dragging'
            gui_anchor.value = str(anchor)
        elif phase == 'drag':
            with state_lock:
                anchor = state['anchor']
                if anchor is None:
                    return
                drag_vec = (end_gs - start_gs)
                scale = float(gui_force_scale.value)
                f = torch.zeros(sim.N, 3)
                f[anchor] = torch.as_tensor(drag_vec * scale, dtype=torch.float32)
                state['force_world'] = f
        elif phase == 'end':
            with state_lock:
                state['force_world'] = torch.zeros(sim.N, 3)
                state['anchor'] = None
                state['drag_start_world'] = None
            gui_status.value = 'idle'
            gui_anchor.value = '-'

    # ----- Background sim + render loop -----
    stop_event = threading.Event()

    def sim_loop():
        substep_dt = args.dt / args.substeps
        while not stop_event.is_set():
            t0 = time.time()
            if not gui_pause.value:
                with state_lock:
                    f = state['force_world'].clone()
                for _ in range(args.substeps):
                    ap_xyz, _, _ = sim.step(substep_dt, f)
                # Push centers to viser (handle is thread-safe internally).
                centers_np = gs_world_y_up_to_viser_z_up(ap_xyz.cpu().numpy().astype(np.float32))
                splat.centers = centers_np
            # ~30 Hz cap
            dt_left = args.dt - (time.time() - t0)
            if dt_left > 0:
                time.sleep(dt_left)

    th = threading.Thread(target=sim_loop, daemon=True)
    th.start()

    print(f'open http://localhost:{args.port}  (drag the plant with left mouse)')
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        stop_event.set()
        th.join(timeout=1.0)


if __name__ == '__main__':
    main()
