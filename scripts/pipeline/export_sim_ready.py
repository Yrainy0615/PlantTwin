"""Serialize a sim-ready plant bundle and demonstrate re-driving it under a new force.

Bundle contents (a single .pt file):
  scene:
    'ap_xyz', 'ap_scales', 'ap_rots', 'ap_opacities', 'ap_colors'
  tree:
    'nodes', 'parent', 'edges_oriented', 'edge_type', 'edge_radius',
    'edge_length', 'subtree_size', 'depth', 'root_idx'
  skinning:
    'bone_idx', 'leaf_idx', 'local_bone', 'local_leaf'
  attachments (per leaf):
    'leaf_parent_idx', 'leaf_child_idx',
    'petiole_offset_from_parent', 'leaf_disk_offset_from_surface'
  params (per-type, learned by the optimizer):
    'log_k_stem', 'log_k_branch', 'log_c_stem', 'log_c_branch', 'log_inertia'

Use `--bundle <bundle.pt>` to load and `--redrive-force-xy <fx> <fy>` plus
`--anchor <node_id>` to apply a constant lateral push during a fresh rollout.
The script renders the resulting trajectory and saves an .mp4.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.io as tvio

from data.gaussian_plant_loader import load_gaussian_plant_scene
from models.structure.graph_cleanup import root_branch_graph
from models.structure.leaf_attachment import infer_leaf_attachments
from models.structure.skinning import compute_skinning, apply_skinning, BoneSkinning
from simulation.articulated_chain import ArticulatedChain
from simulation.contact_force import ContactForceTrajectory


def build_bundle(source: str, output_dir: str, params_pt: str | None) -> dict:
    scene = load_gaussian_plant_scene(source, output_dir)
    tree = root_branch_graph(scene.branch, scene.tube)
    attachments = infer_leaf_attachments(scene.leaves, tree)
    skinning = compute_skinning(scene.app.xyz, tree, attachments)

    edges = tree.edges_oriented
    leaf_parent = torch.tensor([int(edges[a.parent_edge_idx, 0].item()) for a in attachments], dtype=torch.long)
    leaf_child = torch.tensor([int(edges[a.parent_edge_idx, 1].item()) for a in attachments], dtype=torch.long)
    petiole_offset = torch.stack(
        [a.surface_point - tree.nodes[leaf_parent[k]] for k, a in enumerate(attachments)], dim=0
    )
    leaf_disk_offset = torch.stack([a.rest_direction * a.rest_length for a in attachments], dim=0)

    bundle: dict = {
        'scene': {
            'ap_xyz': scene.app.xyz,
            'ap_scales': scene.app.scales,
            'ap_rots': scene.app.rots,
            'ap_opacities': scene.app.opacities,
            'ap_colors': scene.app.colors,
        },
        'tree': {
            'nodes': tree.nodes,
            'parent': tree.parent,
            'edges_oriented': tree.edges_oriented,
            'edge_type': tree.edge_type,
            'edge_radius': tree.edge_radius,
            'edge_length': tree.edge_length,
            'subtree_size': tree.subtree_size,
            'depth': tree.depth,
            'root_idx': tree.root_idx,
        },
        'skinning': {
            'bone_idx': skinning.bone_idx,
            'leaf_idx': skinning.leaf_idx,
            'local_bone': skinning.local_bone,
            'local_leaf': skinning.local_leaf,
        },
        'attachments': {
            'leaf_parent_idx': leaf_parent,
            'leaf_child_idx': leaf_child,
            'petiole_offset_from_parent': petiole_offset,
            'leaf_disk_offset_from_surface': leaf_disk_offset,
        },
    }
    if params_pt is not None:
        bundle['params'] = torch.load(params_pt, map_location='cpu')
    return bundle


def _rebuild_tree(tree_dict):
    """Build a RootedBranchTree-shaped object from the bundle dict (duck-typed)."""
    class _T:
        pass
    t = _T()
    t.nodes = tree_dict['nodes']
    t.parent = tree_dict['parent']
    t.edges_oriented = tree_dict['edges_oriented']
    t.edge_type = tree_dict['edge_type']
    t.edge_radius = tree_dict['edge_radius']
    t.edge_length = tree_dict['edge_length']
    t.subtree_size = tree_dict['subtree_size']
    t.depth = tree_dict['depth']
    t.root_idx = int(tree_dict['root_idx'])
    return t


def redrive(bundle: dict, anchor_node: int, force_xy: tuple[float, float],
            n_frames: int, dt: float, H: int, W: int,
            azimuth: float, elevation: float, device: torch.device,
            out_video: Path):
    tree = _rebuild_tree(bundle['tree'])
    N = tree.nodes.shape[0]
    chain = ArticulatedChain(tree).to(device)

    if 'params' in bundle:
        p = bundle['params']
        k_stem = p['log_k_stem'].exp().to(device)
        k_branch = p['log_k_branch'].exp().to(device)
        c_stem = p['log_c_stem'].exp().to(device)
        c_branch = p['log_c_branch'].exp().to(device)
        inertia = p['log_inertia'].exp().to(device)
    else:
        print('[redrive] no params in bundle, using defaults')
        k_stem = torch.tensor(50.0, device=device); k_branch = torch.tensor(15.0, device=device)
        c_stem = torch.tensor(1.5, device=device); c_branch = torch.tensor(0.8, device=device)
        inertia = torch.tensor(0.04, device=device)

    edges = bundle['tree']['edges_oriented'].to(device)
    etype = bundle['tree']['edge_type'].to(device)
    from models.structure.graph_cleanup import STEM
    per_edge_k = torch.where(etype == STEM, k_stem.expand(edges.shape[0]), k_branch.expand(edges.shape[0]))
    per_edge_c = torch.where(etype == STEM, c_stem.expand(edges.shape[0]), c_branch.expand(edges.shape[0]))
    k_per_node = torch.zeros(N, device=device).scatter(0, edges[:, 1], per_edge_k)
    c_per_node = torch.zeros(N, device=device).scatter(0, edges[:, 1], per_edge_c)
    I_per_node = torch.full((N,), inertia.item(), device=device)

    f_traj = torch.zeros(n_frames, N, 3, device=device)
    f_traj[:, anchor_node, 0] = force_xy[0]
    f_traj[:, anchor_node, 1] = force_xy[1]

    theta0 = torch.zeros(N, 3, device=device)
    omega0 = torch.zeros(N, 3, device=device)
    _, _, pos_t, rot_t = chain.rollout(theta0, omega0, k_per_node, c_per_node, I_per_node, f_traj, dt=dt)

    # Leaf disk pose per frame (rigid follow of parent bone)
    leaf_parent_idx = bundle['attachments']['leaf_parent_idx'].to(device)
    leaf_child_idx = bundle['attachments']['leaf_child_idx'].to(device)
    petiole_offset = bundle['attachments']['petiole_offset_from_parent'].to(device)
    disk_offset = bundle['attachments']['leaf_disk_offset_from_surface'].to(device)
    parent_pos_t = pos_t[:, leaf_parent_idx]
    child_rot_t = rot_t[:, leaf_child_idx]
    surface_t = parent_pos_t + torch.einsum('tlij,lj->tli', child_rot_t, petiole_offset)
    leaf_center_t = surface_t + torch.einsum('tlij,lj->tli', child_rot_t, disk_offset)
    leaf_rot_t = child_rot_t

    skinning = BoneSkinning(
        bone_idx=bundle['skinning']['bone_idx'].to(device),
        leaf_idx=bundle['skinning']['leaf_idx'].to(device),
        local_bone=bundle['skinning']['local_bone'].to(device),
        local_leaf=bundle['skinning']['local_leaf'].to(device),
    )
    ap_xyz_rest = bundle['scene']['ap_xyz'].to(device)
    scales = bundle['scene']['ap_scales'].to(device)
    rots = bundle['scene']['ap_rots'].to(device)
    opacities = bundle['scene']['ap_opacities'].to(device)
    colors = bundle['scene']['ap_colors'].to(device)

    ap_frames = []
    for t in range(pos_t.shape[0]):
        ap_t = apply_skinning(
            skinning, node_pos=pos_t[t], node_rot=rot_t[t], edges_oriented=edges,
            leaf_pos=leaf_center_t[t], leaf_rot=leaf_rot_t[t], ap_xyz_rest=ap_xyz_rest,
        )
        ap_frames.append(ap_t)
    ap_traj = torch.stack(ap_frames, dim=0)

    from models.renderer.gaussian_renderer import GaussianRenderer
    renderer = GaussianRenderer(image_height=H, image_width=W, sh_degree=0).to(device)
    target_pt = tree.nodes.mean(0).to(device)
    bbox_extent = (tree.nodes.max(0).values - tree.nodes.min(0).values).max().item()
    camera = renderer.get_camera(azimuth=azimuth, elevation=elevation,
                                  radius=float(2.0 * bbox_extent), target=target_pt)
    frames = []
    for t in range(ap_traj.shape[0]):
        frame = renderer.render_frame(ap_traj[t], scales, rots, opacities, colors, camera, shs=None)
        frames.append(frame)
    video = torch.stack(frames, dim=0).clamp(0, 1)
    video_u8 = (video.permute(0, 2, 3, 1) * 255).byte().cpu()
    tvio.write_video(str(out_video), video_u8, fps=8)
    print(f'  wrote redrive video {tuple(video.shape)} -> {out_video}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    parser.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    parser.add_argument('--params', default=None, help='Path to optimizer final_params.pt')
    parser.add_argument('--bundle-out', default='outputs/sim_ready/newplant9.pt')
    parser.add_argument('--bundle-in', default=None, help='Load an existing bundle and redrive.')
    parser.add_argument('--anchor', type=int, default=217)
    parser.add_argument('--force-xy', type=float, nargs=2, default=[0.4, 0.0])
    parser.add_argument('--frames', type=int, default=12)
    parser.add_argument('--dt', type=float, default=0.02)
    parser.add_argument('--H', type=int, default=192)
    parser.add_argument('--W', type=int, default=192)
    parser.add_argument('--azimuth', type=float, default=30.0)
    parser.add_argument('--elevation', type=float, default=15.0)
    parser.add_argument('--video-out', default='outputs/sim_ready/redrive.mp4')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    out_bundle = Path(args.bundle_out)
    out_bundle.parent.mkdir(parents=True, exist_ok=True)
    out_video = Path(args.video_out)
    out_video.parent.mkdir(parents=True, exist_ok=True)

    if args.bundle_in is None:
        print('Building sim-ready bundle...')
        bundle = build_bundle(args.source, args.output_dir, args.params)
        torch.save(bundle, out_bundle)
        print(f'  saved bundle -> {out_bundle}')
    else:
        bundle = torch.load(args.bundle_in, map_location='cpu')
        print(f'  loaded bundle <- {args.bundle_in}')

    print('Re-driving with new force...')
    redrive(bundle, args.anchor, tuple(args.force_xy), args.frames, args.dt,
            args.H, args.W, args.azimuth, args.elevation, torch.device(args.device), out_video)


if __name__ == '__main__':
    main()
