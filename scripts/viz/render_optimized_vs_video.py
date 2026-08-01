"""Render the optimized rollout under the COLMAP camera and stack it next to
the target video frame-by-frame. One quick eyeball pass at whether the learned
physics actually moves the plant in a video-like way."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision.io as tvio
from matplotlib.animation import FFMpegWriter
from torchvision.transforms.functional import resize

from data.gaussian_plant_loader import load_gaussian_plant_scene
from data.colmap_loader import (
    read_cameras_bin, read_images_bin, find_image_by_name, colmap_camera_to_renderer,
)
from models.structure.graph_cleanup import root_branch_graph, STEM
from models.structure.leaf_attachment import infer_leaf_attachments
from models.structure.skinning import compute_skinning, BoneSkinning, apply_skinning
from simulation.articulated_chain import ArticulatedChain, exp_so3
from simulation.contact_force import ContactForceTrajectory
from simulation.leaf_dynamics import LeafDynamicsBank


def _per_node_param(val_stem, val_branch, edges, etype, N, device):
    per_edge = torch.where(etype == STEM, val_stem.expand(edges.shape[0]),
                            val_branch.expand(edges.shape[0]))
    per_node = torch.zeros(N, device=device).scatter(0, edges[:, 1], per_edge)
    return per_node


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    parser.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    parser.add_argument('--params', required=True, help='final_params.pt from optimize_per_scene')
    parser.add_argument('--target-video', required=True)
    parser.add_argument('--colmap-image', default='IMG_1388.JPG')
    parser.add_argument('--frames', type=int, default=12)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--dt', type=float, default=0.02)
    parser.add_argument('--H', type=int, default=256)
    parser.add_argument('--W', type=int, default=256)
    parser.add_argument('--fps', type=int, default=8)
    parser.add_argument('--anchor', type=int, default=None,
                        help='Override anchor; default = read from params file if saved.')
    parser.add_argument('--contact-bundle', default=None,
                        help='Optional contact bundle (only used to read the saved anchor_node_id).')
    parser.add_argument('--out', default='outputs/per_scene_optim/optimized_vs_video.mp4')
    parser.add_argument('--skin-max-dist', type=float, default=0.3)
    parser.add_argument('--k-leaf', type=float, default=None,
                        help='Override saved k_leaf (per-edge spring stiffness inside a leaf).')
    parser.add_argument('--c-leaf', type=float, default=None,
                        help='Override saved c_leaf (per-edge damping).')
    parser.add_argument('--m-leaf', type=float, default=None,
                        help='Override saved m_leaf (per-node mass; smaller -> less inertia driving).')
    # v11 final method (docs/per_scene_v11_method.md §3.3) uses RIGID leaves:
    # each leaf's ApPs ride the disk frame via the petiole joint. The intra-leaf
    # KNN spring-mass path is deprecated — its IDW map produces "shattered leaf"
    # artifacts on sparse leaf clusters. Rigid is the default; opt in explicitly.
    parser.add_argument('--leaf-springmass', dest='leaf_springmass', action='store_true',
                        help='(deprecated) Enable intra-leaf KNN spring-mass. Off by default; '
                             'known to shatter sparse leaves.')
    parser.add_argument('--no-leaf-springmass', dest='leaf_springmass', action='store_false',
                        help='Rigid leaves via petiole joint (v11 final method; this is the default).')
    parser.set_defaults(leaf_springmass=False)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    scene = load_gaussian_plant_scene(args.source, args.output_dir)

    # Load params early — if a densified tree was saved (Phase 2), use it
    # instead of rebuilding from the raw branch graph. Tree topology stays on
    # CPU here; we move buffers per-call to the GPU later.
    p = torch.load(args.params, map_location=device, weights_only=False)
    if 'densified_tree' in p:
        from models.structure.graph_cleanup import RootedBranchTree
        dt = p['densified_tree']
        tree = RootedBranchTree(
            nodes=dt['nodes'].cpu(), root_idx=int(dt['root_idx']),
            parent=dt['parent'].cpu(), depth=dt['depth'].cpu(),
            edges_oriented=dt['edges_oriented'].cpu(), edge_length=dt['edge_length'].cpu(),
            edge_radius=dt['edge_radius'].cpu(), edge_type=dt['edge_type'].cpu(),
            subtree_size=dt['subtree_size'].cpu(),
        )
        print(f'[render] using densified tree from params: N={tree.nodes.shape[0]}')
    else:
        tree = root_branch_graph(scene.branch, scene.tube)
    attachments = infer_leaf_attachments(scene.leaves, tree)
    skinning = compute_skinning(scene.app.xyz, tree, attachments, max_dist=args.skin_max_dist)

    # COLMAP camera
    sparse = Path(args.source) / 'sparse' / '0'
    cams = read_cameras_bin(sparse / 'cameras.bin')
    imgs = read_images_bin(sparse / 'images.bin')
    rec = find_image_by_name(imgs, args.colmap_image)
    cam = colmap_camera_to_renderer(rec, cams[rec['cam_id']], args.H, args.W)
    cam = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cam.items()}

    # p already loaded above (for optional densified_tree).
    k_stem = p['log_k_stem'].exp()
    k_branch = p['log_k_branch'].exp()
    c_stem = p['log_c_stem'].exp()
    c_branch = p['log_c_branch'].exp()
    inertia = p['log_inertia'].exp()
    k_petiole = p['log_k_petiole'].exp() if 'log_k_petiole' in p else None
    c_petiole = p['log_c_petiole'].exp() if 'log_c_petiole' in p else None
    k_leaf = p['log_k_leaf'].exp().to(device) if 'log_k_leaf' in p else None
    c_leaf = p['log_c_leaf'].exp().to(device) if 'log_c_leaf' in p else None
    m_leaf = p['log_m_leaf'].exp().to(device) if 'log_m_leaf' in p else None
    if args.k_leaf is not None:
        k_leaf = torch.tensor(args.k_leaf, device=device)
        print(f'[render] override k_leaf = {args.k_leaf}')
    if args.c_leaf is not None:
        c_leaf = torch.tensor(args.c_leaf, device=device)
        print(f'[render] override c_leaf = {args.c_leaf}')
    if args.m_leaf is not None:
        m_leaf = torch.tensor(args.m_leaf, device=device)
        print(f'[render] override m_leaf = {args.m_leaf}')
    force = p['contact_force'].to(device)                    # [T, N_anchor, 3]

    anchor_node_id = args.anchor
    if anchor_node_id is None and 'anchor_node_id' in p:
        anchor_node_id = int(p['anchor_node_id'])
    if anchor_node_id is None and args.contact_bundle is not None:
        b = torch.load(args.contact_bundle, map_location='cpu', weights_only=False)
        ac = b['anchor_node_id']
        valid = ac[ac >= 0]
        anchor_node_id = int(torch.bincount(valid).argmax().item())
    if anchor_node_id is None:
        raise SystemExit('Pass --anchor, --contact-bundle, or use params with saved anchor_node_id.')

    chain = ArticulatedChain(tree).to(device)
    edges = tree.edges_oriented.to(device)
    etype = tree.edge_type.to(device)
    N = tree.nodes.shape[0]
    k_pn = _per_node_param(k_stem, k_branch, edges, etype, N, device)
    c_pn = _per_node_param(c_stem, c_branch, edges, etype, N, device)
    I_pn = inertia.expand(N).clone().to(device)

    # Optional learned structure delta: rest_pos_eff = tree.nodes + delta.
    delta_rest_pos = None
    if 'delta_rest_pos' in p:
        delta_rest_pos = p['delta_rest_pos'].to(device)
        if delta_rest_pos.shape[0] != N:
            print(f'[render] WARNING delta_rest_pos shape {delta_rest_pos.shape} != tree N={N}; ignoring')
            delta_rest_pos = None
    rest_pos_eff = tree.nodes.to(device) + delta_rest_pos if delta_rest_pos is not None else None

    T = min(args.frames, force.shape[0])
    f_traj = torch.zeros(T, N, 3, device=device)
    f_traj[:, anchor_node_id, :] = force[:T, 0, :]

    theta0 = torch.zeros(N, 3, device=device)
    omega0 = torch.zeros(N, 3, device=device)
    _, _, pos_t, rot_t = chain.rollout(
        theta0, omega0, k_pn, c_pn, I_pn, f_traj, dt=args.dt, rest_pos=rest_pos_eff,
    )

    # Leaf disk poses: petiole damped oscillator on top of the parent bone, then
    # IDW-deformed leaf via the spring-mass bank (mirrors optimize_per_scene.forward).
    leaf_parent = torch.tensor([int(edges[a.parent_edge_idx, 0].item()) for a in attachments],
                                dtype=torch.long, device=device)
    leaf_child = torch.tensor([int(edges[a.parent_edge_idx, 1].item()) for a in attachments],
                               dtype=torch.long, device=device)
    petiole = torch.stack(
        [a.surface_point - tree.nodes[int(edges[a.parent_edge_idx, 0].item())] for a in attachments]
    ).to(device)
    disk_off = torch.stack([a.rest_direction * a.rest_length for a in attachments]).to(device)

    child_rot_t = rot_t[:, leaf_child]                                    # [T, N_l, 3, 3]
    surface_t = pos_t[:, leaf_parent] + torch.einsum('tlij,lj->tli', child_rot_t, petiole)

    if k_petiole is not None and c_petiole is not None:
        # Re-roll petiole oscillator with the same scheme as the optimizer.
        Tn, N_leaf = child_rot_t.shape[0], child_rot_t.shape[1]
        kp = k_petiole.to(device); cp = c_petiole.to(device); dt = args.dt
        theta = torch.zeros(N_leaf, 3, device=device)
        omega = torch.zeros(N_leaf, 3, device=device)
        omega_parent_prev = torch.zeros(N_leaf, 3, device=device)
        theta_hist = []
        for t in range(Tn):
            if t == 0:
                alpha_parent = torch.zeros(N_leaf, 3, device=device)
                omega_parent = torch.zeros(N_leaf, 3, device=device)
            else:
                R = child_rot_t[t]; Rp = child_rot_t[t - 1]
                Rd = R @ Rp.transpose(-1, -2)
                omega_parent = torch.stack([
                    Rd[..., 2, 1] - Rd[..., 1, 2],
                    Rd[..., 0, 2] - Rd[..., 2, 0],
                    Rd[..., 1, 0] - Rd[..., 0, 1],
                ], dim=-1) * 0.5 / dt
                alpha_parent = (omega_parent - omega_parent_prev) / dt
            tau = -kp * theta - cp * omega - alpha_parent
            omega = omega + dt * tau
            theta = theta + dt * omega
            theta_hist.append(theta)
            omega_parent_prev = omega_parent
        theta_pet_t = torch.stack(theta_hist, dim=0)                       # [T, N_l, 3]
        R_pet_t = exp_so3(theta_pet_t.reshape(-1, 3)).reshape(Tn, N_leaf, 3, 3)
        leaf_rot_t = torch.einsum('tlij,tljk->tlik', child_rot_t, R_pet_t)
    else:
        leaf_rot_t = child_rot_t

    leaf_center_t = surface_t + torch.einsum('tlij,lj->tli', leaf_rot_t, disk_off)

    skin_local_bone = skinning.local_bone.to(device)
    skin_local_leaf = skinning.local_leaf.to(device)
    # Apply delta correction: local_eff = local - delta[parent_node_of_binding]
    # so canonical-pose rendering is preserved (delta only bends motion).
    if delta_rest_pos is not None:
        bm = skinning.bone_idx.to(device) >= 0
        if bm.any():
            par_for_bone_ap = edges[skinning.bone_idx.to(device)[bm].clamp_min(0), 0]
            skin_local_bone = skin_local_bone.clone()
            skin_local_bone[bm] = skin_local_bone[bm] - delta_rest_pos[par_for_bone_ap]
        lm = skinning.leaf_idx.to(device) >= 0
        if lm.any():
            leaf_parent_for_ap = leaf_parent[skinning.leaf_idx.to(device)[lm].clamp_min(0)]
            skin_local_leaf = skin_local_leaf.clone()
            skin_local_leaf[lm] = skin_local_leaf[lm] - delta_rest_pos[leaf_parent_for_ap]
    skinning_d = BoneSkinning(
        bone_idx=skinning.bone_idx.to(device),
        leaf_idx=skinning.leaf_idx.to(device),
        local_bone=skin_local_bone,
        local_leaf=skin_local_leaf,
    )
    ap_xyz_rest = scene.app.xyz.to(device)
    scales = scene.app.scales.to(device)
    rotations = scene.app.rots.to(device)
    opacities = scene.app.opacities.to(device)
    colors = scene.app.colors.to(device)

    # Leaf intra-disk spring-mass rollout (skip if params not saved or user asked to).
    ap_leaf_world = None
    leaf_bound_idx = None
    if args.leaf_springmass and k_leaf is not None and c_leaf is not None and m_leaf is not None and len(attachments) > 0:
        # Build on CPU (KNN done in init), then move with .to(device).
        leaves_local = [
            scene.leaves[a.leaf_idx].points - a.disk_center for a in attachments
        ]
        pin_local = [a.surface_point - a.disk_center for a in attachments]
        leaf_substeps = 4
        leaf_dt = args.dt / leaf_substeps
        leaf_bank = LeafDynamicsBank(
            leaves_local,
            ap_local_leaf=skinning.local_leaf,
            ap_leaf_idx=skinning.leaf_idx,
            pin_local_list=pin_local,
            k_neighbors=6,
            k_neighbors_ap=4,
            pin_top_k=3,
            dt=leaf_dt,
            n_substeps=leaf_substeps,
        ).to(device)
        with torch.no_grad():
            traj_local = leaf_bank.rollout(
                leaf_center_t, leaf_rot_t, k_leaf, c_leaf, m_leaf,
            )
            ap_leaf_world = leaf_bank.map_ap_world(
                traj_local, skinning_d.local_leaf, leaf_center_t, leaf_rot_t,
            )
        leaf_bound_idx = leaf_bank.leaf_bound_mask.nonzero(as_tuple=False).flatten()

    from models.renderer.gaussian_renderer import GaussianRenderer
    renderer = GaussianRenderer(image_height=args.H, image_width=args.W, sh_degree=0).to(device)

    frames_target, _, _ = tvio.read_video(args.target_video, pts_unit='sec')
    frames_target = frames_target[args.start_frame:args.start_frame + T].permute(0, 3, 1, 2).float() / 255.0
    frames_target = resize(frames_target, [args.H, args.W])

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(args.W * 2 / 80.0, args.H / 80.0), dpi=80)
    writer = FFMpegWriter(fps=args.fps)
    with writer.saving(fig, str(out_path), dpi=80):
        for t in range(T):
            ap_t = apply_skinning(skinning_d, pos_t[t], rot_t[t], edges,
                                   leaf_center_t[t], leaf_rot_t[t], ap_xyz_rest)
            if ap_leaf_world is not None and leaf_bound_idx is not None and leaf_bound_idx.numel() > 0:
                ap_t = ap_t.clone()
                ap_t[leaf_bound_idx] = ap_leaf_world[t]
            frame = renderer.render_frame(ap_t, scales, rotations, opacities, colors, cam, shs=None)
            frame = frame.clamp(0, 1).detach().cpu()
            axes[0].clear(); axes[0].imshow(frame.permute(1, 2, 0).numpy())
            axes[0].set_title('optimized rollout', fontsize=8); axes[0].set_axis_off()
            axes[1].clear(); axes[1].imshow(frames_target[t].permute(1, 2, 0).numpy())
            axes[1].set_title('target video', fontsize=8); axes[1].set_axis_off()
            writer.grab_frame()
    plt.close(fig)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
