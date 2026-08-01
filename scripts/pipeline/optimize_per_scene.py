"""Per-scene joint optimization of plant physics + contact force.

Pipeline:
1. Load GaussianPlant scene (ApP + branch graph + leaf clusters).
2. Stage 0: root the branch graph, fit per-segment radii, infer leaf attachments.
3. Compute LBS skinning (each ApP -> a bone or a leaf).
4. Build the per-scene physics model:
   - per-type joint stiffness / damping (stem, branch) and global inertia
   - learnable per-frame 3D contact force at a chosen anchor node
5. Forward pass per iteration:
   - rollout articulated chain
   - propagate leaf disk frames rigidly with their parent bone (v0)
   - LBS ApPs into a [T, N_ap, 3] trajectory
   - render the trajectory through the gaussian renderer
6. Loss:
   - RGB L2 vs. target video
   - contact-force temporal smoothness + sparsity regularizers
7. Adam update.

Modes:
  --synthetic    : generate the target video by forward sim with ground-truth
                   params, then optimize from random init to recover them.
                   Validates pipeline + identifiability without real video.
  --target <pt>  : load a precomputed target video tensor (.pt with shape
                   [T, 3, H, W] in [0, 1]) and optimize.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch import nn

from data.gaussian_plant_loader import load_gaussian_plant_scene
from models.structure.graph_cleanup import root_branch_graph, STEM
from models.structure.leaf_attachment import infer_leaf_attachments
from models.structure.skinning import compute_skinning, apply_skinning
from simulation.articulated_chain import ArticulatedChain
from simulation.contact_force import ContactForceTrajectory
from simulation.leaf_dynamics import LeafDynamicsBank
from optimization.video_loss import VideoLossBundle


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PerScenePhysicsModel(nn.Module):
    """Holds learnable physics + contact and produces ApP world trajectories."""

    def __init__(
        self,
        tree,
        attachments,
        skinning,
        ap_xyz_rest: torch.Tensor,
        n_frames: int,
        anchor_node_id: int,
        leaf_clusters=None,
        dt: float = 0.01,
        init: dict | None = None,
        leaf_dyn_substeps: int = 4,
        learn_struct: bool = False,
    ):
        super().__init__()
        self.tree = tree
        self.attachments = attachments
        self.skinning = skinning
        self.n_frames = n_frames
        self.dt = dt
        self.use_leaf_dynamics = leaf_clusters is not None and len(attachments) > 0
        self.learn_struct = learn_struct

        N = tree.nodes.shape[0]
        self.chain = ArticulatedChain(tree)

        init = init or {}
        self.log_k_stem = nn.Parameter(torch.log(torch.tensor(init.get('k_stem', 20.0))))
        self.log_k_branch = nn.Parameter(torch.log(torch.tensor(init.get('k_branch', 8.0))))
        self.log_c_stem = nn.Parameter(torch.log(torch.tensor(init.get('c_stem', 0.5))))
        self.log_c_branch = nn.Parameter(torch.log(torch.tensor(init.get('c_branch', 0.3))))
        self.log_inertia = nn.Parameter(torch.log(torch.tensor(init.get('inertia', 0.02))))
        # Petiole joint: per-leaf torsional spring, driven by parent-bone acceleration.
        # I (inertia) is normalized to 1; k is natural frequency^2, c is 2ζω damping coeff.
        self.log_k_petiole = nn.Parameter(torch.log(torch.tensor(init.get('k_petiole', 80.0))))
        self.log_c_petiole = nn.Parameter(torch.log(torch.tensor(init.get('c_petiole', 3.0))))
        # Per-leaf KNN spring-mass (intra-disk membrane deformation).
        # k_leaf is per-edge spring constant; c_leaf per-edge damping; m_leaf per-node mass.
        # Driven by base excitation (disk-center acceleration) in disk-local frame.
        self.log_k_leaf = nn.Parameter(torch.log(torch.tensor(init.get('k_leaf', 20.0))))
        self.log_c_leaf = nn.Parameter(torch.log(torch.tensor(init.get('c_leaf', 2.0))))
        self.log_m_leaf = nn.Parameter(torch.log(torch.tensor(init.get('m_leaf', 0.05))))

        self.contact = ContactForceTrajectory(
            n_frames, N, [anchor_node_id], init_scale=init.get('force_init_scale', 0.05),
        )

        if self.use_leaf_dynamics:
            # Disk-local rest points per leaf: cluster - disk_center.
            leaves_local = [
                leaf_clusters[a.leaf_idx].points - a.disk_center for a in attachments
            ]
            # Pin location in disk-local: petiole attachment = surface_point - disk_center
            # = -rest_direction * rest_length. Cluster nodes nearest this point get
            # rigidly pinned to the disk frame (no spring DOF), so the rest of the
            # leaf wobbles around the petiole anchor instead of drifting bodily.
            pin_local = [
                (a.surface_point - a.disk_center) for a in attachments
            ]
            # Inner dt: split dt across n_substeps so each output frame advances by `dt`.
            leaf_dt = dt / leaf_dyn_substeps
            self.leaf_bank = LeafDynamicsBank(
                leaves_local,
                ap_local_leaf=skinning.local_leaf,
                ap_leaf_idx=skinning.leaf_idx,
                pin_local_list=pin_local,
                k_neighbors=6,
                k_neighbors_ap=4,
                pin_top_k=3,
                dt=leaf_dt,
                n_substeps=leaf_dyn_substeps,
            )

        # Buffers used in forward — moved to device alongside the module
        self.register_buffer('ap_xyz_rest', ap_xyz_rest)
        self.register_buffer('edges_oriented', tree.edges_oriented)
        self.register_buffer('edge_type', tree.edge_type)
        self.register_buffer('skinning_bone_idx', skinning.bone_idx)
        self.register_buffer('skinning_leaf_idx', skinning.leaf_idx)
        self.register_buffer('skinning_local_bone', skinning.local_bone)
        self.register_buffer('skinning_local_leaf', skinning.local_leaf)

        edges = tree.edges_oriented
        # Per-leaf: parent / child node indices on the attachment edge
        leaf_parent = torch.tensor([int(edges[a.parent_edge_idx, 0].item()) for a in attachments], dtype=torch.long)
        leaf_child = torch.tensor([int(edges[a.parent_edge_idx, 1].item()) for a in attachments], dtype=torch.long)
        # Petiole anchor offset in parent's frame at rest (parent's rest frame == world frame at rest)
        petiole_offset_from_parent = torch.stack(
            [a.surface_point - tree.nodes[leaf_parent[k]] for k, a in enumerate(attachments)],
            dim=0,
        )
        # disk_center - surface_point at rest = rest_dir * rest_length
        leaf_disk_offset_from_surface = torch.stack(
            [a.rest_direction * a.rest_length for a in attachments],
            dim=0,
        )
        self.register_buffer('leaf_parent_idx', leaf_parent)
        self.register_buffer('leaf_child_idx', leaf_child)
        self.register_buffer('petiole_offset_from_parent', petiole_offset_from_parent)
        self.register_buffer('leaf_disk_offset_from_surface', leaf_disk_offset_from_surface)

        # Structure-refinement params: a learnable delta on per-node rest position.
        # When applied, we recompute LBS local offsets to preserve canonical-pose
        # rendering exactly (the delta only changes how the kinematic chain bends).
        if self.learn_struct:
            self.delta_rest_pos = nn.Parameter(torch.zeros(N, 3))
        else:
            self.register_buffer('delta_rest_pos', torch.zeros(N, 3))

        # Pre-compute the parent-node index for each bone-bound and leaf-bound ApP
        # (used to subtract delta[parent] from the baked local offset in the LBS).
        bone_mask = skinning.bone_idx >= 0
        leaf_mask = skinning.leaf_idx >= 0
        parent_for_bone_ap = torch.zeros_like(skinning.bone_idx)
        if bone_mask.any():
            parent_for_bone_ap[bone_mask] = edges[skinning.bone_idx[bone_mask], 0]
        leaf_attach_parent_for_leaf_ap = torch.zeros_like(skinning.leaf_idx)
        if leaf_mask.any():
            leaf_attach_parent_for_leaf_ap[leaf_mask] = leaf_parent[skinning.leaf_idx[leaf_mask]]
        self.register_buffer('parent_node_per_bone_ap', parent_for_bone_ap)
        self.register_buffer('leaf_attach_parent_per_leaf_ap', leaf_attach_parent_for_leaf_ap)
        # Edge-list used for the structure smoothness loss (parent <-> child).
        self.register_buffer('struct_edges', tree.edges_oriented)

    def _per_node_param(self, val_stem: torch.Tensor, val_branch: torch.Tensor) -> torch.Tensor:
        N = self.tree.nodes.shape[0]
        edges = self.edges_oriented
        per_edge = torch.where(self.edge_type == STEM, val_stem.expand(edges.shape[0]),
                                val_branch.expand(edges.shape[0]))
        per_node = torch.zeros(N, device=val_stem.device, dtype=val_stem.dtype)
        per_node = per_node.scatter(0, edges[:, 1], per_edge)
        return per_node

    def _rollout_petiole(self, child_rot_t: torch.Tensor) -> torch.Tensor:
        """Base-excited damped oscillator per leaf, driven by parent bone's
        angular acceleration. Returns theta_petiole_history [T, N_leaf, 3] —
        the petiole rotation expressed in the parent bone's frame.
        """
        from simulation.articulated_chain import exp_so3
        T, N_leaf = child_rot_t.shape[0], child_rot_t.shape[1]
        k = self.log_k_petiole.exp()
        c = self.log_c_petiole.exp()
        dt = self.dt
        device = child_rot_t.device

        theta = torch.zeros(N_leaf, 3, device=device)
        omega = torch.zeros(N_leaf, 3, device=device)
        omega_parent_prev = torch.zeros(N_leaf, 3, device=device)
        history = []

        for t in range(T):
            if t == 0:
                alpha_parent = torch.zeros(N_leaf, 3, device=device)
                omega_parent = torch.zeros(N_leaf, 3, device=device)
            else:
                R = child_rot_t[t]
                R_prev = child_rot_t[t - 1]
                # axis-angle of delta R via skew-symm vee; small-angle accurate
                R_delta = R @ R_prev.transpose(-1, -2)
                omega_parent = torch.stack([
                    R_delta[..., 2, 1] - R_delta[..., 1, 2],
                    R_delta[..., 0, 2] - R_delta[..., 2, 0],
                    R_delta[..., 1, 0] - R_delta[..., 0, 1],
                ], dim=-1) * 0.5 / dt
                alpha_parent = (omega_parent - omega_parent_prev) / dt
            # I=1 implicit: omega' = -k*theta - c*omega - alpha_parent
            tau = -k * theta - c * omega - alpha_parent
            omega = omega + dt * tau
            theta = theta + dt * omega
            history.append(theta)
            omega_parent_prev = omega_parent

        return torch.stack(history, dim=0)

    def _skinning_dataclass(self):
        # Return a lightweight wrapper compatible with apply_skinning
        from models.structure.skinning import BoneSkinning
        return BoneSkinning(
            bone_idx=self.skinning_bone_idx,
            leaf_idx=self.skinning_leaf_idx,
            local_bone=self.skinning_local_bone,
            local_leaf=self.skinning_local_leaf,
        )

    def forward(self):
        N = self.tree.nodes.shape[0]

        k_stem = self.log_k_stem.exp()
        k_branch = self.log_k_branch.exp()
        c_stem = self.log_c_stem.exp()
        c_branch = self.log_c_branch.exp()
        inertia = self.log_inertia.exp()

        k_per_node = self._per_node_param(k_stem, k_branch)
        c_per_node = self._per_node_param(c_stem, c_branch)
        I_per_node = inertia.expand(N).clone()

        device = k_stem.device
        theta0 = torch.zeros(N, 3, device=device)
        omega0 = torch.zeros(N, 3, device=device)

        # If learning structure, use rest_pos_eff = tree.nodes + delta everywhere
        # the rest geometry enters (FK, petiole offset, LBS local offsets).
        rest_pos_eff = self.chain.rest_pos + self.delta_rest_pos if self.learn_struct else None

        f_traj = self.contact()
        theta_t, omega_t, pos_t, rot_t = self.chain.rollout(
            theta0, omega0, k_per_node, c_per_node, I_per_node, f_traj, dt=self.dt,
            rest_pos=rest_pos_eff,
        )

        # Leaf disk pose per frame: surface point follows parent bone rigidly,
        # but disk rotation has a petiole joint relative to the parent (base-
        # excited damped oscillator driven by parent's angular acceleration).
        parent_pos_t = pos_t[:, self.leaf_parent_idx]
        child_rot_t = rot_t[:, self.leaf_child_idx]
        surface_t = parent_pos_t + torch.einsum(
            'tlij,lj->tli', child_rot_t, self.petiole_offset_from_parent
        )

        from simulation.articulated_chain import exp_so3
        theta_petiole_t = self._rollout_petiole(child_rot_t)                # [T, N_leaf, 3]
        T_eff, N_leaf = theta_petiole_t.shape[0], theta_petiole_t.shape[1]
        R_petiole_t = exp_so3(theta_petiole_t.reshape(-1, 3)).reshape(T_eff, N_leaf, 3, 3)
        leaf_rot_t = torch.einsum('tlij,tljk->tlik', child_rot_t, R_petiole_t)
        leaf_center_t = surface_t + torch.einsum(
            'tlij,lj->tli', leaf_rot_t, self.leaf_disk_offset_from_surface
        )

        # Per-leaf intra-disk spring-mass deformation: returns disk-local
        # displacements that get IDW-interpolated onto each leaf-bound ApP.
        ap_leaf_world = None
        if self.use_leaf_dynamics:
            k_leaf = self.log_k_leaf.exp()
            c_leaf = self.log_c_leaf.exp()
            m_leaf = self.log_m_leaf.exp()
            traj_local = self.leaf_bank.rollout(
                leaf_center_t, leaf_rot_t, k_leaf, c_leaf, m_leaf,
            )
            ap_leaf_world = self.leaf_bank.map_ap_world(
                traj_local, self.skinning_local_leaf, leaf_center_t, leaf_rot_t,
            )                                                       # [T, N_a_leaf, 3]
            leaf_bound_mask = self.leaf_bank.leaf_bound_mask        # [N_ap]
            leaf_bound_idx = leaf_bound_mask.nonzero(as_tuple=False).flatten()

        # LBS per frame. When learning structure, subtract delta_rest_pos[parent]
        # from each ApP's baked local offset so canonical-pose rendering is
        # preserved (delta only affects how the chain bends in motion).
        skinning = self._skinning_dataclass()
        ap_frames = []
        bone_mask = self.skinning_bone_idx >= 0
        leaf_mask = self.skinning_leaf_idx >= 0
        bone_idx_safe = self.skinning_bone_idx.clamp_min(0)
        leaf_idx_safe = self.skinning_leaf_idx.clamp_min(0)
        for t in range(pos_t.shape[0]):
            ap_t = self.ap_xyz_rest.clone()
            # Bone path
            if bone_mask.any():
                parent_idx_for_ap = self.parent_node_per_bone_ap[bone_mask]
                child_idx_for_ap = self.edges_oriented[bone_idx_safe[bone_mask], 1]
                if self.learn_struct:
                    local_eff = self.skinning_local_bone[bone_mask] - self.delta_rest_pos[parent_idx_for_ap]
                else:
                    local_eff = self.skinning_local_bone[bone_mask]
                R = rot_t[t, child_idx_for_ap]
                parent_p = pos_t[t, parent_idx_for_ap]
                ap_t = ap_t.clone()
                ap_t[bone_mask] = parent_p + (R @ local_eff.unsqueeze(-1)).squeeze(-1)
            # Leaf path
            if leaf_mask.any():
                R = leaf_rot_t[t, leaf_idx_safe[leaf_mask]]
                center = leaf_center_t[t, leaf_idx_safe[leaf_mask]]
                if self.learn_struct:
                    leaf_parent_for_ap = self.leaf_attach_parent_per_leaf_ap[leaf_mask]
                    local_eff = self.skinning_local_leaf[leaf_mask] - self.delta_rest_pos[leaf_parent_for_ap]
                else:
                    local_eff = self.skinning_local_leaf[leaf_mask]
                ap_t = ap_t.clone()
                ap_t[leaf_mask] = center + (R @ local_eff.unsqueeze(-1)).squeeze(-1)
            if ap_leaf_world is not None and leaf_bound_idx.numel() > 0:
                ap_t = ap_t.clone()
                ap_t[leaf_bound_idx] = ap_leaf_world[t]
            ap_frames.append(ap_t)
        ap_traj = torch.stack(ap_frames, dim=0)

        return {
            'ap_traj': ap_traj,
            'node_pos_traj': pos_t,
            'node_rot_traj': rot_t,
            'theta_traj': theta_t,
            'contact_force': f_traj,
        }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _build_renderer(H: int, W: int, device: torch.device, sh_degree: int = 0):
    """Lazy import to keep the rasterizer optional for non-rendering smoke tests."""
    from models.renderer.gaussian_renderer import GaussianRenderer
    return GaussianRenderer(image_height=H, image_width=W, sh_degree=sh_degree).to(device)


def _gaussian_camera(renderer, target: torch.Tensor, radius: float, azim_deg: float, elev_deg: float):
    return renderer.get_camera(
        azimuth=azim_deg, elevation=elev_deg, radius=radius, target=target,
    )


def _colmap_camera(source_dir: str, image_name: str, H: int, W: int, device: torch.device):
    from data.colmap_loader import (
        read_cameras_bin, read_images_bin, find_image_by_name, colmap_camera_to_renderer,
    )
    sparse = Path(source_dir) / 'sparse' / '0'
    cams = read_cameras_bin(sparse / 'cameras.bin')
    imgs = read_images_bin(sparse / 'images.bin')
    rec = find_image_by_name(imgs, image_name)
    if rec is None:
        raise SystemExit(f'{image_name} not in COLMAP images.bin')
    camera = colmap_camera_to_renderer(rec, cams[rec['cam_id']], H, W)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in camera.items()}


def _load_video_target(path: str, H: int, W: int, n_frames: int, device: torch.device,
                        start_frame: int = 0) -> torch.Tensor:
    import torchvision.io as tvio
    from torchvision.transforms.functional import resize
    frames, _, _ = tvio.read_video(path, pts_unit='sec')
    frames = frames.permute(0, 3, 1, 2).float() / 255.0
    frames = frames[start_frame:start_frame + n_frames]
    frames = resize(frames, [H, W])
    return frames.to(device).clamp(0, 1)


def project_world_to_pixels(points_world: torch.Tensor, camera: dict, H: int, W: int) -> torch.Tensor:
    """Project [N, 3] world points to [N, 2] pixel coordinates using the same
    perspective convention as the rasterizer."""
    P = camera['proj_matrix'].T
    homog = torch.cat([points_world, torch.ones(points_world.shape[0], 1, device=points_world.device)], dim=-1)
    clip = (P @ homog.T).T
    w = clip[:, 3:4].clamp_min(1e-6)
    ndc = clip[:, :3] / w
    u = (ndc[:, 0] * 0.5 + 0.5) * W
    v = (ndc[:, 1] * 0.5 + 0.5) * H
    return torch.stack([u, v], dim=-1)


def render_video(
    renderer,
    ap_traj: torch.Tensor,            # [T, N, 3]
    scales: torch.Tensor,             # [N, 3]
    rotations: torch.Tensor,          # [N, 4]
    opacities: torch.Tensor,          # [N, 1]
    colors: torch.Tensor,             # [N, 3]
    camera,
) -> torch.Tensor:
    frames = []
    for t in range(ap_traj.shape[0]):
        frame = renderer.render_frame(
            ap_traj[t], scales, rotations, opacities, colors, camera, shs=None,
        )
        frames.append(frame)
    return torch.stack(frames, dim=0)  # [T, 3, H, W]


# ---------------------------------------------------------------------------
# Synthetic target generation
# ---------------------------------------------------------------------------


def synthesize_target(
    model: PerScenePhysicsModel,
    renderer,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    camera,
    gt_params: dict,
    gt_force: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Run forward sim with known ground-truth params + force, return rendered video."""
    # Temporarily replace parameters
    saved = {}
    with torch.no_grad():
        for key, val in gt_params.items():
            saved[key] = getattr(model, key).data.clone()
            getattr(model, key).data.copy_(torch.log(torch.tensor(val)))
        saved['contact_force'] = model.contact.force.data.clone()
        model.contact.force.data.copy_(gt_force)
        out = model()
        video = render_video(renderer, out['ap_traj'], scales, rotations, opacities, colors, camera)
        video = video.clamp(0, 1)

    # Restore
    with torch.no_grad():
        for key, val in saved.items():
            if key == 'contact_force':
                model.contact.force.data.copy_(val)
            else:
                getattr(model, key).data.copy_(val)
    return video, out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    parser.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--start-frame', type=int, default=0,
                        help='Skip the first N frames of the target video / contact bundle. '
                             'Useful when the hand only enters at frame K (use --start-frame K).')
    parser.add_argument('--dt', type=float, default=0.02)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-2)
    parser.add_argument('--H', type=int, default=128)
    parser.add_argument('--W', type=int, default=128)
    parser.add_argument('--azimuth', type=float, default=30.0)
    parser.add_argument('--elevation', type=float, default=15.0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--synthetic', action='store_true', help='Generate target by forward-simulating GT, then optimize.')
    parser.add_argument('--target', type=str, default=None, help='Load target video tensor (.pt) instead of synthesizing.')
    parser.add_argument('--target-video', type=str, default=None,
                        help='Path to a real .mp4 target video. Loaded into a [T, 3, H, W] tensor.')
    parser.add_argument('--colmap-image', type=str, default=None,
                        help='When set, use the COLMAP camera for the named image (e.g. IMG_1388.JPG).')
    parser.add_argument('--contact-bundle', type=str, default=None,
                        help='Path to contact.pt from precompute_contact.py. Provides per-frame anchor + stem-track supervision.')
    parser.add_argument('--w-stem-proj', type=float, default=0.0,
                        help='Weight of the stem-track 2D projection loss (requires --contact-bundle).')
    parser.add_argument('--w-pin', type=float, default=0.0,
                        help='Weight of the contact-pin loss (anchor projection should match per-frame pin).')
    parser.add_argument('--save-dir', default='outputs/per_scene_optim')
    parser.add_argument('--skin-max-dist', type=float, default=0.3,
                        help='ApPs whose surface distance to both bones and leaves exceeds this '
                             'are kept static. Default 0.3 worked on newplant9; inspect the percentile '
                             'distribution if changing scenes.')
    parser.add_argument('--no-leaf-springmass', action='store_true',
                        help='Disable intra-leaf KNN spring-mass. ApPs ride the disk frame rigidly '
                             '(only petiole joint provides leaf motion). Recommended for monocular '
                             'supervision where leaf-internal params have no identifiable signal.')
    parser.add_argument('--learn-struct', action='store_true',
                        help='Add learnable per-node delta_rest_pos. LBS recomputes local offsets '
                             'with delta correction so canonical pose stays identical; the delta '
                             'only changes how the kinematic chain bends.')
    parser.add_argument('--warm-start', type=str, default=None,
                        help='Load physics params (k/c/I, petiole, contact_force) from a final_params.pt '
                             '(typically from a v11 run) before optimization. Useful when learning '
                             'structure deltas on top of an already-fit physics model.')
    parser.add_argument('--w-struct-l2', type=float, default=0.5,
                        help='Weight of ||delta_rest_pos||^2 (per-node L2 prevents drift).')
    parser.add_argument('--w-struct-smooth', type=float, default=2.0,
                        help='Weight of edge-smoothness on delta_rest_pos (||delta[c]-delta[p]||^2).')
    parser.add_argument('--struct-lr-scale', type=float, default=0.2,
                        help='lr scale for delta_rest_pos relative to --lr. Small so structure '
                             'moves slowly relative to physics.')
    parser.add_argument('--densify-from', type=str, default=None,
                        help='Path to a learn_struct final_params.pt. After building the original '
                             'tree, split top-K edges (by child-delta magnitude) at their midpoints, '
                             'rebind ApPs/leaves, then proceed with optimization on the densified tree.')
    parser.add_argument('--densify-top-k', type=int, default=10,
                        help='Number of edges to split when --densify-from is provided.')
    parser.add_argument('--densify-min-len-frac', type=float, default=0.3,
                        help='Skip edges shorter than (median edge length) * this factor.')
    # Shape + topology constraints on the learnable structure (active with --learn-struct).
    parser.add_argument('--w-geom', type=float, default=20.0,
                        help='Weight of the geometric containment loss: keep every (perturbed / '
                             'densified) node within the dense branch point cloud. 0 disables.')
    parser.add_argument('--w-motion', type=float, default=0.5,
                        help='Weight of the motion-direction consistency loss: neighboring nodes '
                             'along the tree should move in consistent directions. 0 disables.')
    parser.add_argument('--geom-margin-scale', type=float, default=2.0,
                        help='r_tol = quantile_0.95(original node->branch nn dist) * this. Nodes are '
                             'free to slide within this band; penalized only beyond it.')
    parser.add_argument('--geom-max-points', type=int, default=8000,
                        help='Subsample the dense branch point cloud to at most this many points.')
    parser.add_argument('--geom-margin', type=float, default=0.02,
                        help='Per-node slack: a node may drift this far beyond its rest distance '
                             'to the branch cloud before being penalized. Handles tube-uncovered '
                             'regions (pot/root) without forcing them inward.')
    args = parser.parse_args()

    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print('Loading scene...')
    scene = load_gaussian_plant_scene(args.source, args.output_dir)
    print(f'  ApP={scene.app.xyz.shape[0]}, branch_nodes={scene.branch.nodes.shape[0]}, leaves={len(scene.leaves)}')

    print('Stage 0: rooting + leaf attachment...')
    tree = root_branch_graph(scene.branch, scene.tube)

    # Optional densification: split top-K high-residual edges from a prior
    # learn_struct run, then rebind everything against the densified tree.
    densify_warm_delta = None
    if args.densify_from is not None:
        from models.structure.tree_densify import pick_split_edges, split_edges
        prior = torch.load(args.densify_from, map_location='cpu', weights_only=False)
        if 'delta_rest_pos' not in prior:
            raise SystemExit('--densify-from params must include delta_rest_pos.')
        median_len = float(tree.edge_length.median())
        split_idx = pick_split_edges(
            tree, prior['delta_rest_pos'], top_k=args.densify_top_k,
            min_edge_length=median_len * args.densify_min_len_frac,
        )
        print(f'[densify] splitting {len(split_idx)} edges: {split_idx}')
        # Record the endpoints of the edges chosen for splitting (in the ORIGINAL
        # tree, before split_edges renumbers things) for the structure-viz HTML.
        _orig_edges = tree.edges_oriented
        densify_split_endpoints = torch.stack([
            tree.nodes[_orig_edges[split_idx, 0]],
            tree.nodes[_orig_edges[split_idx, 1]],
        ], dim=1) if len(split_idx) else torch.empty(0, 2, 3)  # [K, 2, 3]
        tree, densify_info = split_edges(tree, split_idx)
        print(f'[densify] tree: N={tree.nodes.shape[0]}, max_depth={int(tree.depth.max())}')
        # Pad warm-start delta_rest_pos with zeros for the K new midpoint nodes.
        n_old = prior['delta_rest_pos'].shape[0]
        densify_warm_delta = torch.zeros(tree.nodes.shape[0], 3)
        densify_warm_delta[:n_old] = prior['delta_rest_pos']

    attachments = infer_leaf_attachments(scene.leaves, tree)

    print('Computing LBS skinning...')
    skinning = compute_skinning(scene.app.xyz, tree, attachments, max_dist=args.skin_max_dist)

    # Resolve anchor + contact supervision:
    #   - if a contact bundle is supplied, use its most-common per-frame anchor;
    #     stem_tracks + visibility + node ids become per-frame projection supervision.
    #   - otherwise fall back to a heuristic terminal node.
    contact_bundle = None
    stem_tracks_t = stem_vis_t = stem_node_ids = None
    pin_traj = pin_visible = None
    if args.contact_bundle:
        contact_bundle = torch.load(args.contact_bundle, map_location='cpu', weights_only=False)
        anchors_all = contact_bundle['anchor_node_id']
        # Slice to the active window
        end = args.start_frame + args.frames
        anchors = anchors_all[args.start_frame:end]
        valid_anchors = anchors[anchors >= 0]
        if valid_anchors.numel() == 0:
            raise SystemExit('contact bundle has no valid anchor frames in the selected window')
        anchor_node_id = int(torch.bincount(valid_anchors).argmax().item())
        print(f'[opt] anchor node from bundle: {anchor_node_id} '
              f'(present in {(anchors == anchor_node_id).sum().item()}/{anchors.numel()} frames '
              f'within window [{args.start_frame}, {end}))')
        if 'stem_tracks' in contact_bundle:
            stem_tracks_t = contact_bundle['stem_tracks'][args.start_frame:end].to(device)
            stem_vis_t = contact_bundle['stem_visibility'][args.start_frame:end].to(device)
            stem_node_ids = contact_bundle['stem_track_node_id'].to(device)
        if 'pixel_pin' in contact_bundle:
            pin_traj = contact_bundle['pixel_pin'][args.start_frame:end].to(device)
            pin_visible = ~torch.isnan(pin_traj).any(dim=-1)
    else:
        terminals = (tree.subtree_size == 1).nonzero().flatten().tolist()
        anchor_node_id = terminals[len(terminals) // 2] if terminals else int(tree.depth.argmax().item())

    # Densify path always wants delta_rest_pos on the new tree, so force learn_struct
    learn_struct_effective = args.learn_struct or (args.densify_from is not None)
    model = PerScenePhysicsModel(
        tree, attachments, skinning, scene.app.xyz, args.frames, anchor_node_id,
        leaf_clusters=None if args.no_leaf_springmass else scene.leaves, dt=args.dt,
        learn_struct=learn_struct_effective,
    ).to(device)
    args.learn_struct = learn_struct_effective

    if args.warm_start is not None:
        ws = torch.load(args.warm_start, map_location=device, weights_only=False)
        warm_loaded = []
        with torch.no_grad():
            for key in ('log_k_stem', 'log_k_branch', 'log_c_stem', 'log_c_branch',
                        'log_inertia', 'log_k_petiole', 'log_c_petiole',
                        'log_k_leaf', 'log_c_leaf', 'log_m_leaf'):
                if key in ws and hasattr(model, key):
                    getattr(model, key).data.copy_(ws[key].to(device))
                    warm_loaded.append(key)
            if 'contact_force' in ws:
                f_w = ws['contact_force'].to(device)
                T_load = min(f_w.shape[0], model.contact.force.shape[0])
                model.contact.force.data[:T_load].copy_(f_w[:T_load])
                warm_loaded.append(f'contact_force[:{T_load}]')
            # Delta warm-start: densify path uses the pre-padded version (size matches new tree).
            if args.learn_struct:
                if densify_warm_delta is not None:
                    model.delta_rest_pos.data.copy_(densify_warm_delta.to(device))
                    warm_loaded.append(f'delta_rest_pos (padded {densify_warm_delta.shape[0]})')
                elif 'delta_rest_pos' in ws:
                    if ws['delta_rest_pos'].shape == model.delta_rest_pos.shape:
                        model.delta_rest_pos.data.copy_(ws['delta_rest_pos'].to(device))
                        warm_loaded.append('delta_rest_pos')
                    else:
                        print(f'[warm-start] skip delta_rest_pos: shape mismatch '
                              f'{tuple(ws["delta_rest_pos"].shape)} vs {tuple(model.delta_rest_pos.shape)}')
        print(f'[warm-start] loaded {warm_loaded}')
    scales = scene.app.scales.to(device)
    rotations = scene.app.rots.to(device)
    opacities = scene.app.opacities.to(device)
    colors = scene.app.colors.to(device)

    renderer = _build_renderer(args.H, args.W, device=device, sh_degree=0)
    if args.colmap_image is not None:
        camera = _colmap_camera(args.source, args.colmap_image, args.H, args.W, device)
        print(f'[opt] using COLMAP camera for {args.colmap_image}')
    else:
        target_pt = scene.branch.nodes.mean(0).to(device)
        bbox_extent = (scene.branch.nodes.max(0).values - scene.branch.nodes.min(0).values).max().item()
        cam_radius = float(2.0 * bbox_extent)
        camera = _gaussian_camera(renderer, target_pt, cam_radius, args.azimuth, args.elevation)

    if args.synthetic:
        print('Synthesizing target video with ground-truth physics + force...')
        gt_params = dict(log_k_stem=50.0, log_k_branch=15.0, log_c_stem=1.5, log_c_branch=0.8, log_inertia=0.04)
        gt_force = torch.zeros_like(model.contact.force)
        gt_force[:, 0, 0] = 0.4  # constant push along +x at anchor
        target_video, _ = synthesize_target(
            model, renderer, scales, rotations, opacities, colors, camera, gt_params, gt_force,
        )
        torch.save(target_video.detach().cpu(), save_dir / 'target_synthetic.pt')
        print(f'  saved {save_dir / "target_synthetic.pt"}, video shape {tuple(target_video.shape)}')
    elif args.target is not None:
        target_video = torch.load(args.target, map_location=device)
        print(f'  loaded target {tuple(target_video.shape)} from {args.target}')
    elif args.target_video is not None:
        target_video = _load_video_target(
            args.target_video, args.H, args.W, args.frames, device, start_frame=args.start_frame,
        )
        print(f'  loaded target video {tuple(target_video.shape)} from {args.target_video} '
              f'(start_frame={args.start_frame})')
    else:
        raise SystemExit('Need one of --synthetic, --target <path>, or --target-video <path>')

    loss_fn = VideoLossBundle(w_rgb=1.0, w_mask=0.0, w_flow=0.0, w_temporal=0.0)

    # Dense branch geometry for the shape constraint. The tube surface mesh gives the
    # branch volume the skeleton nodes alone don't; subsample for cheap per-step cdist.
    branch_points = None
    geom_r_tol = None
    geom_tol_vec = None        # per-node, rest-relative tolerance
    if args.learn_struct and args.w_geom > 0:
        from optimization.struct_constraints import (
            subsample_points, estimate_geom_margin, rest_relative_tolerance,
        )
        if scene.tube is not None:
            dense = scene.tube.vertices
        elif scene.raw_branch_surface is not None:
            dense = scene.raw_branch_surface
        else:
            dense = tree.nodes
        gen = torch.Generator().manual_seed(0)
        branch_points = subsample_points(dense, args.geom_max_points, generator=gen).to(device)
        geom_r_tol = estimate_geom_margin(
            tree.nodes.to(device), branch_points, quantile=0.95, scale=args.geom_margin_scale,
        )
        # Per-node band: inside nodes held to the global r_tol; nodes the tube doesn't
        # cover at rest (pot/root) get a band around their own rest distance + slack,
        # so the constraint only stops delta from pushing a node FARTHER out.
        geom_tol_vec = rest_relative_tolerance(
            model.chain.rest_pos.to(device), branch_points, geom_r_tol, margin=args.geom_margin,
        )
        n_struct_out = int((geom_tol_vec > geom_r_tol + 1e-6).sum().item())
        print(f'[geom] dense branch points={branch_points.shape[0]} (from '
              f'{"tube" if scene.tube is not None else "fallback"}), global r_tol={geom_r_tol:.4f}, '
              f'{n_struct_out} nodes tube-uncovered at rest (own band)')

    if args.learn_struct:
        # Separate parameter group: delta_rest_pos moves slowly.
        struct_params = [model.delta_rest_pos]
        other_params = [p for n, p in model.named_parameters() if n != 'delta_rest_pos']
        optimizer = torch.optim.Adam([
            {'params': other_params, 'lr': args.lr},
            {'params': struct_params, 'lr': args.lr * args.struct_lr_scale},
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f'Optimizing for {args.steps} steps with lr={args.lr}, anchor_node={anchor_node_id}, '
          f'stem_proj_w={args.w_stem_proj}, pin_w={args.w_pin}')
    history = []
    delta_snapshots = []  # list of (step, per-node |delta| numpy) for the HTML viz
    for it in range(args.steps):
        optimizer.zero_grad()
        out = model()
        video = render_video(renderer, out['ap_traj'], scales, rotations, opacities, colors, camera)
        total, parts = loss_fn(video, target_video)

        # Contact-force regularizers
        reg = (
            0.001 * model.contact.temporal_smoothness_loss()
            + 0.001 * model.contact.sparsity_loss()
        )
        loss = total + reg

        # Structure regularizers (active only when --learn-struct)
        if args.learn_struct:
            d = model.delta_rest_pos
            edges = model.struct_edges
            l2 = (d * d).sum(-1).mean()
            diff = d[edges[:, 1]] - d[edges[:, 0]]
            smooth = (diff * diff).sum(-1).mean()
            struct_reg = args.w_struct_l2 * l2 + args.w_struct_smooth * smooth
            parts['struct_l2'] = l2
            parts['struct_smooth'] = smooth
            loss = loss + struct_reg

            # Shape constraint: keep effective rest nodes inside the dense branch cloud.
            rest_pos_eff = model.chain.rest_pos + model.delta_rest_pos
            if branch_points is not None:
                from optimization.struct_constraints import geometric_containment_loss
                geom_loss, geom_info = geometric_containment_loss(
                    rest_pos_eff, branch_points, geom_tol_vec,
                )
                parts['geom'] = geom_loss
                parts['geom_n_out'] = torch.tensor(float(geom_info['n_outside']))
                loss = loss + args.w_geom * geom_loss

            # Topology constraint: neighboring nodes should move in consistent directions.
            if args.w_motion > 0:
                from optimization.struct_constraints import motion_direction_consistency_loss
                motion_loss = motion_direction_consistency_loss(
                    out['node_pos_traj'], rest_pos_eff, model.struct_edges,
                )
                parts['motion'] = motion_loss
                loss = loss + args.w_motion * motion_loss

        # Stem-track 2D projection loss (PhysTwin-style point supervision)
        if args.w_stem_proj > 0 and stem_tracks_t is not None:
            node_pos_t = out['node_pos_traj']                # [T, N, 3]
            T_eff = min(node_pos_t.shape[0], stem_tracks_t.shape[0])
            stem_loss_sum = 0.0
            n_used = 0
            for t in range(T_eff):
                proj_t = project_world_to_pixels(
                    node_pos_t[t, stem_node_ids], camera, args.H, args.W,
                )                                                            # [K_s, 2]
                visible = stem_vis_t[t] & (stem_node_ids >= 0)
                if visible.any():
                    diff = (proj_t[visible] - stem_tracks_t[t, visible]).pow(2).mean()
                    stem_loss_sum = stem_loss_sum + diff
                    n_used += 1
            if n_used > 0:
                stem_proj = stem_loss_sum / n_used
                parts['stem_proj'] = stem_proj
                loss = loss + args.w_stem_proj * stem_proj

        # Contact-pin loss: anchor's projected position should follow per-frame pin
        if args.w_pin > 0 and pin_traj is not None:
            node_pos_t = out['node_pos_traj']
            T_eff = min(node_pos_t.shape[0], pin_traj.shape[0])
            proj_anchor = project_world_to_pixels(
                node_pos_t[:T_eff, anchor_node_id], camera, args.H, args.W,
            )                                                                # [T, 2]
            visible = pin_visible[:T_eff]
            if visible.any():
                pin_loss = (proj_anchor[visible] - pin_traj[:T_eff][visible]).pow(2).mean()
                parts['pin'] = pin_loss
                loss = loss + args.w_pin * pin_loss

        loss.backward()
        optimizer.step()

        row = {
            'step': it,
            'loss': float(loss.item()),
            'rgb': float(parts.get('rgb', torch.tensor(0.0)).item()),
        }
        for k_extra in ('stem_proj', 'pin', 'struct_l2', 'struct_smooth', 'geom', 'motion', 'geom_n_out'):
            if k_extra in parts:
                row[k_extra] = float(parts[k_extra].item())
        if args.learn_struct:
            dnorm = model.delta_rest_pos.detach().norm(dim=-1)
            row['delta_max'] = float(dnorm.max().item())
            row['delta_mean'] = float(dnorm.mean().item())
            row['delta_p95'] = float(dnorm.quantile(0.95).item())
            if it % 5 == 0 or it == args.steps - 1:
                delta_snapshots.append((it, dnorm.cpu().numpy().copy()))
        history.append(row)

        if it % 5 == 0 or it == args.steps - 1:
            params = {
                'k_stem': model.log_k_stem.exp().item(),
                'k_branch': model.log_k_branch.exp().item(),
                'c_stem': model.log_c_stem.exp().item(),
                'c_branch': model.log_c_branch.exp().item(),
                'inertia': model.log_inertia.exp().item(),
                'k_pet': model.log_k_petiole.exp().item(),
                'c_pet': model.log_c_petiole.exp().item(),
                'k_leaf': model.log_k_leaf.exp().item(),
                'c_leaf': model.log_c_leaf.exp().item(),
                'm_leaf': model.log_m_leaf.exp().item(),
                'force_norm': model.contact.force.detach().norm().item(),
            }
            extras = ''
            for k_extra in ('stem_proj', 'pin', 'struct_l2', 'struct_smooth', 'geom', 'motion'):
                if k_extra in parts:
                    extras += f' {k_extra}={parts[k_extra].item():.4e}'
            if 'geom_n_out' in parts:
                extras += f' geom_nout={int(parts["geom_n_out"].item())}'
            if args.learn_struct:
                delta_norm = model.delta_rest_pos.detach().norm(dim=-1)
                extras += f' delta(max={delta_norm.max().item():.4f},mean={delta_norm.mean().item():.4f})'
            print(f'  [{it:03d}] loss={loss.item():.4e} rgb={parts.get("rgb", torch.tensor(0)).item():.4e}'
                  f'{extras} params={ {k: round(v, 3) for k, v in params.items()} }')

    save_dict = {
        'log_k_stem': model.log_k_stem.detach().cpu(),
        'log_k_branch': model.log_k_branch.detach().cpu(),
        'log_c_stem': model.log_c_stem.detach().cpu(),
        'log_c_branch': model.log_c_branch.detach().cpu(),
        'log_inertia': model.log_inertia.detach().cpu(),
        'log_k_petiole': model.log_k_petiole.detach().cpu(),
        'log_c_petiole': model.log_c_petiole.detach().cpu(),
        'log_k_leaf': model.log_k_leaf.detach().cpu(),
        'log_c_leaf': model.log_c_leaf.detach().cpu(),
        'log_m_leaf': model.log_m_leaf.detach().cpu(),
        'contact_force': model.contact.force.detach().cpu(),
        'anchor_node_id': anchor_node_id,
    }
    if args.learn_struct:
        save_dict['delta_rest_pos'] = model.delta_rest_pos.detach().cpu()
    save_dict['history'] = history
    if geom_r_tol is not None:
        save_dict['geom_r_tol'] = geom_r_tol
    if delta_snapshots:
        import numpy as _np
        save_dict['delta_snapshots'] = {
            'steps': [s for s, _ in delta_snapshots],
            'delta_norm': _np.stack([d for _, d in delta_snapshots], axis=0),  # [S, N]
        }
    if args.densify_from is not None:
        # Save the densified tree so the renderer can rebuild the same topology.
        save_dict['densified_tree'] = {
            'nodes': tree.nodes.cpu(),
            'root_idx': tree.root_idx,
            'parent': tree.parent.cpu(),
            'depth': tree.depth.cpu(),
            'edges_oriented': tree.edges_oriented.cpu(),
            'edge_length': tree.edge_length.cpu(),
            'edge_radius': tree.edge_radius.cpu(),
            'edge_type': tree.edge_type.cpu(),
            'subtree_size': tree.subtree_size.cpu(),
        }
        save_dict['densify_split_endpoints'] = densify_split_endpoints.cpu()
    torch.save(save_dict, save_dir / 'final_params.pt')
    print(f'  saved {save_dir / "final_params.pt"}')


if __name__ == '__main__':
    main()
