"""Synthetic ground-truth PoC: does monocular motion improve structure recovery?

Controlled experiment with a known answer (the true skeleton), so we can *measure*
whether adding a faithful monocular motion video to the optimization recovers the
structure better than static-only.

Pipeline:
  1. Treat the GaussianPlant reconstruction's branch tree as the TRUE skeleton P*.
  2. Forward-simulate a known wind motion on P* (articulated chain) -> pos_t*.
  3. Render a FAITHFUL video from the monocular (video) camera via edge-binding -> the
     observed motion video (perfectly aligned, by construction).
  4. Degrade the skeleton: perturb branch nodes with depth-biased 3D noise -> P_init.
  5. Recover node positions (delta) under:
        motion_out : single static view RGB only,
        motion_in  : single static view RGB + the GT motion video.
  6. Report node RMSE vs P*, split into image-plane vs along-camera-ray (depth)
     components, plus chamfer to the dense tube. Verdict: does motion_in win?

structure question, or jointly optimized.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data.colmap_loader import (
    read_cameras_bin, read_images_bin, find_image_by_name, colmap_camera_to_renderer,
)
from data.gaussian_plant_loader import load_gaussian_plant_scene
from models.structure.graph_cleanup import root_branch_graph
from models.structure.edge_binding import build_edge_binding, reconstruct, reconstruct_traj
from simulation.articulated_chain import ArticulatedChain


def kinematic_theta(N, depth, n_frames, amp, device, seed=0):
    """A prescribed, bounded joint-angle sway trajectory [T,N,3].

    Each joint sways sinusoidally about a random axis, with amplitude scaled by depth
    (distal joints sway more, root ~0) and a random phase. This replaces a forward
    dynamics rollout: it is stable by construction (bounded angles -> bounded motion)
    and depends on the skeleton only through FK, which is exactly what we need to test
    whether observing the motion helps recover the structure. The optimizer is given
    the same theta_t (known motion), so the only unknown is the structure P.
    """
    g = torch.Generator().manual_seed(seed)
    axis = torch.randn(N, 3, generator=g).to(device)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    phase = torch.rand(N, generator=g).to(device) * 2 * np.pi
    dfac = (depth.float() / depth.float().clamp(min=1).max()).clamp(min=0.0)
    t = torch.linspace(0, 2 * np.pi, n_frames, device=device)
    th = torch.stack([amp * dfac.unsqueeze(-1) * axis * torch.sin(t[i] + phase).unsqueeze(-1)
                      for i in range(n_frames)], dim=0)
    return th


def fk_traj(chain, theta_t, rest_pos):
    """FK every frame: theta_t [T,N,3] -> pos_t [T,N,3], rot_t [T,N,3,3]."""
    poss, rots = [], []
    for i in range(theta_t.shape[0]):
        pos, rot = chain.fk(theta_t[i], rest_pos=rest_pos)
        poss.append(pos); rots.append(rot)
    return torch.stack(poss, 0), torch.stack(rots, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    ap.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    ap.add_argument('--static-views', nargs='+', default=['IMG_1388.JPG'])
    ap.add_argument('--video-camera', default='IMG_1388.JPG')
    ap.add_argument('--frames', type=int, default=14)
    ap.add_argument('--sway-amp', type=float, default=0.05, help='Joint sway amplitude (rad).')
    ap.add_argument('--perturb-std', type=float, default=0.015)
    ap.add_argument('--perturb-depth-scale', type=float, default=3.0,
                    help='Multiply perturbation along the camera ray to make depth the hard part.')
    ap.add_argument('--pure-depth', action='store_true',
                    help='Perturb strictly along the camera ray (isolate the depth claim).')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--grad-clip', type=float, default=0.5)
    ap.add_argument('--branch-thresh', type=float, default=0.3)
    ap.add_argument('--H', type=int, default=512)
    ap.add_argument('--W', type=int, default=340)
    ap.add_argument('--w-video', type=float, default=2.0)
    ap.add_argument('--save-dir', default='outputs/per_scene_optim/synth_poc')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    scene = load_gaussian_plant_scene(args.source, args.output_dir, load_tube=True)
    tree = root_branch_graph(scene.branch, scene.tube)
    Pstar = tree.nodes.to(device)
    edges = tree.edges_oriented.to(device)
    is_stem = (tree.depth <= 1).to(device)
    ap_xyz = scene.app.xyz.to(device)
    scales = scene.app.scales.to(device); rots = scene.app.rots.to(device)
    opac = scene.app.opacities.to(device); colors = scene.app.colors.to(device)
    leaf_centroids = (torch.stack([c.points.mean(0) for c in scene.leaves], 0).to(device)
                      if scene.leaves else None)
    binding = build_edge_binding(ap_xyz, Pstar, edges, branch_thresh=args.branch_thresh,
                                 leaf_centroids=leaf_centroids)
    nb = int(binding['is_branch'].sum())
    print(f'binding: branch-bound {nb}  leaf-bound {ap_xyz.shape[0]-nb}  leaf-instances '
          f'{0 if leaf_centroids is None else leaf_centroids.shape[0]}')

    chain = ArticulatedChain(tree).to(device)

    # cameras
    src = Path(args.source)
    cams = read_cameras_bin(src / 'sparse' / '0' / 'cameras.bin')
    imgs = read_images_bin(src / 'sparse' / '0' / 'images.bin')

    from models.renderer.gaussian_renderer import GaussianRenderer
    renderer = GaussianRenderer(image_height=args.H, image_width=args.W, sh_degree=0).to(device)

    def make_cam(name):
        rec = find_image_by_name(imgs, name)
        cam = colmap_camera_to_renderer(rec, cams[rec['cam_id']], args.H, args.W)
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cam.items()}

    static_cams = {n: make_cam(n) for n in args.static_views}
    vid_cam = make_cam(args.video_camera)

    def render(ap_pos, cam):
        return renderer.render_frame(ap_pos, scales, rots, opac, colors, cam).clamp(0, 1)

    # --- GT motion: prescribed kinematic sway (stable, structure-dependent) ---
    Nn = Pstar.shape[0]
    theta_t = kinematic_theta(Nn, tree.depth.to(device), args.frames, args.sway_amp, device, seed=args.seed)
    with torch.no_grad():
        pos_t_star, rot_t_star = fk_traj(chain, theta_t, Pstar)
        ap_traj_star = reconstruct_traj(pos_t_star, edges, binding, rot_t=rot_t_star)
        nm = (pos_t_star - Pstar.unsqueeze(0)).norm(dim=-1)
        print(f'GT node motion over clip: mean={nm.mean()*100:.1f}cm max={nm.max()*100:.1f}cm')
        video_gt = torch.stack([render(ap_traj_star[t], vid_cam) for t in range(args.frames)], 0)
        static_gt = {n: render(reconstruct(Pstar, edges, binding), static_cams[n]) for n in args.static_views}

    # --- degrade: depth-biased perturbation on branch nodes ---
    # camera forward (world) for the video camera = view-matrix row 2 (z).
    Rwc = vid_cam['view_matrix'].T[:3, :3]          # world->cam rotation (rows)
    cam_fwd = Rwc[2]                                  # world dir of camera +z (depth)
    cam_fwd = cam_fwd / cam_fwd.norm()
    g = torch.Generator(device='cpu').manual_seed(args.seed + 1)
    if args.pure_depth:
        # perturb strictly along the camera ray: the component static (monocular) cannot
        # see at all, so any recovery of it must come from motion.
        scal = torch.randn(Pstar.shape[0], 1, generator=g).to(device) * args.perturb_std
        noise = scal * cam_fwd
    else:
        noise = torch.randn(Pstar.shape[0], 3, generator=g).to(device) * args.perturb_std
        depth_comp = (noise @ cam_fwd).unsqueeze(-1) * cam_fwd
        noise = noise + (args.perturb_depth_scale - 1.0) * depth_comp
    # perturb only branch nodes (depth>0), keep root fixed
    branch_node = is_stem.logical_not()
    perturb = torch.zeros_like(Pstar)
    perturb[branch_node] = noise[branch_node]
    P_init = Pstar + perturb

    def node_errors(P):
        e = (P - Pstar)
        full = e.norm(dim=-1)
        depth = (e * cam_fwd).sum(-1).abs()
        plane = (e - (e * cam_fwd).sum(-1, keepdim=True) * cam_fwd).norm(dim=-1)
        m = branch_node
        return full[m].pow(2).mean().sqrt(), plane[m].pow(2).mean().sqrt(), depth[m].pow(2).mean().sqrt()

    def chamfer_to_tube(P):
        tv = scene.tube.vertices.to(device)
        idx = torch.randperm(tv.shape[0])[:20000]
        return torch.cdist(P[branch_node], tv[idx]).min(1).values.mean()

    init_full, init_plane, init_depth = node_errors(P_init)
    print(f'[init] node RMSE full={init_full*100:.2f}cm plane={init_plane*100:.2f}cm depth={init_depth*100:.2f}cm '
          f'chamfer={chamfer_to_tube(P_init)*100:.2f}cm')

    results = {}
    for mode in ['motion_out', 'motion_in']:
        delta = torch.nn.Parameter(torch.zeros_like(Pstar))   # delta=0 => P=P_init (base)
        base = P_init.detach()
        opt = torch.optim.Adam([delta], lr=args.lr)
        hist = []
        for step in range(args.steps):
            opt.zero_grad()
            P = base + delta
            loss = torch.zeros((), device=device)
            ap_rest = reconstruct(P, edges, binding)
            l_static = torch.zeros((), device=device)
            for n in args.static_views:
                r = render(ap_rest, static_cams[n])
                l_static = l_static + (r - static_gt[n]).abs().mean()
            l_static = l_static / len(args.static_views)
            loss = loss + l_static
            l_vid = torch.zeros((), device=device)
            if mode == 'motion_in':
                pos_t, rot_t = fk_traj(chain, theta_t, P)
                ap_traj = reconstruct_traj(pos_t, edges, binding, rot_t=rot_t)
                for t in range(args.frames):
                    r = render(ap_traj[t], vid_cam)
                    l_vid = l_vid + (r - video_gt[t]).abs().mean()
                l_vid = l_vid / args.frames
                loss = loss + args.w_video * l_vid
            # mild smoothness so unobserved nodes follow neighbors
            l_sm = ((delta[edges[:, 1]] - delta[edges[:, 0]]) ** 2).mean()
            loss = loss + 0.5 * l_sm
            loss.backward()
            torch.nn.utils.clip_grad_norm_([delta], args.grad_clip)
            opt.step()
            if step % 25 == 0 or step == args.steps - 1:
                with torch.no_grad():
                    f, pl, dp = node_errors(base + delta)
                print(f'[{mode} {step:03d}] static={float(l_static):.4f} video={float(l_vid):.4f} '
                      f'| RMSE full={f*100:.2f} plane={pl*100:.2f} depth={dp*100:.2f}cm')
                hist.append({'step': step, 'static': float(l_static), 'video': float(l_vid),
                             'rmse_full': float(f), 'rmse_plane': float(pl), 'rmse_depth': float(dp)})
        with torch.no_grad():
            P = base + delta
            f, pl, dp = node_errors(P)
            cham = chamfer_to_tube(P)
        results[mode] = {'rmse_full': float(f), 'rmse_plane': float(pl), 'rmse_depth': float(dp),
                         'chamfer': float(cham), 'P': P.detach().cpu(), 'hist': hist}
        print(f'==> {mode}: RMSE full={f*100:.2f}cm plane={pl*100:.2f}cm depth={dp*100:.2f}cm chamfer={cham*100:.2f}cm\n')

    # verdict
    mo, mi = results['motion_out'], results['motion_in']
    print('================ VERDICT ================')
    print(f'init      : full={init_full*100:.2f}cm depth={init_depth*100:.2f}cm')
    print(f'motion_out: full={mo["rmse_full"]*100:.2f}cm depth={mo["rmse_depth"]*100:.2f}cm chamfer={mo["chamfer"]*100:.2f}cm')
    print(f'motion_in : full={mi["rmse_full"]*100:.2f}cm depth={mi["rmse_depth"]*100:.2f}cm chamfer={mi["chamfer"]*100:.2f}cm')
    dfull = (mo['rmse_full'] - mi['rmse_full']) / mo['rmse_full'] * 100
    ddep = (mo['rmse_depth'] - mi['rmse_depth']) / mo['rmse_depth'] * 100
    print(f'motion_in improves full RMSE by {dfull:.1f}%, depth RMSE by {ddep:.1f}%')
    print('=========================================')

    torch.save({'results': {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
                'init': {'full': float(init_full), 'plane': float(init_plane), 'depth': float(init_depth)},
                'Pstar': Pstar.cpu(), 'P_init': P_init.cpu(), 'edges': edges.cpu(),
                'args': vars(args)}, save_dir / 'synth_poc.pt')
    print(f'wrote {save_dir/"synth_poc.pt"}')


if __name__ == '__main__':
    main()
