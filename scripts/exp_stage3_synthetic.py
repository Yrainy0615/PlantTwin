"""Controlled synthetic experiment: does dynamic (motion) information refine plant
STRUCTURE that static reconstruction gets wrong?

Thesis under test
-----------------
A static GaussianPlant reconstruction can MISS a branch connection (漏检): a real
branch is present as StPr but not linked into the articulated structure, so it is
treated as static. Appearance alone cannot fix this — the branch looks the same
whether or not it is connected. MOTION can: under wind the missed branch visibly
sways, and that observed motion tells us which existing branch it belongs to.

Controlled setup (known ground truth)
-------------------------------------
- GT structure  = StPr kinematic tree (kNN-MST over branch StPr).
- GT motion     = FK wind sway, depth-scaled, T frames. Deform 163k AppGas by
                  Gaussian-distance LBS -> render the GT video V_gt. The GT StPr
                  world trajectory is the synthetic "observation" motion gives us.
- Stage 2 (static, corrupted): delete D tree edges -> the child subtrees are orphaned
                  and (as in a real missed detection) rendered STATIC.
- Stage 3 (dynamic refine): for each orphan fragment, re-associate it to the existing
                  node whose rigid motion best explains the fragment's observed motion
                  (min Kabsch residual), then drive it by that best rigid fit.

Metrics (Stage 2 vs Stage 3, both vs GT)
- structure: was the missed edge's parent recovered (exactly / motion-equivalently)?
- motion:    per-orphan-node unexplained-motion residual.
- render:    PSNR of the swaying video vs V_gt.

Outputs under outputs/rerun_2026-07/stage3_synth/:
  gt.mp4 stage2.mp4 stage3.mp4 triptych.mp4  metrics.json  figure.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio

from data.gaussian_plant_loader import _load_app_gaussians
from data.colmap_loader import read_cameras_bin, read_images_bin, colmap_camera_to_renderer
from models.renderer.gaussian_renderer import GaussianRenderer
from models.structure.strpr_motion import (load_strpr_branch, gaussian_lbs_binding,
                                           apply_strpr_lbs, apply_strpr_lbs_full,
                                           estimate_node_motion_excl)
from simulation.strpr_chain import build_strpr_tree, fk, fk_from_node_rotations
from simulation.articulated_chain import exp_so3


# --------------------------------------------------------------------------- #
def kabsch_per_frame(P0: torch.Tensor, Pt: torch.Tensor):
    """Best rigid (R,t) aligning rest points P0 [n,3] to each frame Pt [T,n,3].

    Returns R [T,3,3], t [T,3], and predicted Pt_hat [T,n,3].
    """
    c0 = P0.mean(0)
    ct = Pt.mean(1)                                 # [T,3]
    Pc = P0 - c0                                     # [n,3]
    Qc = Pt - ct[:, None, :]                         # [T,n,3]
    H = torch.einsum('ni,tnj->tij', Pc, Qc)
    U, _, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vt.transpose(-1, -2) @ U.transpose(-1, -2)))
    D = torch.eye(3, device=P0.device).expand(H.shape[0], 3, 3).clone()
    D[:, 2, 2] = d
    R = Vt.transpose(-1, -2) @ D @ U.transpose(-1, -2)       # maps Pc -> Qc
    t = ct - torch.einsum('tij,j->ti', R, c0)
    Pt_hat = torch.einsum('tij,nj->tni', R, P0) + t[:, None, :]
    return R, t, Pt_hat


def subtree_nodes(parent: torch.Tensor, root_child: int) -> list[int]:
    S = parent.shape[0]
    children = [[] for _ in range(S)]
    for i in range(S):
        p = int(parent[i].item())
        if p >= 0:
            children[p].append(i)
    out, stack = [root_child], [root_child]
    while stack:
        u = stack.pop()
        for v in children[u]:
            out.append(v); stack.append(v)
    return out


def render_traj(rnd, app, binding, strpr, node_pos_t, node_rot_t, cam, ap_rest):
    """Render the deformed AppGas: LBS positions + rotate Gaussian orientations
    (without the orientation update, anisotropic leaf Gaussians shatter visually)."""
    frames = []
    for t in range(node_pos_t.shape[0]):
        xyz, quat = apply_strpr_lbs_full(binding, strpr, node_pos_t[t], node_rot_t[t],
                                         ap_rest, app.rots)
        img = rnd.render_frame(xyz, app.scales, quat, app.opacities, app.colors,
                               cam, shs=None).clamp(0, 1)
        frames.append((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    return frames


def psnr(a, b):
    a = a.astype(np.float32) / 255; b = b.astype(np.float32) / 255
    mse = ((a - b) ** 2).mean()
    return 20 * math.log10(1.0) - 10 * math.log10(max(mse, 1e-10))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strpr', default='/mnt/data/gaussianplant_data/gsplant_results/strpr_branch.ply')
    # point_cloud_clean.ply is the appearance aligned with strpr_branch.ply (same plant);
    # point_cloud.ply in this mixed results folder is a DIFFERENT plant -> skeleton floats
    # off the appearance -> AppGas bound to far StPr get flung under motion (the dislocation).
    ap.add_argument('--appgas', default='/mnt/data/gaussianplant_data/gsplant_results/point_cloud_clean.ply')
    ap.add_argument('--scene', default='/mnt/data/gaussianplant_data/newplant4')
    ap.add_argument('--out', default='outputs/rerun_2026-07/stage3_synth')
    ap.add_argument('--frames', type=int, default=24)
    ap.add_argument('--amp', type=float, default=0.18, help='GT sway amplitude (rad)')
    ap.add_argument('--n-miss', type=int, default=3, help='number of edges to "miss"')
    ap.add_argument('--min-frag', type=int, default=5, help='min StPr in a missed fragment')
    ap.add_argument('--max-frag', type=int, default=25, help='max StPr in a missed fragment')
    ap.add_argument('--track-noise', type=float, default=0.03,
                    help='AppGas tracking noise (x median edge length) so the fit is genuine')
    ap.add_argument('--height', type=int, default=832)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ---- load scene ----
    strpr = load_strpr_branch(args.strpr).to(dev)
    app = _load_app_gaussians(Path(args.appgas))
    app = type(app)(**{k: getattr(app, k).to(dev) for k in
                       ['xyz', 'scales', 'rots', 'opacities', 'colors', 'shs']})
    ap_rest = app.xyz
    tree = build_strpr_tree(strpr).to(dev)
    S = len(strpr)
    # k=1 -> every AppGas rides exactly one StPr rigidly (v11 rigid-leaf lesson:
    # soft multi-node blending tears leaves apart under large deformation).
    binding = gaussian_lbs_binding(ap_rest, strpr, k=1)

    # ---- camera (first COLMAP image) ----
    src = Path(args.scene)
    cams = read_cameras_bin(src / 'sparse/0/cameras.bin')
    imgs = read_images_bin(src / 'sparse/0/images.bin')
    first = sorted(imgs.values(), key=lambda r: r['name'])[0]
    ci = cams[first['cam_id']]
    H = args.height; W = int(round(H * ci['w'] / ci['h'])) // 2 * 2
    cam = colmap_camera_to_renderer(first, ci, H, W)
    cam = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in cam.items()}
    rnd = GaussianRenderer(image_height=H, image_width=W, sh_degree=0).to(dev)

    # ---- depth (distance from root along tree) for sway scaling ----
    depth = torch.zeros(S, dtype=torch.long)
    for i in tree.order.tolist():
        p = int(tree.parent[i].item())
        depth[i] = 0 if p < 0 else depth[p] + 1
    depth = depth.to(dev)
    dmax = depth.max().clamp_min(1)

    # per-node bending axis: perpendicular to branch axis, in horizontal plane
    axis = strpr.axis                                          # [S,3]
    wind = torch.tensor([1.0, 0.0, 0.3], device=dev); wind = wind / wind.norm()
    bend_axis = torch.cross(axis, wind.expand_as(axis), dim=-1)
    bend_axis = bend_axis / bend_axis.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    # ---- GT motion: theta_gt[t] = amp * (depth/dmax) * sin(2*pi t/T) * bend_axis ----
    T = args.frames
    ts = torch.arange(T, device=dev).float()
    phase = torch.sin(2 * math.pi * ts / T)                    # [T]
    # uniform per-joint angle amp/dmax -> CUMULATIVE bend at a depth-d node is
    # ~amp*d/dmax <= amp (joint angles stack along the chain; scaling per-joint
    # by depth instead would fold the plant: tip bend ~amp*dmax/2).
    # gusty wind: random per-joint phase (turbulence). In-phase sway moves the whole
    # plant in unison, so inter-branch distances barely change and motion carries little
    # structural information; with phase diversity only TRUE bones keep constant length.
    phi = 2 * math.pi * torch.rand(S, device=dev)
    phase_n = torch.sin(2 * math.pi * ts[:, None] / T + phi[None, :])          # [T,S]
    theta_gt = (args.amp / float(dmax)) * phase_n[:, :, None] * bend_axis[None, :, :].expand(T, S, 3)

    pos_gt = torch.zeros(T, S, 3, device=dev)
    rot_gt = torch.zeros(T, S, 3, 3, device=dev)
    for t in range(T):
        pos_gt[t], rot_gt[t] = fk(tree, theta_gt[t])

    print('[GT] rendering...')
    gt_frames = render_traj(rnd, app, binding, strpr, pos_gt, rot_gt, cam, ap_rest)

    # OBSERVED AppGas trajectory (what tracking the GT video yields) + mild tracking noise
    # so the fit is genuine (not a copy of the generative parameters).
    ap_obs = torch.stack([apply_strpr_lbs_full(binding, strpr, pos_gt[t], rot_gt[t],
                                               ap_rest, app.rots)[0] for t in range(T)])
    med_edge = (strpr.xyz[tree.edges[:, 0]] - strpr.xyz[tree.edges[:, 1]]).norm(dim=-1).median()
    ap_obs = ap_obs + args.track_noise * med_edge * torch.randn_like(ap_obs)

    # ---- pick edges to "miss": most-moving edges, but DISJOINT subtrees whose true
    # parent stays in the graph (so exact topological recovery is achievable) ----
    # #AppGas bound to each StPr (influence mass) -> weight the visual/pivot significance
    bound_mass = torch.zeros(S, device=dev)
    bound_mass.scatter_add_(0, binding.idx.clamp_min(0).reshape(-1),
                            binding.weight.reshape(-1))
    cand = []
    for ei, (a, b) in enumerate(tree.edges.tolist()):
        child = b if int(tree.parent[b].item()) == a else a
        sub = subtree_nodes(tree.parent, child)
        if not (args.min_frag <= len(sub) <= args.max_frag):
            continue                                          # a real sub-branch, not half the plant
        motion = pos_gt[:, sub].std(0).norm(dim=-1).mean().item()
        mass = float(bound_mass[torch.tensor(sub, device=dev)].sum().item())
        cand.append((ei, child, sub, motion, motion * mass))
    cand.sort(key=lambda x: -x[4])                            # most-moving * most-mass first
    missed, claimed = [], set()
    for ei, child, sub, motion, score in cand:
        par = int(tree.parent[child].item())
        if par < 0 or par in claimed or any(n in claimed for n in sub):
            continue                                          # keep subtrees + parents disjoint
        missed.append((ei, child, sub, motion))
        claimed.update(sub)
        if len(missed) >= args.n_miss:
            break
    print('[corrupt] missed edges (child, |subtree|, motion):',
          [(c[1], len(c[2]), round(c[3], 3)) for c in missed])

    orphan_nodes = sorted(set(n for _, _, sub, _ in missed for n in sub))
    orphan_mask = torch.zeros(S, dtype=torch.bool, device=dev); orphan_mask[orphan_nodes] = True
    true_parent = {c[1]: int(tree.parent[c[1]].item()) for c in missed}

    def bfs_order(parent):
        S_ = parent.shape[0]
        children = [[] for _ in range(S_)]
        roots = []
        for i in range(S_):
            p = int(parent[i].item())
            (children[p].append(i) if p >= 0 else roots.append(i))
        order, stack = [], list(roots)
        while stack:
            u = stack.pop(0); order.append(u); stack.extend(children[u])
        return torch.tensor(order, dtype=torch.long, device=parent.device)

    # ---- observation step: per-StPr node trajectories from the tracked AppGas ----
    # (Procrustes per StPr cluster). This is only the OBSERVATION; the motion model that
    # explains it is the FK chain below, which enforces connectivity (bone lengths fixed,
    # children anchored to parents) so branches cannot break apart.
    obs_pos, R_obs, obs_mask, node_var = estimate_node_motion_excl(
        strpr, binding, ap_rest, ap_obs, tree.edges)
    print(f'[obs] {int(obs_mask.sum())}/{S} StPr directly observed; rest inherit '
          f'graph-neighbor rotations')

    # ---- STAGE 2: static structure MISSES the 3 branches -> they are outside the
    # kinematic tree (every orphan node has no parent), so the FK fit leaves them at rest.
    parent2 = tree.parent.clone()
    parent2[orphan_mask] = -1
    tree2 = build_strpr_tree(strpr).to(dev)
    tree2.parent[:] = parent2
    tree2.order[:] = bfs_order(parent2)
    print('[stage2] connected FK reconstruction (missed branches outside the tree)...')
    pos2, rot2 = fk_from_node_rotations(tree2, R_obs)
    s2_frames = render_traj(rnd, app, binding, strpr, pos2, rot2, cam, ap_rest)

    # ---- STAGE 3: recover each missed edge from motion, re-add it, RE-FIT the motion ----
    # Parent found by BONE-LENGTH INVARIANCE (true parent = only node whose distance to the
    # fragment root is temporally constant). After re-attaching, we FIT theta again on the
    # repaired topology -> the once-frozen branch can now be explained by the observed motion.
    cand_q = torch.where(~orphan_mask)[0]
    repaired_parent = tree.parent.clone()
    recovered = {}
    n_motion_exact = n_rest_exact = 0
    for ei, child, sub, mot in missed:
        root = child                                          # fragment root = the missed child
        r_traj = obs_pos[:, root]                             # [T,3] OBSERVED trajectory
        # motion cue: temporal variance of bone length to each candidate (observed)
        seg = obs_pos[:, cand_q] - r_traj[:, None, :]         # [T,Nq,3]
        length = seg.norm(dim=-1)                             # [T,Nq]
        len_var = length.var(0)                               # [Nq] minimal for true parent
        # A true bone has ZERO articulation variance, so its measured bone-length variance
        # equals its tracking-noise floor (node_var[q] + node_var[root], which differs per
        # candidate with cluster size). Normalize by that floor: the true parent's ratio
        # ~O(1) while any non-bone pair carries real articulation variance on top.
        # Propagated (unobserved) nodes carry a fabricated trajectory -> excluded (inf).
        floor = node_var[cand_q] + node_var[root]
        ratio = len_var / floor.clamp_min(1e-30)
        # TEST, not ranking: ratio ~ 1 means the bone length varies only by tracking
        # noise (a rigid bone). Candidates passing the test form the motion-consistent
        # set; attach to the spatially nearest of them. (Ranking by ratio would instead
        # favor whichever candidate has the biggest noise floor.)
        rd = (strpr.xyz[cand_q] - strpr.xyz[root]).norm(dim=-1)
        passing = ratio < 4.0
        if bool(passing.any()):
            rd_m = torch.where(passing, rd, torch.full_like(rd, float('inf')))
            q_motion = int(cand_q[rd_m.argmin()].item())
        else:
            q_motion = int(cand_q[ratio.argmin()].item())
        # rest-distance baseline: nearest non-orphan node at rest (static appearance only)
        rest_d = (strpr.xyz[cand_q] - strpr.xyz[root]).norm(dim=-1)
        q_rest = int(cand_q[rest_d.argmin()].item())
        tp = true_parent[child]
        n_motion_exact += (q_motion == tp); n_rest_exact += (q_rest == tp)
        repaired_parent[root] = q_motion
        recovered[child] = {
            'true_parent': tp,
            'motion_recovered': q_motion, 'motion_exact': q_motion == tp,
            'rest_recovered': q_rest, 'rest_exact': q_rest == tp,
            'bonelen_var_at_parent': float(len_var.min().item()),
            'frag_size': len(sub), 'motion': mot,
        }

    # ---- STAGE 3: re-add the recovered edges -> the fragments join the kinematic tree,
    # and the SAME FK fit now explains their observed motion (connected by construction).
    tree3 = build_strpr_tree(strpr).to(dev)
    tree3.parent[:] = repaired_parent
    tree3.order[:] = bfs_order(repaired_parent)
    print('[stage3] connected FK reconstruction (repaired tree)...')
    pos3, rot3 = fk_from_node_rotations(tree3, R_obs)
    s3_frames = render_traj(rnd, app, binding, strpr, pos3, rot3, cam, ap_rest)

    # ---- discriminator strength: bone-length invariance cleanly separates the true parent
    # (rigid bone -> ~zero temporal length variance) from every other node, for each missed
    # fragment. This is the clean-observation signal motion provides for re-attachment.
    all_parent = tree.parent
    sep = []
    for ei, child, sub, mot in missed:
        cq = torch.tensor([q for q in range(S) if q not in set(sub)], device=dev)
        lv = (pos_gt[:, cq] - pos_gt[:, child][:, None, :]).norm(dim=-1).var(0)
        tp = int(all_parent[child].item())
        var_true = float((pos_gt[:, tp] - pos_gt[:, child]).norm(dim=-1).var(0).item())
        var_foreign = float(lv[cq != tp].min().item())       # best competing (wrong) node
        sep.append({'child': child, 'var_true_parent': var_true, 'var_best_foreign': var_foreign})

    # ---- metrics ----
    # unexplained-motion residual on the MISSED branches' AppGas (fit vs clean GT motion)
    ap_gt = torch.stack([apply_strpr_lbs_full(binding, strpr, pos_gt[t], rot_gt[t],
                                              ap_rest, app.rots)[0] for t in range(T)])
    orphan_ap = orphan_mask[binding.idx[:, 0]]                 # [N] AppGas on a missed branch
    def recon(posN, rotN):
        return torch.stack([apply_strpr_lbs_full(binding, strpr, posN[t], rotN[t],
                                                 ap_rest, app.rots)[0] for t in range(T)])
    ap2, ap3 = recon(pos2, rot2), recon(pos3, rot3)
    res2 = (ap2[:, orphan_ap] - ap_gt[:, orphan_ap]).norm(dim=-1).mean().item()
    res3 = (ap3[:, orphan_ap] - ap_gt[:, orphan_ap]).norm(dim=-1).mean().item()
    psnr2 = float(np.mean([psnr(a, b) for a, b in zip(s2_frames, gt_frames)]))
    psnr3 = float(np.mean([psnr(a, b) for a, b in zip(s3_frames, gt_frames)]))
    n_found = len(missed)
    metrics = {
        'n_missed_edges': n_found, 'n_orphan_nodes': int(orphan_mask.sum().item()),
        'orphan_motion_residual': {'stage2': res2, 'stage3': res3,
                                   'reduction_pct': 100 * (1 - res3 / max(res2, 1e-9))},
        'render_psnr_vs_gt': {'stage2': psnr2, 'stage3': psnr3, 'gain_db': psnr3 - psnr2},
        'structure_recovery': {
            'motion_exact_parent': f'{n_motion_exact}/{n_found}',
            'rest_baseline_exact_parent': f'{n_rest_exact}/{n_found}',
            'note': ('on this synthetic proximity-MST the true parent is also the nearest '
                     'node, so rest-distance ties motion on parent-ID; motion\'s decisive '
                     'contribution is DETECTION (a missed branch is invisible to static '
                     'appearance but sways under motion).'),
            'bone_length_separation': sep,
            'detail': recovered},
        'frames': T, 'amp_rad': args.amp, 'resolution': [W, H], 'n_appgas': int(ap_rest.shape[0]),
        'n_strpr': S,
    }
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    # ---- videos ----
    def save(name, frames): imageio.mimsave(out / name, frames, fps=12, quality=8)
    save('gt.mp4', gt_frames); save('stage2.mp4', s2_frames); save('stage3.mp4', s3_frames)
    trip = [np.concatenate([g, a, b], axis=1) for g, a, b in zip(gt_frames, s2_frames, s3_frames)]
    save('triptych.mp4', trip)

    # ---- summary figure ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    # (a) frame triptych at peak sway
    tp_peak = int(np.argmax(np.abs(np.sin(2 * np.pi * np.arange(T) / T))))
    ax[0].imshow(trip[tp_peak]); ax[0].axis('off')
    ax[0].set_title('peak-sway frame:  GT  |  Stage2 (missed=static)  |  Stage3 (recovered)', fontsize=9)
    # (b) residual + PSNR bars
    b2 = ax[1]
    x = np.arange(2)
    b2.bar(x - 0.2, [res2, res3], 0.4, label='orphan motion residual', color='#d1495b')
    b2.set_ylabel('motion residual (world units)', color='#d1495b')
    b2.set_xticks(x); b2.set_xticklabels(['Stage 2\n(static)', 'Stage 3\n(dynamic)'])
    b2b = b2.twinx()
    b2b.plot(x, [psnr2, psnr3], 'o-', color='#2e7d32', label='render PSNR')
    b2b.set_ylabel('render PSNR vs GT (dB)', color='#2e7d32')
    b2.set_title(f'missed-branch recovery\n{100*(1-res3/max(res2,1e-9)):.0f}% residual drop, '
                 f'+{psnr3-psnr2:.0f} dB', fontsize=9)
    # (c) bone-length invariance: the motion cue that identifies the true parent
    b3 = ax[2]
    idx = np.arange(len(sep))
    vt = [max(s['var_true_parent'], 1e-30) for s in sep]
    vf = [s['var_best_foreign'] for s in sep]
    b3.bar(idx - 0.2, vt, 0.4, label='true parent', color='#2e7d32')
    b3.bar(idx + 0.2, vf, 0.4, label='best competing node', color='#8d6e63')
    b3.set_yscale('log'); b3.set_xticks(idx); b3.set_xticklabels([f'frag@{s["child"]}' for s in sep], fontsize=8)
    b3.set_ylabel('bone-length temporal variance (log)')
    b3.set_title('motion cue for re-attachment:\ntrue parent = rigid bone (~0 variance)', fontsize=9)
    b3.legend(fontsize=8); b3.grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(out / 'figure.png', dpi=120); plt.close()

    print('\n==== RESULTS ====')
    print(f'orphan motion residual : stage2 {res2:.4f} -> stage3 {res3:.4f} '
          f'({metrics["orphan_motion_residual"]["reduction_pct"]:.1f}% drop)')
    print(f'render PSNR vs GT      : stage2 {psnr2:.2f} dB -> stage3 {psnr3:.2f} dB '
          f'(+{psnr3 - psnr2:.2f} dB)')
    print(f'parent recovery        : motion {n_motion_exact}/{n_found} exact '
          f'(bone-length invariance) | rest-distance baseline {n_rest_exact}/{n_found}')
    for s in sep:
        print(f'   frag@{s["child"]}: bonelen var  true-parent {s["var_true_parent"]:.2e}  '
              f'vs best-foreign {s["var_best_foreign"]:.2e}  '
              f'(sep x{s["var_best_foreign"]/max(s["var_true_parent"],1e-30):.0e})')
    for ch, v in recovered.items():
        m = 'OK' if v['motion_exact'] else f'->{v["motion_recovered"]}'
        r = 'OK' if v['rest_exact'] else f'->{v["rest_recovered"]}'
        print(f'   frag@{ch} (size {v["frag_size"]}, true parent {v["true_parent"]}): '
              f'motion {m}, rest {r}  | bonelen_var={v["bonelen_var_at_parent"]:.2e}')
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
