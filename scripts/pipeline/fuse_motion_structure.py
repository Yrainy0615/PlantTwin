"""Joint motion+structure optimizer: both validated signals wired into one loop.

Two PoCs established the signals separately; this integrates them into a single joint
optimization over (a) node positions and (b) soft branch-graph topology:

  - NODE GEOMETRY (depth): a faithful monocular motion video + multi-view static RGB,
    with AppGas attached to the skeleton by `edge_binding` (branch -> edge local frame,
    leaf instances -> rigid swing about their attachment node). Monocular static cannot
    see depth; the swaying video supplies it. Drives `delta` (per-node position).
  - TOPOLOGY (connectivity): 3D motion RIGIDITY. A true branch segment keeps ~constant
    length under bending; a spurious cross-limb candidate edge stretches. A soft weight
    per candidate edge is pulled toward low-strain (true) edges. Drives `edge_logit`.

The two couple through the per-frame node motion `pos_t = FK(theta_t, P+delta)`: better
node depth -> cleaner motion -> sharper rigidity -> better topology, and vice versa.

Controlled (synthetic-GT) protocol so we can *measure* improvement:
  GT structure P*, E* = the reconstruction's branch tree (treated as truth).
  Degrade: depth-perturb branch nodes (delta init) + present candidate edges
  (E* u kNN) with geometry-only logit init (true edges unknown to the optimizer).
  Run --mode motion_in (both signals) vs motion_out (static + geometry only).
  Report node depth RMSE and topology AUC/precision for both.

The diffusion-generated wind clips are not pixel-faithful to this plant, so the motion
observation here is rendered from the reconstruction itself (faithful, aligned).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from data.colmap_loader import read_cameras_bin, read_images_bin, find_image_by_name, colmap_camera_to_renderer
from data.gaussian_plant_loader import load_gaussian_plant_scene
from models.structure.graph_cleanup import root_branch_graph
from models.structure.edge_binding import build_edge_binding, reconstruct, reconstruct_traj
from models.structure.tree_decimate import decimate_tree
from models.structure.tree_densify import split_edges
from models.structure.motion_residual import (edge_fk_residual, edge_nonrigidity,
                                              pick_split_edges_by_motion)
from simulation.articulated_chain import ArticulatedChain


# --------------------------------------------------------------------------- #
# motion + projection helpers
# --------------------------------------------------------------------------- #

def sway(N, depth, n_frames, amp, device, seed=0):
    """Independent per-joint sway (random axis/phase, distal-scaled). FK propagates each
    joint to its subtree -> connected nodes share motion, different limbs are
    independent: true segments stay rigid, cross-limb candidate edges stretch."""
    g = torch.Generator().manual_seed(seed)
    axis = torch.randn(N, 3, generator=g).to(device)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    phase = torch.rand(N, generator=g).to(device) * 2 * np.pi
    dfac = (depth.float() / depth.float().clamp(min=1).max()).clamp(min=0.0)
    t = torch.linspace(0, 2 * np.pi, n_frames + 1, device=device)[:-1]
    return torch.stack([amp * dfac.unsqueeze(-1) * axis * torch.sin(t[i] + phase).unsqueeze(-1)
                        for i in range(n_frames)], 0)


def fk_traj(chain, theta_t, rest_pos):
    poss, rots = [], []
    for i in range(theta_t.shape[0]):
        pos, rot = chain.fk(theta_t[i], rest_pos=rest_pos)
        poss.append(pos); rots.append(rot)
    return torch.stack(poss, 0), torch.stack(rots, 0)


def candidate_edges(nodes, tree_edges, k):
    N = nodes.shape[0]
    d = torch.cdist(nodes, nodes); d.fill_diagonal_(float('inf'))
    knn = d.topk(min(k, N - 1), largest=False).indices
    tset = {tuple(sorted(e)) for e in tree_edges.tolist()}
    pairs = set(tset)
    for i in range(N):
        for j in knn[i].tolist():
            pairs.add(tuple(sorted((i, int(j)))))
    edges = torch.tensor(sorted(pairs), dtype=torch.long, device=nodes.device)
    is_true = torch.tensor([tuple(e.tolist()) in tset for e in edges], device=nodes.device)
    return edges, is_true


def roc_auc(score, label):
    s = score.detach().cpu().numpy(); y = label.detach().cpu().numpy().astype(np.int64)
    order = np.argsort(-s); y = y[order]
    P = y.sum(); Nn = len(y) - P
    if P == 0 or Nn == 0:
        return float('nan')
    tpr = np.cumsum(y) / P; fpr = np.cumsum(1 - y) / Nn
    return float(np.trapezoid(tpr, fpr))


def precision_at(score, label, ksel):
    idx = torch.argsort(-score)[:ksel]
    return float(label[idx].float().mean())


# --------------------------------------------------------------------------- #
# chamfer of StPr cylinder surface vs dense branch points (densify benefit metric)
# --------------------------------------------------------------------------- #

def sample_cylinders(P, edges, er, device, around=12, per_len=300, cap=40):
    a = P[edges[:, 0]]; b = P[edges[:, 1]]
    axis = b - a; L = axis.norm(dim=-1).clamp_min(1e-6); dax = axis / L.unsqueeze(-1)
    up = torch.tensor([0., 0., 1.], device=device).expand_as(dax)
    e1 = torch.cross(dax, up, dim=-1); n = e1.norm(dim=-1, keepdim=True)
    e1 = torch.where(n > 1e-4, e1 / n.clamp_min(1e-6),
                     torch.tensor([1., 0., 0.], device=device).expand_as(dax))
    e2 = torch.cross(dax, e1, dim=-1)
    ang = torch.linspace(0, 2 * np.pi, around + 1, device=device)[:-1]
    nalong = (L * per_len).clamp(2, cap).long()
    pts = []
    for i in range(edges.shape[0]):
        na = int(nalong[i]); ts = torch.linspace(0, 1, na, device=device)
        cen = a[i].unsqueeze(0) + ts.unsqueeze(-1) * axis[i].unsqueeze(0)
        ring = er[i] * (torch.cos(ang).unsqueeze(-1) * e1[i] + torch.sin(ang).unsqueeze(-1) * e2[i])
        pts.append((cen.unsqueeze(1) + ring.unsqueeze(0)).reshape(-1, 3))
    return torch.cat(pts, 0)


def chamfer(X, Y, chunk=2048):
    def nn(A, B):
        out = torch.empty(A.shape[0], device=A.device)
        for i in range(0, A.shape[0], chunk):
            out[i:i + chunk] = torch.cdist(A[i:i + chunk], B).min(1).values
        return out
    return float(0.5 * (nn(X, Y).mean() + nn(Y, X).mean()))


# --------------------------------------------------------------------------- #
# densification experiment: decimate GT -> coarse init, recover joints from motion
# --------------------------------------------------------------------------- #

def run_densify(args, device, save_dir, full_tree, theta_t_full, Pstar, ap_xyz,
                leaf_centroids, static_cams, vid_cam, render, static_gt, video_gt,
                gt_ap_traj, dense_pts):
    """Outer densify loop: optimize delta+topology -> score edges by motion non-rigidity
    -> split -> rebuild -> re-optimize. Benefit measured by non-rigidity / chamfer /
    depth / topo (theta stays prescribed)."""
    Dpts = dense_pts.to(device)
    coarse, dinfo = decimate_tree(full_tree, args.decimate_frac, seed=args.seed)
    kept = dinfo['kept_gt_idx'].to(device)
    Nc0 = coarse.nodes.shape[0]
    N_full = full_tree.nodes.shape[0]
    print(f'decimate: full N={N_full} -> coarse N={Nc0} '
          f'({dinfo["n_collapsed"]}/{dinfo["n_eligible"]} eligible degree-2 collapsed), '
          f'{int(dinfo["edge_has_collapsed"].sum())} coarse edges bear a removed joint')

    # Which collapsed GT joints lie on which edge, keyed by (parent, child) node ids
    # (node ids are stable across split_edges, edge row order is not). Propagated
    # through splits so recovery precision stays measurable in every round.
    Pstar_cpu = Pstar.cpu()
    edge_joints: dict[tuple[int, int], list[int]] = {}
    for j in range(dinfo['collapsed_gt_idx'].numel()):
        pair = tuple(coarse.edges_oriented[int(dinfo['collapsed_onto_edge'][j])].tolist())
        edge_joints.setdefault(pair, []).append(int(dinfo['collapsed_gt_idx'][j]))

    cam_fwd = vid_cam['view_matrix'].T[:3, :3][2]; cam_fwd = cam_fwd / cam_fwd.norm()
    gt_kept = Pstar[kept]                                   # GT positions of coarse nodes
    branch_node0 = (coarse.depth.to(device) > 1)
    g = torch.Generator(device='cpu').manual_seed(args.seed + 1)
    perturb = (torch.randn(Nc0, 1, generator=g).to(device) * args.perturb_std) * cam_fwd
    perturb[~branch_node0] = 0
    P_curr = coarse.nodes.to(device) + perturb
    theta_t = theta_t_full[:, kept].contiguous()           # [T, Nc0, 3] prescribed motion

    def depth_rmse_kept(P):
        e = (P[:Nc0] - gt_kept)[branch_node0]
        return float((e * cam_fwd).sum(-1).abs().pow(2).mean().sqrt())

    tree = coarse
    rounds_hist, recovery, prev_split = [], None, None
    for r in range(args.densify_rounds + 1):
        edges = tree.edges_oriented.to(device)
        nodes_dev = tree.nodes.to(device)
        binding = build_edge_binding(ap_xyz, nodes_dev, edges, branch_thresh=args.branch_thresh,
                                     leaf_centroids=leaf_centroids)
        chain = ArticulatedChain(tree).to(device)
        cand, is_true = candidate_edges(nodes_dev, edges, args.knn)
        ce_a, ce_b = cand[:, 0], cand[:, 1]
        cand_len0 = (nodes_dev[ce_a] - nodes_dev[ce_b]).norm(dim=-1)
        kE = int(is_true.sum())
        Ncur = tree.nodes.shape[0]
        if theta_t.shape[1] < Ncur:
            theta_t = torch.cat([theta_t, torch.zeros(theta_t.shape[0], Ncur - theta_t.shape[1], 3,
                                                      device=device)], 1)

        # Non-rigidity residual depends only on the clean tree + binding + GT motion,
        # so compute it up front: it both scores edges for splitting (below) and
        # gates the video gradient (here). An edge whose motion the current
        # articulation cannot explain would push delta to deform geometry as
        # compensation — exactly the drift seen without gating — so its nodes get
        # their video gradient attenuated by exp(-residual/sigma).
        if args.residual_mode == 'fk':
            with torch.no_grad():
                pos_fk, rot_fk = fk_traj(chain, theta_t[:, :Ncur], nodes_dev)
                pred_traj = reconstruct_traj(pos_fk, edges, binding, rot_t=rot_fk)
            res = edge_fk_residual(tree, binding, gt_ap_traj, pred_traj)
        else:
            res = edge_nonrigidity(tree, binding, gt_ap_traj)
        conf_node = None
        if args.vid_conf_gate:
            # FK is causal: a missing joint displaces its WHOLE descendant subtree
            # in every video frame, so confidence must propagate root->leaf as
            # min(parent conf, incoming-edge conf), not per-incident-edge.
            # Gate on residual EXCESS over the noise floor: local Kabsch never hits
            # zero even on a perfectly articulated tree (~0.05-0.1cm sampling
            # noise), and min-propagation along a 20-deep root path turns any
            # nonzero floor into conf ~0 everywhere — the gate could never open.
            # Most edges are clean, so the median positive residual is the floor.
            rd = res.to(device)
            floor = rd[rd > 0].median() if (rd > 0).any() else rd.new_zeros(())
            conf_e = torch.exp(-(rd - floor).clamp(min=0) / (args.vid_conf_sigma_cm / 100.0))
            conf_in = torch.ones(Ncur, device=device)
            conf_in[edges[:, 1]] = conf_e
            conf_node = torch.ones(Ncur, device=device)
            parent_l = tree.parent.tolist()
            for u in torch.argsort(tree.depth).tolist():
                pu = parent_l[u]
                if pu >= 0:
                    conf_node[u] = torch.minimum(conf_node[pu], conf_in[u])
            nzr = rd[rd > 0]
            print(f'  gate-dbg: floor={floor*100:.3f}cm res>0 n={nzr.numel()}/{rd.numel()} '
                  f'q50={nzr.median()*100:.3f} q90={nzr.quantile(0.9)*100:.3f} '
                  f'max={nzr.max()*100:.3f}cm conf_e<0.5: {(conf_e < 0.5).sum()} edges')
            if r == 0:
                bf = build_edge_binding(ap_xyz, full_tree.nodes.to(device),
                                        full_tree.edges_oriented.to(device),
                                        branch_thresh=args.branch_thresh,
                                        leaf_centroids=leaf_centroids)
                top = rd.topk(5).indices
                for e in top.tolist():
                    sel = ((binding['edge'].to(device) == e) & binding['is_branch'].to(device))
                    selL = sel.nonzero().flatten()
                    nfull = bf['edge'][selL].unique().numel()
                    flip = int((~bf['is_branch'][selL]).sum())
                    print(f'    dbg edge {e}: res={rd[e]*100:.2f}cm n={selL.numel()} '
                          f'spans {nfull} full-edges, {flip} were LEAF in full binding')
            print(f'  video-gate: {(conf_node > 0.5).float().mean():.0%} of nodes open '
                  f'(min {conf_node.min():.3f}, mean {conf_node.mean():.3f})')

        # Per-pixel video weight: gated subtrees render at wrong poses, and their
        # bad pixels bleed gradient into overlapping confident gaussians through
        # alpha blending — node gating alone cannot stop that. Render each frame
        # once with per-gaussian confidence as color (white bg -> empty pixels keep
        # weight 1, preserving silhouette signal) and weight the video residual.
        wmap = None
        if conf_node is not None:
            conf_gauss = conf_node[edges[binding['edge'].to(device), 1]]
            conf_col = conf_gauss.unsqueeze(1).expand(-1, 3).contiguous()
            with torch.no_grad():
                pos_c, rot_c = fk_traj(chain, theta_t, P_curr)
                traj_c = reconstruct_traj(pos_c, edges, binding, rot_t=rot_c)
                wmap = torch.stack([render(traj_c[t], vid_cam, conf_col)
                                    for t in range(args.frames)], 0)

        base = P_curr.detach()
        delta = torch.nn.Parameter(torch.zeros(Ncur, 3, device=device))
        edge_logit = torch.nn.Parameter(torch.zeros(cand.shape[0], device=device))
        opt = torch.optim.Adam([{'params': [delta], 'lr': args.lr},
                                {'params': [edge_logit], 'lr': args.lr_topo}])
        for step in range(args.steps):
            opt.zero_grad()
            P = base + delta
            ap_rest = reconstruct(P, edges, binding)
            l_static = sum((render(ap_rest, static_cams[n]) - static_gt[n]).abs().mean()
                           for n in args.static_views) / len(args.static_views)
            pos_t, rot_t = fk_traj(chain, theta_t, P)
            ap_traj = reconstruct_traj(pos_t, edges, binding, rot_t=rot_t)
            if wmap is None:
                l_vid = sum((render(ap_traj[t], vid_cam) - video_gt[t]).abs().mean()
                            for t in range(args.frames)) / args.frames
            else:
                l_vid = sum(((render(ap_traj[t], vid_cam) - video_gt[t]).abs()
                             * wmap[t]).mean() for t in range(args.frames)) / args.frames
            w = torch.sigmoid(edge_logit)
            eff_len = (P[ce_a] - P[ce_b]).norm(dim=-1).detach()
            l_short = (w * (eff_len / (cand_len0.mean() + 1e-6))).mean()
            deg = torch.zeros(Ncur, device=device).index_add_(0, ce_a, w).index_add_(0, ce_b, w)
            l_deg = ((deg - 2.0).clamp(min=0) ** 2).mean() + ((1.0 - deg).clamp(min=0) ** 2).mean()
            el = (pos_t[:, ce_a] - pos_t[:, ce_b]).norm(dim=-1).detach()
            strain = el.std(0) / (eff_len + 1e-6)
            l_rig = (w * (strain / (strain.mean() + 1e-6))).mean()
            l_topo = l_short + 0.5 * l_deg + args.w_rigidity * l_rig
            l_sm = ((delta[edges[:, 1]] - delta[edges[:, 0]]) ** 2).mean()
            if conf_node is not None:
                (args.w_video * l_vid).backward(retain_graph=True)
                if delta.grad is not None:
                    delta.grad.mul_(conf_node.unsqueeze(1))
                (l_static + l_topo + 0.5 * l_sm).backward()
            else:
                (l_static + args.w_video * l_vid + l_topo + 0.5 * l_sm).backward()
            torch.nn.utils.clip_grad_norm_([delta], args.grad_clip)
            opt.step()
        P_curr = (base + delta).detach()

        # --- metrics ---
        nz = res[res > 0]
        nonrig = float(nz.mean() * 100) if nz.numel() else 0.0
        # Residual of last round's split edges: their two halves, identified by the
        # inserted midpoint node. The mean over all edges is diluted by the many
        # already-straight ones; this is the direct before/after densify signal.
        split_after = None
        if prev_split is not None:
            ms = prev_split['new_nodes']
            rows = [i for i, (a2, b2) in enumerate(edges.tolist()) if a2 in ms or b2 in ms]
            half = res[torch.tensor(rows, dtype=torch.long, device=res.device)]
            half = half[half > 0]
            split_after = float(half.mean() * 100) if half.numel() else 0.0
        er = tree.edge_radius.to(device).clamp(max=float(tree.edge_radius.quantile(0.95)))
        Scyl = sample_cylinders(P_curr, edges, er, device)
        cd = chamfer(Scyl, Dpts) * 100
        drmse = depth_rmse_kept(P_curr) * 100                  # cm
        w = torch.sigmoid(edge_logit)
        auc = roc_auc(w, is_true); prec = precision_at(w, is_true, kE)
        rounds_hist.append({'round': r, 'N': Ncur, 'nonrig_cm': nonrig, 'chamfer_cm': cd,
                            'depth_cm': drmse, 'auc': auc, 'prec': prec,
                            'split_before_cm': prev_split['res_before'] if prev_split else None,
                            'split_after_cm': split_after})
        print(f'[round {r}] N={Ncur}  nonrig={nonrig:.3f}cm  chamfer={cd:.3f}cm  '
              f'depthRMSE={drmse:.2f}cm  topoAUC={auc:.3f} P@{kE}={prec:.3f}')
        if split_after is not None:
            print(f'  split-edge residual: {prev_split["res_before"]:.3f}cm before '
                  f'-> {split_after:.3f}cm after splitting')

        if r < args.densify_rounds:
            min_len = float(tree.edge_length.median()) * args.densify_min_len_frac
            if args.densify_oracle:
                # Upper-bound diagnostic: split exactly the edges hiding a joint.
                # Isolates the recovery MECHANISM from the detection signal.
                row_of = {tuple(e): i for i, e in enumerate(tree.edges_oriented.tolist())}
                idx = sorted(row_of[pc] for pc in edge_joints if pc in row_of)[:args.densify_topk]
            else:
                idx = pick_split_edges_by_motion(tree, res.cpu(), args.densify_topk,
                                                 min_edge_length=min_len)
            if not idx:
                print('  no edges to split; stopping early'); break
            idx_t = torch.tensor(idx, dtype=torch.long)
            pairs = [tuple(tree.edges_oriented[i].tolist()) for i in idx]
            hit = sum(1 for pc in pairs if pc in edge_joints)
            left = sum(len(v) for v in edge_joints.values())
            print(f'  recovery@round{r}: split {len(pairs)} edges, {hit} carry a removed '
                  f'joint -> precision {hit / len(pairs):.3f} ({left} joints still hidden)')
            if r == 0:
                recovery = {'split': len(pairs), 'hit': hit, 'precision': hit / len(pairs),
                            'n_decimated_edges': int(dinfo['edge_has_collapsed'].sum())}
            # Split the structural tree in its clean (GT-coordinate) frame so the
            # next round's binding/chain/candidates are not polluted by this
            # round's geometry error; extend the optimized state separately.
            nodes_pre = tree.nodes
            sp, sc = (tree.edges_oriented[idx_t, 0].to(device),
                      tree.edges_oriented[idx_t, 1].to(device))
            prev_split = {'res_before': float(res[idx_t.to(res.device)].mean() * 100)}
            tree, sinfo = split_edges(tree, idx)
            newM = sinfo['new_node_indices'].tolist()
            prev_split['new_nodes'] = set(newM)
            # Hand each split edge's hidden joints to the half they lie on; the new
            # joint optionally inherits the prescribed GT theta of the nearest
            # hidden joint it recovers (theta stays prescribed, never fit) and is
            # placed at that joint's projection onto the edge — articulating at the
            # midpoint of a bend whose joint sits off-center misexplains the motion.
            theta_new = torch.zeros(theta_t.shape[0], len(newM), 3, device=device)
            tvec = torch.full((len(newM),), 0.5)
            for k, ((p, c), M) in enumerate(zip(pairs, newM)):
                L = edge_joints.pop((p, c), [])
                a, b = nodes_pre[p], nodes_pre[c]
                ab = b - a
                denom = float((ab * ab).sum()) + 1e-12
                ts = [float(torch.dot(Pstar_cpu[g2] - a, ab)) / denom for g2 in L]
                tk = 0.5
                if args.densify_theta_inherit and L:
                    kb = min(range(len(L)), key=lambda i2: abs(ts[i2] - 0.5))
                    theta_new[:, k] = theta_t_full[:, L[kb]].to(device)
                    tk = min(max(ts[kb], 0.1), 0.9)
                    if args.densify_oracle:
                        # True upper bound: the chord projection flattens the bend,
                        # so binding/FK around it stay wrong forever; place the
                        # structural node at the actual GT joint.
                        tree.nodes[M] = Pstar_cpu[L[kb]]
                    else:
                        tree.nodes[M] = a + tk * ab
                    L = [g2 for i2, g2 in enumerate(L) if i2 != kb]
                    ts = [t2 for i2, t2 in enumerate(ts) if i2 != kb]
                tvec[k] = tk
                lo = [g2 for g2, t2 in zip(L, ts) if t2 <= tk]
                hi = [g2 for g2, t2 in zip(L, ts) if t2 > tk]
                if lo: edge_joints[(p, M)] = lo
                if hi: edge_joints[(M, c)] = hi
            if args.densify_theta_inherit:
                eo = tree.edges_oriented
                tree = replace(tree, edge_length=(tree.nodes[eo[:, 0]]
                                                  - tree.nodes[eo[:, 1]]).norm(dim=-1))
            theta_t = torch.cat([theta_t, theta_new], 1)
            tdev = tvec.to(device).unsqueeze(1)
            P_curr = torch.cat([P_curr.detach(),
                                P_curr[sp] + tdev * (P_curr[sc] - P_curr[sp])], 0)

    h0, hL = rounds_hist[0], rounds_hist[-1]
    print('\n================ DENSIFY VERDICT ================')
    print(f'              N    nonrig(cm)  chamfer(cm)  depthRMSE(cm)  topoAUC')
    for h in rounds_hist:
        print(f'round {h["round"]}    {h["N"]:4d}   {h["nonrig_cm"]:7.3f}    {h["chamfer_cm"]:7.3f}'
              f'      {h["depth_cm"]:6.2f}      {h["auc"]:.3f}')
    print(f'non-rigidity {h0["nonrig_cm"]:.3f}->{hL["nonrig_cm"]:.3f}cm, '
          f'chamfer {h0["chamfer_cm"]:.3f}->{hL["chamfer_cm"]:.3f}cm, N {h0["N"]}->{hL["N"]} (GT {N_full})')
    print('=================================================')

    out = save_dir / 'densify.pt'
    torch.save({'rounds': rounds_hist, 'recovery': recovery, 'args': vars(args),
                'final_P': P_curr.cpu(), 'final_edges': tree.edges_oriented.cpu(),
                'final_edge_radius': tree.edge_radius.cpu(),
                'coarse_nodes': coarse.nodes.cpu(), 'coarse_edges': coarse.edges_oriented.cpu(),
                'coarse_edge_radius': coarse.edge_radius.cpu(),
                'edge_has_collapsed': dinfo['edge_has_collapsed'].cpu(),
                'collapsed_gt_idx': dinfo['collapsed_gt_idx'].cpu(),
                'kept_gt_idx': dinfo['kept_gt_idx'].cpu(),
                'Pstar': Pstar.cpu(), 'N_full': N_full}, out)
    print(f'wrote {out}')


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    ap.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    ap.add_argument('--static-views', nargs='+', default=['IMG_1388.JPG'])
    ap.add_argument('--video-camera', default='IMG_1388.JPG')
    ap.add_argument('--frames', type=int, default=14)
    ap.add_argument('--sway-amp', type=float, default=0.10)
    ap.add_argument('--perturb-std', type=float, default=0.03)
    ap.add_argument('--pure-depth', action='store_true', default=True)
    ap.add_argument('--knn', type=int, default=6)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--lr-topo', type=float, default=5e-2)
    ap.add_argument('--grad-clip', type=float, default=0.5)
    ap.add_argument('--branch-thresh', type=float, default=0.3)
    ap.add_argument('--H', type=int, default=512)
    ap.add_argument('--W', type=int, default=340)
    ap.add_argument('--w-video', type=float, default=3.0)
    ap.add_argument('--w-rigidity', type=float, default=5.0)
    ap.add_argument('--save-dir', default='outputs/per_scene_optim/fuse')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    # --- densification (motion-driven decimate+recover); decimate-frac=0 -> legacy path ---
    ap.add_argument('--decimate-frac', type=float, default=0.0,
                    help='>0 enables densify experiment: fraction of degree-2 nodes to collapse for the coarse init')
    ap.add_argument('--densify-rounds', type=int, default=3)
    ap.add_argument('--densify-oracle', action='store_true',
                    help='diagnostic upper bound: split exactly the joint-hiding edges')
    ap.add_argument('--residual-mode', choices=['kabsch', 'fk'], default='kabsch',
                    help='fk: |FK-predicted - GT| onset per edge (catches twist joints '
                         'and lever-arm effects local Kabsch misses; needs theta)')
    ap.add_argument('--vid-conf-gate', action='store_true',
                    help='attenuate the video gradient on nodes whose incident edges '
                         'have high motion non-rigidity (unexplainable motion would '
                         'otherwise deform geometry to compensate)')
    ap.add_argument('--vid-conf-sigma-cm', type=float, default=0.3)
    ap.add_argument('--densify-theta-inherit', action='store_true',
                    help='new midpoint joints inherit the prescribed GT theta of the '
                         'hidden joint they recover (theta is still never optimized)')
    ap.add_argument('--densify-topk', type=int, default=15)
    ap.add_argument('--densify-min-len-frac', type=float, default=0.3,
                    help='skip splitting edges shorter than this * median edge length')
    args = ap.parse_args()

    device = torch.device(args.device)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    scene = load_gaussian_plant_scene(args.source, args.output_dir, load_tube=True,
                                      load_raw_branch=args.decimate_frac > 0)
    tree = root_branch_graph(scene.branch, scene.tube)
    Pstar = tree.nodes.to(device); edges_true = tree.edges_oriented.to(device)
    depth = tree.depth.to(device); N = Pstar.shape[0]
    ap_xyz = scene.app.xyz.to(device)
    scales = scene.app.scales.to(device); rots = scene.app.rots.to(device)
    opac = scene.app.opacities.to(device); colors = scene.app.colors.to(device)
    leaf_centroids = (torch.stack([c.points.mean(0) for c in scene.leaves], 0).to(device)
                      if scene.leaves else None)
    binding = build_edge_binding(ap_xyz, Pstar, edges_true, branch_thresh=args.branch_thresh,
                                 leaf_centroids=leaf_centroids)
    chain = ArticulatedChain(tree).to(device)

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

    def render(ap_pos, cam, col=None):
        return renderer.render_frame(ap_pos, scales, rots, opac,
                                     colors if col is None else col, cam).clamp(0, 1)

    # --- GT faithful motion + static observations ---
    theta_t = sway(N, depth, args.frames, args.sway_amp, device, seed=args.seed)
    with torch.no_grad():
        pos_t_star, rot_t_star = fk_traj(chain, theta_t, Pstar)
        video_gt = torch.stack([render(reconstruct_traj(pos_t_star, edges_true, binding, rot_t=rot_t_star)[t],
                                        vid_cam) for t in range(args.frames)], 0)
        static_gt = {n: render(reconstruct(Pstar, edges_true, binding), static_cams[n]) for n in args.static_views}

    # --- densification experiment (motion-driven decimate+recover) ---
    if args.decimate_frac > 0:
        with torch.no_grad():
            gt_ap_traj = reconstruct_traj(pos_t_star, edges_true, binding, rot_t=rot_t_star)
        run_densify(args, device, save_dir, tree, theta_t, Pstar, ap_xyz, leaf_centroids,
                    static_cams, vid_cam, render, static_gt, video_gt, gt_ap_traj,
                    scene.raw_branch_surface)
        return

    # candidate edges (E* u kNN); is_true used only for evaluation
    cand, is_true = candidate_edges(Pstar, edges_true, args.knn)
    ce_a, ce_b = cand[:, 0], cand[:, 1]
    cand_len0 = (Pstar[ce_a] - Pstar[ce_b]).norm(dim=-1)
    kE = int(is_true.sum())

    # --- degrade structure: depth-biased node perturbation ---
    cam_fwd = vid_cam['view_matrix'].T[:3, :3][2]; cam_fwd = cam_fwd / cam_fwd.norm()
    g = torch.Generator(device='cpu').manual_seed(args.seed + 1)
    branch_node = (depth > 1)
    if args.pure_depth:
        perturb = (torch.randn(N, 1, generator=g).to(device) * args.perturb_std) * cam_fwd
    else:
        perturb = torch.randn(N, 3, generator=g).to(device) * args.perturb_std
    perturb[~branch_node] = 0
    P_init = Pstar + perturb

    def node_depth_rmse(P):
        e = (P - Pstar)[branch_node]
        return (e * cam_fwd).sum(-1).abs().pow(2).mean().sqrt()

    print(f'binding branch/leaf {int(binding["is_branch"].sum())}/{N and ap_xyz.shape[0]-int(binding["is_branch"].sum())}'
          f'  candidate edges {cand.shape[0]} (true {kE})')
    print(f'GT motion mean={ (pos_t_star-Pstar).norm(dim=-1).mean()*100:.1f}cm  '
          f'init node depth RMSE={node_depth_rmse(P_init)*100:.2f}cm\n')

    results = {}
    for mode in ['motion_out', 'motion_in']:
        delta = torch.nn.Parameter(torch.zeros_like(Pstar))           # P = P_init + delta
        edge_logit = torch.nn.Parameter(torch.zeros(cand.shape[0], device=device))
        opt = torch.optim.Adam([{'params': [delta], 'lr': args.lr},
                                {'params': [edge_logit], 'lr': args.lr_topo}])
        base = P_init.detach()
        hist = []
        for step in range(args.steps):
            opt.zero_grad()
            P = base + delta
            loss = torch.zeros((), device=device)

            # static multiview (anchors in-plane geometry)
            ap_rest = reconstruct(P, edges_true, binding)
            l_static = sum((render(ap_rest, static_cams[n]) - static_gt[n]).abs().mean()
                           for n in args.static_views) / len(args.static_views)
            loss = loss + l_static

            # motion (faithful video) -> node depth ; and pos_t for rigidity
            l_vid = torch.zeros((), device=device)
            pos_t = None
            if mode == 'motion_in':
                pos_t, rot_t = fk_traj(chain, theta_t, P)
                ap_traj = reconstruct_traj(pos_t, edges_true, binding, rot_t=rot_t)
                l_vid = sum((render(ap_traj[t], vid_cam) - video_gt[t]).abs().mean()
                            for t in range(args.frames)) / args.frames
                loss = loss + args.w_video * l_vid

            # soft topology — geometry is DETACHED so these terms drive only edge_logit,
            # never the node positions (otherwise the optimizer cheats by moving nodes to
            # fake short/rigid edges, corrupting delta).
            w = torch.sigmoid(edge_logit)
            eff_len = (P[ce_a] - P[ce_b]).norm(dim=-1).detach()
            l_short = (w * (eff_len / (cand_len0.mean() + 1e-6))).mean()
            deg = torch.zeros(N, device=device).index_add_(0, ce_a, w).index_add_(0, ce_b, w)
            l_deg = ((deg - 2.0).clamp(min=0) ** 2).mean() + ((1.0 - deg).clamp(min=0) ** 2).mean()
            l_topo = l_short + 0.5 * l_deg
            if mode == 'motion_in':
                # rigidity: true segments keep constant length under motion -> low strain
                el = (pos_t[:, ce_a] - pos_t[:, ce_b]).norm(dim=-1).detach()   # [T,E]
                strain = el.std(0) / (eff_len + 1e-6)
                l_rig = (w * (strain / (strain.mean() + 1e-6))).mean()
                l_topo = l_topo + args.w_rigidity * l_rig
            loss = loss + l_topo

            # struct smoothness on delta (neighbors move together)
            l_sm = ((delta[edges_true[:, 1]] - delta[edges_true[:, 0]]) ** 2).mean()
            loss = loss + 0.5 * l_sm

            loss.backward()
            torch.nn.utils.clip_grad_norm_([delta], args.grad_clip)
            opt.step()

            if step % 50 == 0 or step == args.steps - 1:
                with torch.no_grad():
                    w = torch.sigmoid(edge_logit)
                    auc = roc_auc(w, is_true); pat = precision_at(w, is_true, kE)
                    dep = node_depth_rmse(base + delta)
                print(f'[{mode} {step:03d}] static={float(l_static):.4f} video={float(l_vid):.4f} '
                      f'| depthRMSE={dep*100:.2f}cm topoAUC={auc:.3f} P@{kE}={pat:.3f}')
                hist.append({'step': step, 'depth': float(dep), 'auc': auc, 'prec': pat})

        with torch.no_grad():
            P = base + delta
            w = torch.sigmoid(edge_logit)
            results[mode] = {'depth': float(node_depth_rmse(P)), 'auc': roc_auc(w, is_true),
                             'prec': precision_at(w, is_true, kE), 'hist': hist,
                             'edge_weight': w.cpu(), 'P': P.cpu()}
        r = results[mode]
        print(f'==> {mode}: node depth RMSE={r["depth"]*100:.2f}cm  topo AUC={r["auc"]:.3f}  prec@{kE}={r["prec"]:.3f}\n')

    mo, mi = results['motion_out'], results['motion_in']
    print('================ JOINT VERDICT ================')
    print(f'                node depth RMSE   topo AUC   topo prec@{kE}')
    print(f'motion_out      {mo["depth"]*100:6.2f}cm          {mo["auc"]:.3f}      {mo["prec"]:.3f}')
    print(f'motion_in       {mi["depth"]*100:6.2f}cm          {mi["auc"]:.3f}      {mi["prec"]:.3f}')
    print(f'depth improved {(mo["depth"]-mi["depth"])/mo["depth"]*100:.1f}%, '
          f'topo AUC {mo["auc"]:.3f}->{mi["auc"]:.3f}, prec {mo["prec"]:.3f}->{mi["prec"]:.3f}')
    print('===============================================')

    torch.save({'results': results,  # keep optimized P per mode for downstream rendering
                'is_true': is_true.cpu(), 'cand': cand.cpu(), 'Pstar': Pstar.cpu(),
                'P_init': P_init.cpu(), 'edges_true': edges_true.cpu(),
                'edge_radius': tree.edge_radius.cpu(), 'edge_length': tree.edge_length.cpu(),
                'edge_type': tree.edge_type.cpu(), 'depth': depth.cpu(), 'args': vars(args)},
               save_dir / 'joint.pt')
    print(f'wrote {save_dir/"joint.pt"}')


if __name__ == '__main__':
    main()
