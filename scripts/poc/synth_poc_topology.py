"""Topology half of the PoC: does observed motion improve branch-GRAPH recovery?

Static geometry alone is nearly blind to *connectivity*: two nodes on different limbs
can be spatially close (a short spurious candidate edge) yet not connected. Motion
disambiguates it — truly connected nodes move coherently, spatially-close-but-separate
nodes do not (the user's "neighboring nodes' motion directions should be consistent").

Controlled test (no rendering, isolates the topology-information question):
  1. True topology E* = the reconstruction's branch tree edges.
  2. Candidate edges = E* ∪ kNN(nodes)  (kNN adds short spurious candidates).
  3. GT motion = smooth, proximal-driven sway (whole sub-branches swing together), so
     connected endpoints are coherent and cross-limb pairs are not. Project node motion
     to the monocular video camera -> 2D image-space tracks (what a tracker would give).
  4. Score each candidate edge by:
        geometry  : -length            (static prior; shorter = more likely an edge)
        motion    : 2D motion coherence (cos of endpoint image-motion directions)
        geo+motion: combination
  5. Evaluate how well each score separates true vs spurious edges (ROC-AUC,
     precision@|E*|). Verdict: does motion add topology information over geometry?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data.colmap_loader import read_cameras_bin, read_images_bin, find_image_by_name, colmap_camera_to_renderer
from data.gaussian_plant_loader import load_gaussian_plant_scene
from models.structure.graph_cleanup import root_branch_graph
from simulation.articulated_chain import ArticulatedChain


def sway(N, depth, n_frames, amp, device, seed=0):
    """Independent per-joint sway (random axis/phase per joint, distal-scaled).

    FK propagates each joint's rotation to its subtree, so connected nodes share motion
    (intra-branch coherent) while different limbs, driven by different joints, move
    independently. This is what makes a spurious cross-limb candidate edge *stretch*
    under motion while a true segment stays rigid. The period is 2*pi with the duplicate
    final frame dropped.
    """
    g = torch.Generator().manual_seed(seed)
    axis = torch.randn(N, 3, generator=g).to(device)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    phase = torch.rand(N, generator=g).to(device) * 2 * np.pi
    dfac = (depth.float() / depth.float().clamp(min=1).max()).clamp(min=0.0)
    t = torch.linspace(0, 2 * np.pi, n_frames + 1, device=device)[:-1]
    return torch.stack([amp * dfac.unsqueeze(-1) * axis * torch.sin(t[i] + phase).unsqueeze(-1)
                        for i in range(n_frames)], 0)


def fk_traj(chain, theta_t, rest_pos):
    poss = []
    for i in range(theta_t.shape[0]):
        pos, _ = chain.fk(theta_t[i], rest_pos=rest_pos)
        poss.append(pos)
    return torch.stack(poss, 0)


def project(P, cam, H, W):
    M = torch.cat([P, torch.ones(P.shape[0], 1, device=P.device)], -1)
    clip = (cam['proj_matrix'].T @ M.T).T
    w = clip[:, 3:4].clamp_min(1e-6)
    ndc = clip[:, :3] / w
    return torch.stack([(ndc[:, 0] * 0.5 + 0.5) * W, (ndc[:, 1] * 0.5 + 0.5) * H], -1)


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
    """AUC that higher score -> more likely true (label=1)."""
    s = score.detach().cpu().numpy(); y = label.detach().cpu().numpy().astype(np.int64)
    order = np.argsort(-s)
    y = y[order]
    P = y.sum(); Ntot = len(y) - P
    if P == 0 or Ntot == 0:
        return float('nan')
    tps = np.cumsum(y); fps = np.cumsum(1 - y)
    tpr = tps / P; fpr = fps / Ntot
    return float(np.trapz(tpr, fpr))


def precision_at(score, label, ksel):
    idx = torch.argsort(-score)[:ksel]
    return float(label[idx].float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    ap.add_argument('--output-dir', default='outputs/gsplant_output/newplant9')
    ap.add_argument('--video-camera', default='IMG_1388.JPG')
    ap.add_argument('--frames', type=int, default=24)
    ap.add_argument('--sway-amp', type=float, default=0.12)
    ap.add_argument('--knn', type=int, default=6)
    ap.add_argument('--H', type=int, default=512)
    ap.add_argument('--W', type=int, default=340)
    ap.add_argument('--geo-weight', type=float, default=1.0)
    ap.add_argument('--save-dir', default='outputs/per_scene_optim/synth_poc')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    scene = load_gaussian_plant_scene(args.source, args.output_dir, load_tube=True)
    tree = root_branch_graph(scene.branch, scene.tube)
    P = tree.nodes.to(device); edges_true = tree.edges_oriented.to(device)
    depth = tree.depth.to(device); N = P.shape[0]
    chain = ArticulatedChain(tree).to(device)

    src = Path(args.source)
    cams = read_cameras_bin(src / 'sparse' / '0' / 'cameras.bin')
    imgs = read_images_bin(src / 'sparse' / '0' / 'images.bin')
    rec = find_image_by_name(imgs, args.video_camera)
    cam = colmap_camera_to_renderer(rec, cams[rec['cam_id']], args.H, args.W)
    cam = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cam.items()}

    # GT motion: independent per-branch sway so different limbs move differently
    # (distal-scaled, independent axis/phase per joint -> intra-branch coherent,
    #  cross-limb independent). Drop the duplicate last frame (period 2*pi).
    theta_t = sway(N, depth, args.frames, args.sway_amp, device, seed=args.seed)
    pos_t = fk_traj(chain, theta_t, P)                             # [T,N,3]
    uv_t = torch.stack([project(pos_t[t], cam, args.H, args.W) for t in range(args.frames)], 0)  # [T,N,2]

    edges, is_true = candidate_edges(P, edges_true, args.knn)
    a, b = edges[:, 0], edges[:, 1]
    kE = int(is_true.sum())

    # --- geometry score (static prior): shorter candidate = more likely an edge ---
    length = (P[a] - P[b]).norm(dim=-1)
    geo_score = -length / length.mean()

    # --- motion RIGIDITY score: a true branch segment keeps ~constant length under
    #     bending; a spurious cross-limb edge stretches as the limbs move independently.
    el3d = (pos_t[:, a] - pos_t[:, b]).norm(dim=-1)                # [T,E] 3D length
    strain3d = el3d.std(0) / (length + 1e-6)                       # rel. length variation
    rig3d = -strain3d
    # observable (2D) variant: projected edge-length variation (camera-only)
    el2d = (uv_t[:, a] - uv_t[:, b]).norm(dim=-1)
    strain2d = el2d.std(0) / (el2d.mean(0) + 1e-6)
    rig2d = -strain2d

    nm2d = (uv_t - uv_t.mean(0, keepdim=True)).norm(dim=-1).max(0).values
    print(f'candidate edges: {edges.shape[0]}  (true {kE}, spurious {edges.shape[0]-kE})')
    print(f'GT 2D node motion (px): median {nm2d.median():.1f}  max {nm2d.max():.1f}\n')

    rows = [('geometry only            [motion_out]', geo_score),
            ('motion rigidity 3D       [motion_in]', rig3d),
            ('motion rigidity 2D (observable)', rig2d),
            ('geometry + rigidity3D', geo_score + 3.0 * rig3d)]
    res = {}
    for name, sc in rows:
        auc = roc_auc(sc, is_true); pat = precision_at(sc, is_true, kE)
        res[name] = (auc, pat)
        print(f'{name:40s}  AUC={auc:.3f}  precision@{kE}={pat:.3f}')

    spurious = ~is_true
    short_spurious = spurious & (length < length[is_true].quantile(0.5))
    print(f'\nshort spurious edges (geometry-confusable): {int(short_spurious.sum())}')
    if short_spurious.any():
        print(f'  their 3D strain: median {strain3d[short_spurious].median():.2e}  '
              f'vs true edges median {strain3d[is_true].median():.2e}')

    torch.save({'edges': edges.cpu(), 'is_true': is_true.cpu(), 'length': length.cpu(),
                'strain3d': strain3d.cpu(), 'strain2d': strain2d.cpu(),
                'geo_score': geo_score.cpu(), 'res': res,
                'uv_t': uv_t.cpu(), 'pos_t': pos_t.cpu(), 'P': P.cpu(),
                'edges_true': edges_true.cpu()}, Path(args.save_dir) / 'topo_poc.pt')
    print(f"\nwrote {Path(args.save_dir)/'topo_poc.pt'}")


if __name__ == '__main__':
    main()
