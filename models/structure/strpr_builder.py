"""Rebuild GaussianPlant Structural Primitives (StPr) from a pretrain 3DGS checkpoint.

GaussianPlant's `build_strpr_from_gs` clusters the dense pretrain Gaussians (k-means
on xyz) and fits one anisotropic Gaussian per cluster via PCA; each StPr is then read
as a *cylinder* (branch) or *disk* (leaf) depending on anisotropy. The branch graph is
later built from the branch StPr cylinder endpoints.

This module reproduces that StPr construction in a self-contained way (sklearn
MiniBatchKMeans as a drop-in for faiss; vectorized per-cluster PCA via scatter), so the
StPr become a *learnable* entity we can drive with motion — rather than loading the
frozen, post-export `branch_mst.ply`.

Per-StPr quantities (all torch tensors):
  center    [S,3]  cluster mean
  scale     [S,3]  sqrt eigenvalues (descending): scale[:,0] major .. scale[:,2] minor
  axis      [S,3]  major PCA axis  (cylinder axis)
  normal    [S,3]  minor PCA axis  (disk normal)
  anisotropy[S]    scale0 / scale1
  label     [S]    soft branch prob in {0.4 leaf, 0.6 branch} init (logit-friendly)
  count     [S]    points per cluster
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class StPrSet:
    center: torch.Tensor      # [S,3]
    scale: torch.Tensor       # [S,3] sqrt eigvals desc
    axis: torch.Tensor        # [S,3] major axis (cylinder)
    normal: torch.Tensor      # [S,3] minor axis (disk normal)
    anisotropy: torch.Tensor  # [S]
    label: torch.Tensor       # [S] branch prob init
    count: torch.Tensor       # [S]

    def to(self, device):
        return StPrSet(**{k: v.to(device) for k, v in self.__dict__.items()})

    @property
    def branch_mask(self) -> torch.Tensor:
        return self.label > 0.5


def load_pretrain_xyz(chkpnt_path: str | Path) -> torch.Tensor:
    """Read just the Gaussian centers from a GaussianPlant pretrain checkpoint.

    Checkpoint is `(model_args, iter)` where model_args[1] is `_xyz` [N,3].
    """
    blob = torch.load(str(chkpnt_path), map_location='cpu', weights_only=False)
    model_args = blob[0] if isinstance(blob, (tuple, list)) else blob
    xyz = model_args[1]
    return xyz.detach().float()


def _scatter_cluster_pca(xyz: torch.Tensor, labels: torch.Tensor, n_clusters: int):
    """Vectorized per-cluster mean + covariance eigdecomposition.

    Returns center [S,3], eigvals_desc [S,3], eigvecs_desc [S,3,3] (columns are axes),
    count [S]. Empty/degenerate clusters get identity eigvecs and tiny eigvals.
    """
    device = xyz.device
    S = n_clusters
    count = torch.zeros(S, device=device).index_add_(0, labels, torch.ones_like(labels, dtype=xyz.dtype))
    sum_x = torch.zeros(S, 3, device=device).index_add_(0, labels, xyz)
    cnt_safe = count.clamp(min=1.0).unsqueeze(-1)
    mean = sum_x / cnt_safe

    # E[x x^T] per cluster via outer-product scatter (flatten 3x3 -> 9)
    outer = (xyz.unsqueeze(-1) * xyz.unsqueeze(-2)).reshape(-1, 9)  # [N,9]
    sum_xx = torch.zeros(S, 9, device=device).index_add_(0, labels, outer).reshape(S, 3, 3)
    cov = sum_xx / cnt_safe.unsqueeze(-1) - mean.unsqueeze(-1) * mean.unsqueeze(-2)
    # regularize for stability / degenerate clusters
    cov = cov + torch.eye(3, device=device).unsqueeze(0) * 1e-9

    eigvals, eigvecs = torch.linalg.eigh(cov)           # ascending
    eigvals = eigvals.flip(-1).clamp(min=1e-12)         # descending
    eigvecs = eigvecs.flip(-1)                          # columns reordered to match
    return mean, eigvals, eigvecs, count


def build_strpr_from_pretrain(
    chkpnt_path: str | Path,
    points_per_cluster: int = 100,
    anisotropy_threshold: float = 3.0,
    min_cluster_size: int = 6,
    seed: int = 0,
    device: str = 'cuda',
    cache_labels: str | Path | None = None,
) -> StPrSet:
    """Cluster pretrain GS and fit one StPr per cluster (GaussianPlant-style)."""
    from sklearn.cluster import MiniBatchKMeans

    xyz = load_pretrain_xyz(chkpnt_path).to(device)
    N = xyz.shape[0]
    n_clusters = max(8, N // points_per_cluster)

    labels_np = None
    if cache_labels is not None and Path(cache_labels).exists():
        labels_np = np.load(cache_labels)
        if labels_np.max() + 1 != n_clusters:
            labels_np = None
    if labels_np is None:
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed,
                             batch_size=8192, n_init=3, max_iter=100)
        labels_np = km.fit_predict(xyz.cpu().numpy().astype(np.float32))
        if cache_labels is not None:
            Path(cache_labels).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_labels, labels_np)

    labels = torch.from_numpy(labels_np.astype(np.int64)).to(device)
    mean, eigvals, eigvecs, count = _scatter_cluster_pca(xyz, labels, n_clusters)

    scale = torch.sqrt(eigvals)                          # [S,3] desc
    axis = eigvecs[:, :, 0]                               # major
    normal = eigvecs[:, :, 2]                             # minor
    anisotropy = scale[:, 0] / scale[:, 1].clamp(min=1e-8)

    keep = count >= min_cluster_size
    label = torch.where(anisotropy > anisotropy_threshold,
                        torch.full_like(anisotropy, 0.6),
                        torch.full_like(anisotropy, 0.4))

    s = StPrSet(center=mean[keep], scale=scale[keep], axis=axis[keep], normal=normal[keep],
                anisotropy=anisotropy[keep], label=label[keep], count=count[keep])
    return s


def cylinder_endpoints(strpr: StPrSet, h_scale: float = 1.5):
    """Branch-StPr cylinder endpoints: center +/- (h_scale * scale_major) * axis. [S,2,3]."""
    h = (strpr.scale[:, 0] * h_scale).unsqueeze(-1)
    top = strpr.center + h * strpr.axis
    bot = strpr.center - h * strpr.axis
    return torch.stack([top, bot], dim=1)


def save_strpr_ply(strpr: StPrSet, path: str | Path):
    """Save StPr centers colored by branch(red)/leaf(green) for quick inspection."""
    from plyfile import PlyData, PlyElement
    c = strpr.center.detach().cpu().numpy().astype(np.float32)
    is_branch = strpr.branch_mask.detach().cpu().numpy()
    col = np.where(is_branch[:, None], np.array([[220, 40, 40]]), np.array([[40, 200, 60]])).astype(np.uint8)
    v = np.empty(c.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    v['x'], v['y'], v['z'] = c[:, 0], c[:, 1], c[:, 2]
    v['red'], v['green'], v['blue'] = col[:, 0], col[:, 1], col[:, 2]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(v, 'vertex')], text=True).write(str(path))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--chkpnt', default='/mnt/data/gaussianplant_data/newplant9/pretrain_3dgs/chkpnt7000.pth')
    ap.add_argument('--points-per-cluster', type=int, default=100)
    ap.add_argument('--out', default='outputs/per_scene_optim/strpr_rebuilt.ply')
    ap.add_argument('--cache', default='outputs/per_scene_optim/strpr_kmeans_labels.npy')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    s = build_strpr_from_pretrain(args.chkpnt, points_per_cluster=args.points_per_cluster,
                                  device=args.device, cache_labels=args.cache)
    nb = int(s.branch_mask.sum())
    print(f'StPr total={s.center.shape[0]}  branch={nb}  leaf={s.center.shape[0]-nb}')
    print(f'anisotropy  min/median/max = {s.anisotropy.min():.2f}/{s.anisotropy.median():.2f}/{s.anisotropy.max():.2f}')
    print(f'count/cluster min/median/max = {int(s.count.min())}/{int(s.count.median())}/{int(s.count.max())}')
    save_strpr_ply(s, args.out)
    print(f'wrote {args.out}')
