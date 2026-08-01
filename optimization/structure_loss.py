"""
Structure-aware losses for part-aware plant pretrain.

Constrains simulation output using plant structure priors:
  - Trunk anchoring (minimal displacement)
  - Branch deformation limit (soft upper bound)
  - Material consistency (similar params within a part)
  - Material ordering (k_trunk > k_branch > k_leaf)
  - Attachment preservation (boundary Gaussians stay connected)
  - Spatial smoothness (kNN-based param smoothness)
"""
import torch
import torch.nn.functional as F


def trunk_anchor_loss(trajectory, trunk_mask):
    """Penalize trunk displacement from canonical position."""
    if trunk_mask.sum() == 0:
        return torch.tensor(0.0, device=trajectory.device)
    displacement = (trajectory - trajectory[0:1]).norm(dim=-1)  # [T, N]
    return displacement[:, trunk_mask].mean()


def branch_limit_loss(trajectory, branch_mask, threshold=0.05):
    """Soft upper bound on branch deformation amplitude."""
    if branch_mask.sum() == 0:
        return torch.tensor(0.0, device=trajectory.device)
    displacement = (trajectory - trajectory[0:1]).norm(dim=-1)  # [T, N]
    return F.relu(displacement[:, branch_mask] - threshold).mean()


def _per_gaussian_k(k):
    """Aggregate per-spring k [N, K] to per-Gaussian [N] via mean."""
    if k.dim() == 2:
        return k.mean(dim=1)
    return k


def material_consistency_loss(predicted_params, part_labels):
    """Penalize variance of k/damp within each part instance."""
    k = _per_gaussian_k(predicted_params['k'])
    damp = _per_gaussian_k(predicted_params['damp'])
    loss = torch.tensor(0.0, device=k.device)
    labels = part_labels.labels
    for part_id in labels.unique():
        mask = (labels == part_id)
        if mask.sum() < 2:
            continue
        loss = loss + k[mask].var()
        loss = loss + damp[mask].var()
    return loss


def material_ordering_loss(predicted_params, part_labels):
    """Enforce k_trunk > k_branch > k_leaf."""
    k = _per_gaussian_k(predicted_params['k'])
    loss = torch.tensor(0.0, device=k.device)

    trunk_k = k[part_labels.trunk_mask].mean() if part_labels.trunk_mask.sum() > 0 else None
    branch_k = k[part_labels.branch_mask].mean() if part_labels.branch_mask.sum() > 0 else None
    leaf_k = k[part_labels.leaf_mask].mean() if part_labels.leaf_mask.sum() > 0 else None

    if leaf_k is not None and branch_k is not None:
        loss = loss + F.relu(leaf_k - branch_k)
    if branch_k is not None and trunk_k is not None:
        loss = loss + F.relu(branch_k - trunk_k)

    return loss


def attachment_loss(trajectory, part_labels, knn_index, init_boundary_dist=None):
    """
    Preserve connectivity at part boundaries.
    Gaussians near part boundaries should maintain their relative distances.
    """
    labels = part_labels.labels
    device = trajectory.device

    # Find boundary Gaussians: particles whose KNN contains particles from different parts
    neighbor_labels = labels[knn_index]  # [N, K]
    is_boundary = (neighbor_labels != labels.unsqueeze(1)).any(dim=1)  # [N]

    if is_boundary.sum() < 2:
        return torch.tensor(0.0, device=device)

    boundary_idx = torch.where(is_boundary)[0]
    if boundary_idx.shape[0] > 512:
        perm = torch.randperm(boundary_idx.shape[0], device=device)[:512]
        boundary_idx = boundary_idx[perm]

    # Compare distances at each frame vs. initial frame
    init_pos = trajectory[0, boundary_idx]  # [B, 3]
    init_dist = torch.cdist(init_pos, init_pos)  # [B, B]

    loss = torch.tensor(0.0, device=device)
    T = trajectory.shape[0]
    for t in range(1, T):
        pos_t = trajectory[t, boundary_idx]
        dist_t = torch.cdist(pos_t, pos_t)
        loss = loss + (dist_t - init_dist).pow(2).mean()

    return loss / max(T - 1, 1)


def spatial_smoothness_loss(predicted_params, knn_index):
    """KNN-based spatial smoothness on predicted parameters."""
    k = _per_gaussian_k(predicted_params['k'])
    k_neighbors = k[knn_index]  # [N, K]
    return (k_neighbors - k.unsqueeze(1)).pow(2).mean()


class StructureLoss:
    """Combine all structure-aware losses with configurable weights."""

    def __init__(self, lambda_anchor=10.0, lambda_branch=1.0,
                 lambda_consistency=0.1, lambda_ordering=0.5,
                 lambda_attach=1.0, lambda_smooth=0.01,
                 branch_threshold=0.05):
        self.lambda_anchor = lambda_anchor
        self.lambda_branch = lambda_branch
        self.lambda_consistency = lambda_consistency
        self.lambda_ordering = lambda_ordering
        self.lambda_attach = lambda_attach
        self.lambda_smooth = lambda_smooth
        self.branch_threshold = branch_threshold

    def __call__(self, trajectory, predicted_params, part_labels, knn_index):
        """
        Args:
            trajectory: [T, N, 3]
            predicted_params: dict with k, damp, drag, mass
            part_labels: PartLabels
            knn_index: [N, K] KNN indices from simulator

        Returns:
            total_loss: scalar
            loss_dict: dict of individual losses for logging
        """
        l_anchor = trunk_anchor_loss(trajectory, part_labels.trunk_mask)
        l_branch = branch_limit_loss(trajectory, part_labels.branch_mask, self.branch_threshold)
        l_consist = material_consistency_loss(predicted_params, part_labels)
        l_order = material_ordering_loss(predicted_params, part_labels)
        l_attach = attachment_loss(trajectory, part_labels, knn_index)
        l_smooth = spatial_smoothness_loss(predicted_params, knn_index)

        total = (
            self.lambda_anchor * l_anchor
            + self.lambda_branch * l_branch
            + self.lambda_consistency * l_consist
            + self.lambda_ordering * l_order
            + self.lambda_attach * l_attach
            + self.lambda_smooth * l_smooth
        )

        loss_dict = {
            'anchor': l_anchor.item(),
            'branch': l_branch.item(),
            'consistency': l_consist.item(),
            'ordering': l_order.item(),
            'attachment': l_attach.item(),
            'smoothness': l_smooth.item(),
        }

        return total, loss_dict
