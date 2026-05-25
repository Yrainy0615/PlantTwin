"""
Differentiable spring-mass simulator for plant physics.
Adapted from ReconPhys (third_party/ReconPhys/sms_lib/models/spring_mass/Spring_Mass.py).
Simplified and decoupled from the ReconPhys training pipeline for standalone use.
"""
import torch
import torch.nn as nn
from copy import deepcopy


class SpringMassSimulator(nn.Module):
    """
    Differentiable spring-mass system on a KNN graph.
    Simulates plant deformation given physical parameters.
    """

    def __init__(self, xyz, k_neighbors=256, k_binding=16, dt=0.03, n_step=100,
                 gravity=None, damping=True):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.k_binding = k_binding
        self.dt = dt
        self.n_step = n_step
        self.damping_enabled = damping
        self.eps = 1e-14

        if gravity is None:
            gravity = [0.0, -9.8, 0.0]
        self.register_buffer('gravity', torch.tensor(gravity, dtype=torch.float32))

        self._initialize(xyz)

    def _initialize(self, xyz):
        """Build KNN graph from initial positions."""
        self.n_points = xyz.shape[0]
        self.register_buffer('init_xyz', xyz.detach().clone())
        self.register_buffer('init_v', torch.zeros_like(xyz))

        origin_len, knn_index = self._build_knn(xyz, self.k_neighbors)
        self.register_buffer('origin_len', origin_len)
        self.register_buffer('knn_index', knn_index)

    def _build_knn(self, xyz, k):
        """Compute KNN graph using torch.cdist. Returns (distances, indices)."""
        dists = torch.cdist(xyz, xyz)
        dists.fill_diagonal_(float('inf'))
        topk_dist, topk_idx = dists.topk(k, dim=1, largest=False)
        return topk_dist, topk_idx

    def _build_interpolation(self, xyz_all, xyz_sample):
        """Build interpolation weights from sampled points to all Gaussian points."""
        dists = torch.cdist(xyz_all, xyz_sample)
        topk_dist, topk_idx = dists.topk(self.k_binding, dim=1, largest=False)
        coef = 1.0 / (topk_dist ** 0.5 + self.eps)
        coef = coef / coef.sum(dim=-1, keepdim=True)
        return topk_idx, coef

    def compute_force(self, xyz, v, K, damp=None):
        """Compute spring forces on all particles."""
        knn_xyz = xyz[self.knn_index]
        delta_pos = knn_xyz - xyz.unsqueeze(1)
        curr_len = torch.norm(delta_pos, dim=2)
        norm_delta_pos = delta_pos / (curr_len.unsqueeze(2) + self.eps)

        # Strain-based force: F = K * (curr_len/origin_len - 1) * direction
        strain = curr_len / (self.origin_len + self.eps) - 1.0
        strain = strain.clamp(-0.5, 0.5)  # prevent extreme deformation
        force = (strain * K).unsqueeze(2) * norm_delta_pos

        if self.damping_enabled and damp is not None:
            knn_v = v[self.knn_index]
            delta_v = knn_v - v.unsqueeze(1)
            damp_force = (damp * torch.sum(delta_v * norm_delta_pos, dim=-1)).unsqueeze(-1) * norm_delta_pos
            force = force + damp_force

        return force.sum(dim=1)

    def step_single(self, xyz, v, K, m, damp, dt):
        """One integration step (semi-implicit Euler)."""
        force = self.compute_force(xyz, v, K, damp)
        gravity_force = m.unsqueeze(1) * self.gravity.unsqueeze(0)
        force_total = force + gravity_force

        v = v + force_total * dt / m.unsqueeze(1)
        xyz = xyz + v * dt
        return xyz, v

    def forward(self, physics_params, n_frames=10, xyz_all=None):
        """
        Run simulation forward for n_frames.

        Args:
            physics_params: dict with keys:
                - k: [N] or scalar, spring stiffness
                - m: [N] or scalar, mass
                - damp: [N, k_neighbors] or scalar (optional)
                - init_velocity: [1, 3] or [N, 3] (optional)
            n_frames: number of output frames
            xyz_all: [M, 3] all Gaussian positions (for interpolation to full set)

        Returns:
            trajectory: [n_frames, N, 3] or [n_frames, M, 3] if xyz_all given
        """
        k_raw = physics_params['k']
        m_raw = physics_params['m']

        if k_raw.dim() == 0 or (k_raw.dim() == 1 and k_raw.shape[0] == 1):
            K = k_raw.expand(self.n_points, self.k_neighbors)
        elif k_raw.dim() == 1 and k_raw.shape[0] == self.n_points:
            K = k_raw.unsqueeze(1).expand(-1, self.k_neighbors)
        else:
            K = k_raw

        # No origin_len normalization here — strain-based force handles it in compute_force

        if m_raw.dim() == 0 or (m_raw.dim() == 1 and m_raw.shape[0] == 1):
            m = m_raw.expand(self.n_points)
        else:
            m = m_raw

        damp = None
        if self.damping_enabled and 'damp' in physics_params:
            damp_raw = physics_params['damp']
            if damp_raw.dim() <= 1:
                damp = damp_raw.expand(self.n_points, self.k_neighbors)
            else:
                damp = damp_raw

        init_vel = physics_params.get('init_velocity', torch.zeros(1, 3, device=self.init_xyz.device))

        interp_idx, interp_coef = None, None
        if xyz_all is not None:
            interp_idx, interp_coef = self._build_interpolation(xyz_all, self.init_xyz)

        xyz = self.init_xyz.clone()
        v = self.init_v.clone() + init_vel

        dt = self.dt / self.n_step
        trajectory = []

        for frame in range(n_frames):
            for _ in range(self.n_step):
                xyz, v = self.step_single(xyz, v, K, m, damp, dt)

            if xyz_all is not None:
                delta = xyz - self.init_xyz
                delta_interp = delta[interp_idx]
                xyz_frame = xyz_all + (delta_interp * interp_coef.unsqueeze(-1)).sum(dim=1)
                trajectory.append(xyz_frame)
            else:
                trajectory.append(xyz.clone())

        return torch.stack(trajectory, dim=0)
