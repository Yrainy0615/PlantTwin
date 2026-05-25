"""
Differentiable Gaussian Splatting renderer for PlantTwin.
Wraps diff_gaussian_rasterization to render deformed plant Gaussians.
Supports rendering a trajectory of positions into a video sequence.
"""
import math
import torch
import torch.nn as nn
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def quaternion_multiply(q1, q2):
    """Multiply two quaternions (wxyz format)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def getWorld2View(R, t):
    """Construct W2C matrix from R (3x3) and t (3,). Following 3DGS convention."""
    Rt = torch.zeros(4, 4, device=R.device)
    Rt[:3, :3] = R.T
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt


def getProjectionMatrix(znear, zfar, fovX, fovY, device='cuda'):
    """3DGS-style projection matrix."""
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device=device)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def look_at_to_R_T(eye, target, up=None):
    """Convert look-at camera to R, T for 3DGS (OpenGL convention)."""
    if up is None:
        up = torch.tensor([0., 1., 0.], device=eye.device)
    z = eye - target
    z = z / z.norm()
    x = torch.cross(up, z)
    x = x / (x.norm() + 1e-8)
    y = torch.cross(z, x)

    R = torch.stack([x, y, z], dim=1)  # [3, 3] columns are x, y, z
    T = -R.T @ eye  # translation in camera space
    return R, T


class GaussianRenderer(nn.Module):
    """
    Differentiable renderer for 3D Gaussians.
    Given Gaussian attributes + positions, renders an image from a camera viewpoint.
    """

    def __init__(self, image_height=256, image_width=256, fov=60.0,
                 bg_color=None, sh_degree=0):
        super().__init__()
        self.H = image_height
        self.W = image_width
        self.fov = math.radians(fov)
        self.sh_degree = sh_degree

        if bg_color is None:
            bg_color = [1.0, 1.0, 1.0]
        self.register_buffer('bg', torch.tensor(bg_color, dtype=torch.float32))

        self.znear = 0.01
        self.zfar = 100.0

    def get_camera(self, azimuth=0.0, elevation=0.0, radius=3.0, target=None):
        """
        Generate camera using TRELLIS/utils3d convention.
        Azimuth/elevation in radians, Z-up coordinate system.
        """
        if target is None:
            target = torch.zeros(3, device=self.bg.device)

        import utils3d
        yaw = torch.tensor(float(math.radians(azimuth)), device=self.bg.device)
        pitch = torch.tensor(float(math.radians(elevation)), device=self.bg.device)

        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ], device=self.bg.device) * radius + target

        extr = utils3d.torch.extrinsics_look_at(
            orig, target, torch.tensor([0., 0., 1.], device=self.bg.device)
        )

        fov_rad = torch.tensor(self.fov, device=self.bg.device)
        intr = utils3d.torch.intrinsics_from_fov_xy(fov_rad, fov_rad)

        # Build projection matrix (TRELLIS convention)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        perspective = torch.zeros(4, 4, device=self.bg.device)
        perspective[0, 0] = 2 * fx
        perspective[1, 1] = 2 * fy
        perspective[0, 2] = 2 * cx - 1
        perspective[1, 2] = -2 * cy + 1
        perspective[2, 2] = self.zfar / (self.zfar - self.znear)
        perspective[2, 3] = self.znear * self.zfar / (self.znear - self.zfar)
        perspective[3, 2] = 1.0

        view = extr  # W2C 4x4
        campos = torch.inverse(view)[:3, 3]
        fovx = 2 * torch.atan(0.5 / fx)
        fovy = 2 * torch.atan(0.5 / fy)

        return {
            'view_matrix': view.T.contiguous(),
            'proj_matrix': (perspective @ view).T.contiguous(),
            'campos': campos,
            'tanfovx': math.tan(fovx.item() * 0.5),
            'tanfovy': math.tan(fovy.item() * 0.5),
        }

    def render_frame(self, means3D, scales, rotations, opacities, colors, camera):
        """
        Render one frame.

        Args:
            means3D: [N, 3] Gaussian centers
            scales: [N, 3] Gaussian scales
            rotations: [N, 4] quaternions (wxyz)
            opacities: [N, 1] opacity values
            colors: [N, 3] RGB colors (precomputed)
            camera: dict from get_camera()

        Returns:
            image: [3, H, W] rendered image
        """
        N = means3D.shape[0]
        subpixel_offset = torch.zeros(self.H, self.W, 2, device=means3D.device)

        settings = GaussianRasterizationSettings(
            image_height=self.H,
            image_width=self.W,
            tanfovx=camera['tanfovx'],
            tanfovy=camera['tanfovy'],
            kernel_size=0.0,
            subpixel_offset=subpixel_offset,
            bg=self.bg,
            scale_modifier=1.0,
            viewmatrix=camera['view_matrix'],
            projmatrix=camera['proj_matrix'],
            sh_degree=self.sh_degree,
            campos=camera['campos'],
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=settings)
        means2D = torch.zeros(N, 3, device=means3D.device, requires_grad=True)

        rendered, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=colors,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
        )
        return rendered  # [3, H, W]

    def render_trajectory(self, trajectory, scales, rotations, opacities, colors,
                          camera=None, azimuth=0.0, elevation=15.0, radius=3.0):
        """
        Render a trajectory of Gaussian positions into a video.

        Args:
            trajectory: [T, N, 3] positions over time
            scales: [N, 3] (static)
            rotations: [N, 4] (static)
            opacities: [N, 1] (static)
            colors: [N, 3] (static)
            camera: optional pre-computed camera dict

        Returns:
            video: [T, 3, H, W]
        """
        if camera is None:
            target = trajectory[0].mean(dim=0).detach()
            camera = self.get_camera(azimuth=azimuth, elevation=elevation,
                                     radius=radius, target=target)

        frames = []
        for t in range(trajectory.shape[0]):
            frame = self.render_frame(trajectory[t], scales, rotations, opacities, colors, camera)
            frames.append(frame)

        return torch.stack(frames, dim=0)  # [T, 3, H, W]
