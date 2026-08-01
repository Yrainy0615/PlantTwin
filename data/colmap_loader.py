"""Minimal COLMAP sparse-reconstruction reader.

We only need the bits required for camera alignment:
- cameras.bin: intrinsics (model, w, h, params)
- images.bin : per-image quaternion + translation + camera_id + filename

Pose convention (COLMAP):
    P_cam = R(q) @ P_world + t
    R(q) rows are (qw, qx, qy, qz) normalized; the camera looks toward +Z in cam frame.
    The camera center in world coords is C = -R(q).T @ t.

Build a render-compatible (view_matrix, proj_matrix, ...) bundle with
`colmap_camera_to_renderer()`. Output is compatible with
`models.renderer.gaussian_renderer.GaussianRenderer.render_frame`.
"""

from __future__ import annotations

import struct
from collections import namedtuple
from pathlib import Path

import torch

CameraModel = namedtuple('CameraModel', ['name', 'num_params'])
CAMERA_MODELS = {
    0: CameraModel('SIMPLE_PINHOLE', 3),
    1: CameraModel('PINHOLE', 4),
    2: CameraModel('SIMPLE_RADIAL', 4),
    3: CameraModel('RADIAL', 5),
    4: CameraModel('OPENCV', 8),
    5: CameraModel('OPENCV_FISHEYE', 8),
}


def read_cameras_bin(path: Path) -> dict:
    cams = {}
    with open(path, 'rb') as f:
        num = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num):
            cid = struct.unpack('<I', f.read(4))[0]
            model_id = struct.unpack('<i', f.read(4))[0]
            w = struct.unpack('<Q', f.read(8))[0]
            h = struct.unpack('<Q', f.read(8))[0]
            m = CAMERA_MODELS[model_id]
            params = struct.unpack(f'<{m.num_params}d', f.read(8 * m.num_params))
            cams[cid] = {'model': m.name, 'w': int(w), 'h': int(h), 'params': params}
    return cams


def read_images_bin(path: Path) -> dict:
    imgs = {}
    with open(path, 'rb') as f:
        num = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num):
            iid = struct.unpack('<I', f.read(4))[0]
            q = struct.unpack('<4d', f.read(32))   # qw qx qy qz
            t = struct.unpack('<3d', f.read(24))
            cid = struct.unpack('<I', f.read(4))[0]
            name = b''
            while True:
                c = f.read(1)
                if c == b'\x00' or c == b'':
                    break
                name += c
            n_pts2d = struct.unpack('<Q', f.read(8))[0]
            # Each point2D: x (double, 8), y (double, 8), point3D_id (uint64, 8) = 24 bytes
            f.read(24 * n_pts2d)
            imgs[iid] = {
                'q': q,
                't': t,
                'cam_id': cid,
                'name': name.decode('utf-8', errors='replace'),
            }
    return imgs


def quat_to_R(q) -> torch.Tensor:
    """COLMAP convention: q = (qw, qx, qy, qz). Returns 3x3 rotation."""
    qw, qx, qy, qz = q
    return torch.tensor([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=torch.float32)


def find_image_by_name(images: dict, name: str) -> dict | None:
    for v in images.values():
        if v['name'] == name:
            return v
    # Try basename
    base = name.split('/')[-1].split('\\')[-1]
    for v in images.values():
        if v['name'].split('/')[-1].split('\\')[-1] == base:
            return v
    return None


def colmap_camera_to_renderer(
    image_record: dict,
    camera_record: dict,
    H_target: int,
    W_target: int,
    znear: float = 0.01,
    zfar: float = 100.0,
) -> dict:
    """Convert a COLMAP (image, camera) pair into a camera dict matching
    `GaussianRenderer.render_frame`. Intrinsics are rescaled from the source
    image resolution to (H_target, W_target).
    """
    R = quat_to_R(image_record['q'])                    # world -> COLMAP cam (X right, Y down, Z fwd)
    t = torch.tensor(image_record['t'], dtype=torch.float32)
    # COLMAP intrinsics: SIMPLE_PINHOLE = (f, cx, cy); PINHOLE = (fx, fy, cx, cy)
    params = camera_record['params']
    src_W = camera_record['w']
    src_H = camera_record['h']
    sx = W_target / src_W
    sy = H_target / src_H
    if camera_record['model'] == 'SIMPLE_PINHOLE':
        f, cx, cy = params
        fx = fy = f
    else:
        fx, fy, cx, cy = params[:4]
    fx *= sx; fy *= sy
    cx *= sx; cy *= sy

    # View matrix (W2C, 4x4) in row-major form, then transposed for the rasterizer
    view = torch.eye(4, dtype=torch.float32)
    view[:3, :3] = R
    view[:3, 3] = t
    view_T = view.T.contiguous()

    # Build a perspective matrix in TRELLIS / 3DGS convention so it composes with
    # `view` correctly with the rasterizer. The renderer's get_camera() uses
    # normalized intrinsics (fx, fy in [0, 1]) and builds perspective from those;
    # here we directly build a projection from the pixel-space intrinsics.
    perspective = torch.zeros(4, 4, dtype=torch.float32)
    perspective[0, 0] = 2.0 * fx / W_target
    perspective[1, 1] = 2.0 * fy / H_target
    perspective[0, 2] = 2.0 * cx / W_target - 1.0
    perspective[1, 2] = -(2.0 * cy / H_target - 1.0)
    perspective[2, 2] = zfar / (zfar - znear)
    perspective[2, 3] = -znear * zfar / (zfar - znear)
    perspective[3, 2] = 1.0

    proj = perspective @ view
    proj_T = proj.T.contiguous()

    # Camera center in world coords
    campos = -R.T @ t

    # tanfov for the rasterizer kernel
    tanfovx = 0.5 * W_target / fx
    tanfovy = 0.5 * H_target / fy

    return {
        'view_matrix': view_T,
        'proj_matrix': proj_T,
        'campos': campos,
        'tanfovx': float(tanfovx),
        'tanfovy': float(tanfovy),
        'H': H_target,
        'W': W_target,
        'fx': float(fx), 'fy': float(fy), 'cx': float(cx), 'cy': float(cy),
        'src_HW': (src_H, src_W),
    }
