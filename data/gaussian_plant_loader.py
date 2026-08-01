"""Load a GaussianPlant reconstruction bundle.

A scene bundles four kinds of artifacts:
- ApP: appearance Gaussians (the 3DGS PLY produced by training)
- branch_mst: a rooted branch graph (vertices + tree edges) — already MST'd
- branch_tube: a tube mesh around the branch skeleton, used to fit per-segment radii
- leaf_clusters: a list of per-leaf point clusters (positions only)

Layout assumed on disk (matches newplant9):

    <source_root>/
        pretrain_3dgs/point_cloud/iteration_<N>/point_cloud.ply   # ApP
        newplant9_leaves/cluster_*.ply                            # leaf clusters
        branch.ply                                                # raw branch surface samples (optional)

    <output_root>/
        branch_mst.ply                                            # cleaned branch graph
        branch_tube.ply                                           # branch tube mesh
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData


@dataclass
class AppGaussians:
    xyz: torch.Tensor          # [N, 3]
    scales: torch.Tensor       # [N, 3] post-exp
    rots: torch.Tensor         # [N, 4] normalized wxyz
    opacities: torch.Tensor    # [N, 1] post-sigmoid
    colors: torch.Tensor       # [N, 3] DC band → RGB
    shs: torch.Tensor          # [N, 3*K] DC + rest


@dataclass
class BranchGraph:
    nodes: torch.Tensor        # [N_b, 3]
    edges: torch.Tensor        # [E, 2] long, undirected


@dataclass
class BranchTube:
    vertices: torch.Tensor     # [V, 3]
    faces: torch.Tensor        # [F, 3] long


@dataclass
class LeafCluster:
    name: str
    points: torch.Tensor       # [N_k, 3]


@dataclass
class GaussianPlantScene:
    name: str
    source_root: Path
    output_root: Path
    app: AppGaussians
    branch: BranchGraph
    tube: BranchTube | None
    leaves: list[LeafCluster] = field(default_factory=list)
    raw_branch_surface: torch.Tensor | None = None  # [N_s, 3] from branch.ply


def _read_ply_xyz(path: Path) -> torch.Tensor:
    ply = PlyData.read(str(path))
    v = ply['vertex']
    xyz = np.stack([np.asarray(v['x']), np.asarray(v['y']), np.asarray(v['z'])], -1)
    return torch.tensor(xyz, dtype=torch.float32)


def _read_ply_xyz_edges(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    ply = PlyData.read(str(path))
    v = ply['vertex']
    xyz = np.stack([np.asarray(v['x']), np.asarray(v['y']), np.asarray(v['z'])], -1)
    edges = ply['edge']
    e = np.stack([np.asarray(edges['vertex1']), np.asarray(edges['vertex2'])], -1)
    return torch.tensor(xyz, dtype=torch.float32), torch.tensor(e, dtype=torch.long)


def _read_ply_mesh(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    ply = PlyData.read(str(path))
    v = ply['vertex']
    xyz = np.stack([np.asarray(v['x']), np.asarray(v['y']), np.asarray(v['z'])], -1)
    face_data = ply['face']
    # 'vertex_indices' is the standard field name for face vertex lists
    face_field = 'vertex_indices' if 'vertex_indices' in face_data.data.dtype.names else face_data.data.dtype.names[0]
    faces = np.stack([np.asarray(fi, dtype=np.int64) for fi in face_data[face_field]])
    return torch.tensor(xyz, dtype=torch.float32), torch.tensor(faces, dtype=torch.long)


def _load_app_gaussians(ply_path: Path) -> AppGaussians:
    ply = PlyData.read(str(ply_path))
    v = ply['vertex']
    xyz = torch.tensor(np.stack([v['x'], v['y'], v['z']], -1), dtype=torch.float32)
    scales = torch.exp(torch.tensor(
        np.stack([v['scale_0'], v['scale_1'], v['scale_2']], -1), dtype=torch.float32
    ))
    rots = torch.tensor(
        np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], -1), dtype=torch.float32
    )
    rots = rots / (rots.norm(dim=1, keepdim=True) + 1e-8)
    opacities = torch.sigmoid(torch.tensor(v['opacity'][:, None], dtype=torch.float32))

    sh_dc = torch.tensor(
        np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], -1), dtype=torch.float32
    )
    SH_C0 = 0.28209479177387814
    colors = (SH_C0 * sh_dc + 0.5).clamp(0.0, 1.0)

    sh_keys = sorted(
        [k for k in v.data.dtype.names if k.startswith('f_rest_')],
        key=lambda k: int(k.split('_')[-1]),
    )
    if sh_keys:
        sh_rest = torch.tensor(np.stack([v[k] for k in sh_keys], -1), dtype=torch.float32)
        shs = torch.cat([sh_dc, sh_rest], dim=-1)
    else:
        shs = sh_dc

    return AppGaussians(xyz=xyz, scales=scales, rots=rots, opacities=opacities, colors=colors, shs=shs)


def _find_app_ply(source_root: Path) -> Path:
    pc_root = source_root / 'pretrain_3dgs' / 'point_cloud'
    if not pc_root.exists():
        raise FileNotFoundError(f'pretrain_3dgs/point_cloud not found under {source_root}')
    # Pick the latest iteration directory by suffix integer.
    iters = sorted(
        [p for p in pc_root.iterdir() if p.is_dir() and p.name.startswith('iteration_')],
        key=lambda p: int(p.name.split('_')[-1]),
    )
    if not iters:
        raise FileNotFoundError(f'No iteration_* directories under {pc_root}')
    return iters[-1] / 'point_cloud.ply'


def _load_leaf_clusters(leaves_dir: Path) -> list[LeafCluster]:
    clusters = []
    for path in sorted(leaves_dir.glob('cluster_*.ply')):
        points = _read_ply_xyz(path)
        clusters.append(LeafCluster(name=path.stem, points=points))
    return clusters


def load_gaussian_plant_scene(
    source_root: str | Path,
    output_root: str | Path,
    name: str | None = None,
    load_tube: bool = True,
    load_raw_branch: bool = False,
) -> GaussianPlantScene:
    """Load a GaussianPlant scene from its source + output directories.

    Args:
        source_root: e.g. /mnt/data/gaussianplant_data/newplant9
        output_root: e.g. <repo>/outputs/gsplant_output/newplant9
        name: optional scene name; defaults to source_root.name
        load_tube: if True, load branch_tube.ply for per-segment radius fitting
        load_raw_branch: if True, also load branch.ply (raw surface samples)
    """
    source_root = Path(source_root)
    output_root = Path(output_root)

    app = _load_app_gaussians(_find_app_ply(source_root))

    mst_path = output_root / 'branch_mst.ply'
    nodes, edges = _read_ply_xyz_edges(mst_path)
    branch = BranchGraph(nodes=nodes, edges=edges)

    tube = None
    if load_tube:
        tube_path = output_root / 'branch_tube.ply'
        if tube_path.exists():
            verts, faces = _read_ply_mesh(tube_path)
            tube = BranchTube(vertices=verts, faces=faces)

    leaves_dir = source_root / f'{source_root.name}_leaves'
    if not leaves_dir.exists():
        # Fall back to a generic 'leaves' directory if present.
        alt = source_root / 'leaves'
        leaves_dir = alt if alt.exists() else leaves_dir
    leaves = _load_leaf_clusters(leaves_dir) if leaves_dir.exists() else []

    raw_branch = None
    if load_raw_branch:
        raw_path = source_root / 'branch.ply'
        if raw_path.exists():
            raw_branch = _read_ply_xyz(raw_path)

    return GaussianPlantScene(
        name=name or source_root.name,
        source_root=source_root,
        output_root=output_root,
        app=app,
        branch=branch,
        tube=tube,
        leaves=leaves,
        raw_branch_surface=raw_branch,
    )


def scene_summary(scene: GaussianPlantScene) -> str:
    lines = [
        f'GaussianPlantScene: {scene.name}',
        f'  source: {scene.source_root}',
        f'  output: {scene.output_root}',
        f'  app gaussians: N={scene.app.xyz.shape[0]}, shs_dim={scene.app.shs.shape[1]}',
        f'  branch graph: nodes={scene.branch.nodes.shape[0]}, edges={scene.branch.edges.shape[0]}',
    ]
    if scene.tube is not None:
        lines.append(f'  branch tube: V={scene.tube.vertices.shape[0]}, F={scene.tube.faces.shape[0]}')
    lines.append(f'  leaf clusters: {len(scene.leaves)}')
    if scene.leaves:
        sizes = [c.points.shape[0] for c in scene.leaves]
        lines.append(f'    per-leaf points min/median/max: {min(sizes)}/{int(np.median(sizes))}/{max(sizes)}')
    if scene.raw_branch_surface is not None:
        lines.append(f'  raw branch surface: N={scene.raw_branch_surface.shape[0]}')
    return '\n'.join(lines)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='/mnt/data/gaussianplant_data/newplant9')
    parser.add_argument('--output', default='outputs/gsplant_output/newplant9')
    parser.add_argument('--raw-branch', action='store_true')
    args = parser.parse_args()

    scene = load_gaussian_plant_scene(args.source, args.output, load_raw_branch=args.raw_branch)
    print(scene_summary(scene))
