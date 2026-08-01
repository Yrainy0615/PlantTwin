"""
Visualize PhysX-Omni VLM voxel output as colored point clouds.
Each part gets a different color. Saves a matplotlib 3D scatter plot.

Usage:
    python scripts/vis_physx_voxel.py --input_dir data/physx_vlm_output/IMG_1388
    python scripts/vis_physx_voxel.py --input_dir data/physx_vlm_output/IMG_6698
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


PART_COLORS = [
    "#e74c3c",  # red
    "#8B4513",  # brown
    "#27ae60",  # green
    "#3498db",  # blue
    "#f39c12",  # orange
    "#9b59b6",  # purple
    "#1abc9c",  # teal
    "#e67e22",  # dark orange
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--elev", type=float, default=20)
    parser.add_argument("--azim", type=float, default=45)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    name = input_dir.name

    basic_info = (input_dir / "basic_info.txt").read_text(encoding="utf-8")
    part_names = []
    for line in basic_info.splitlines():
        line = line.strip()
        if line.startswith("l_"):
            pname = line.split("|")[0].split(":")[1].strip()
            part_names.append(pname)

    part_files = sorted(input_dir.glob("ind_*.npy"))
    if not part_files:
        print(f"No voxel files found in {input_dir}")
        return

    fig = plt.figure(figsize=(14, 6))

    # --- Per-part separate views ---
    n_parts = len(part_files)
    for i, pf in enumerate(part_files):
        coords = np.load(pf)
        pname = part_names[i] if i < len(part_names) else f"part_{i}"
        color = PART_COLORS[i % len(PART_COLORS)]

        ax = fig.add_subplot(1, n_parts + 1, i + 1, projection="3d")
        if len(coords) > 0:
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                       c=color, s=2, alpha=0.6)
        ax.set_title(f"l_{i}: {pname}\n({len(coords)} voxels)", fontsize=9)
        ax.set_xlim(0, 64); ax.set_ylim(0, 64); ax.set_zlim(0, 64)
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_xlabel("X", fontsize=7)
        ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
        ax.tick_params(labelsize=6)

    # --- Combined view ---
    ax_all = fig.add_subplot(1, n_parts + 1, n_parts + 1, projection="3d")
    for i, pf in enumerate(part_files):
        coords = np.load(pf)
        color = PART_COLORS[i % len(PART_COLORS)]
        pname = part_names[i] if i < len(part_names) else f"part_{i}"
        if len(coords) > 0:
            ax_all.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                           c=color, s=2, alpha=0.6, label=f"l_{i}: {pname}")
    ax_all.set_title("All parts", fontsize=9)
    ax_all.set_xlim(0, 64); ax_all.set_ylim(0, 64); ax_all.set_zlim(0, 64)
    ax_all.view_init(elev=args.elev, azim=args.azim)
    ax_all.legend(fontsize=7, loc="upper left")
    ax_all.set_xlabel("X", fontsize=7)
    ax_all.set_ylabel("Y", fontsize=7)
    ax_all.set_zlabel("Z", fontsize=7)
    ax_all.tick_params(labelsize=6)

    plt.suptitle(f"PhysX-Omni VLM: {name}", fontsize=12)
    plt.tight_layout()
    out_path = input_dir / "voxel_vis.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
