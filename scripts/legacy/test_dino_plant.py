"""
Test DINOv2 feature extraction on real plant images.
Visualize: PCA of patch features + spectral clustering into trunk/branch/leaf.
"""
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, KMeans

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_dino(model_name="dinov2_vits14", device="cuda"):
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract_features(model, img_pil, device="cuda", max_dim=518):
    patch_size = 14

    W_raw, H_raw = img_pil.size
    scale = min(max_dim / H_raw, max_dim / W_raw)
    if scale < 1.0:
        H_raw = int(H_raw * scale)
        W_raw = int(W_raw * scale)

    H = (H_raw // patch_size) * patch_size
    W = (W_raw // patch_size) * patch_size
    img_resized = img_pil.resize((W, H), Image.LANCZOS)

    img_t = torch.from_numpy(np.array(img_resized)).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_t = (img_t - mean) / std
    img_t = img_t.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model.forward_features(img_t)
        patch_tokens = out["x_norm_patchtokens"]  # [1, N_patches, D]

    H_feat = H // patch_size
    W_feat = W // patch_size
    feat_map = patch_tokens[0].cpu().numpy()  # [H_feat*W_feat, D]

    return feat_map, H_feat, W_feat, np.array(img_resized)


def visualize(img, feat_map, H_feat, W_feat, n_clusters, save_path, name):
    D = feat_map.shape[1]
    fg_mask = None

    img_small = Image.fromarray(img).resize((W_feat, H_feat), Image.LANCZOS)
    img_small_arr = np.array(img_small)
    brightness = img_small_arr.mean(axis=-1) / 255.0
    fg_mask = brightness < 0.95

    fg_flat = fg_mask.reshape(-1)
    fg_feats = feat_map[fg_flat]

    if len(fg_feats) < n_clusters:
        print(f"  Too few foreground patches ({len(fg_feats)}), skipping")
        return

    # PCA to 3 dims for RGB visualization
    pca = PCA(n_components=3)
    pca_all = pca.fit_transform(feat_map)
    pca_fg = pca_all[fg_flat]

    # Normalize PCA to [0,1] for visualization
    pca_vis = pca_all.copy()
    for c in range(3):
        mn, mx = pca_fg[:, c].min(), pca_fg[:, c].max()
        pca_vis[:, c] = (pca_vis[:, c] - mn) / (mx - mn + 1e-8)
    pca_vis = np.clip(pca_vis, 0, 1)
    pca_img = pca_vis.reshape(H_feat, W_feat, 3)

    # Clustering on foreground only
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    fg_labels = kmeans.fit_predict(fg_feats)

    label_map = np.full(H_feat * W_feat, -1, dtype=int)
    label_map[fg_flat] = fg_labels
    label_img = label_map.reshape(H_feat, W_feat)

    cluster_colors = np.array([
        [0.9, 0.3, 0.3],  # red - likely trunk/pot
        [0.6, 0.4, 0.2],  # brown - likely branch
        [0.2, 0.7, 0.3],  # green - likely leaf
        [0.2, 0.5, 0.8],  # blue
        [0.9, 0.6, 0.1],  # orange
        [0.6, 0.2, 0.8],  # purple
    ])

    cluster_vis = np.ones((H_feat, W_feat, 3))
    for c in range(n_clusters):
        mask = label_img == c
        cluster_vis[mask] = cluster_colors[c % len(cluster_colors)]

    # Also try spectral clustering
    n_fg = len(fg_feats)
    n_neighbors = min(10, n_fg - 1)
    spectral = SpectralClustering(
        n_clusters=n_clusters, affinity="nearest_neighbors",
        n_neighbors=n_neighbors, random_state=42,
    )
    spec_labels = spectral.fit_predict(fg_feats)

    spec_label_map = np.full(H_feat * W_feat, -1, dtype=int)
    spec_label_map[fg_flat] = spec_labels
    spec_label_img = spec_label_map.reshape(H_feat, W_feat)

    spec_vis = np.ones((H_feat, W_feat, 3))
    for c in range(n_clusters):
        mask = spec_label_img == c
        spec_vis[mask] = cluster_colors[c % len(cluster_colors)]

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(img)
    axes[0].set_title(f"Input: {name}")
    axes[0].axis("off")

    axes[1].imshow(pca_img)
    axes[1].set_title("DINO PCA (3 components)")
    axes[1].axis("off")

    axes[2].imshow(cluster_vis)
    axes[2].set_title(f"KMeans ({n_clusters} clusters)")
    axes[2].axis("off")

    axes[3].imshow(spec_vis)
    axes[3].set_title(f"Spectral ({n_clusters} clusters)")
    axes[3].axis("off")

    plt.suptitle(f"DINOv2 Plant Decomposition: {name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="data/dino_plant_test")
    parser.add_argument("--n_clusters", type=int, default=3)
    parser.add_argument("--model", type=str, default="dinov2_vits14")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading DINOv2 ({args.model})...")
    dino = load_dino(args.model, args.device)

    for img_path in args.images:
        name = Path(img_path).stem
        print(f"\nProcessing: {name}")

        img_pil = Image.open(img_path).convert("RGB")
        feat_map, H_feat, W_feat, img_arr = extract_features(dino, img_pil, args.device)
        print(f"  Feature map: {H_feat}x{W_feat} patches, dim={feat_map.shape[1]}")

        save_path = output_dir / f"{name}_dino.png"
        visualize(img_arr, feat_map, H_feat, W_feat, args.n_clusters, save_path, name)

        # Also try with more clusters
        save_path_5 = output_dir / f"{name}_dino_5clusters.png"
        visualize(img_arr, feat_map, H_feat, W_feat, 5, save_path_5, name)

    print("\nDone!")


if __name__ == "__main__":
    main()
