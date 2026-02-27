#!/usr/bin/env python3
"""
Restriction-map visual analytics.

This utility produces four complementary views of the transports stored in
`restriction_maps.pt`:

1. Heatmaps (edge rows × restriction dimensions) expose coarse patterns such as saturation to ±1, 
   edge clusters with similar weights, or noisy layers.
2. Histograms plot the marginal distribution of every dimension. They show whether a dimension is 
   bimodal, collapsed to zero, or still exploring.
3. PCA scatter plots embed per-edge transports into 2-D. They highlight whether the learner 
   separates edge types (e.g., inter/intra-class edges) or if all edges live in the same region 
   of the transport space.
4. Top-k exemplars surface the highest-norm edge maps, letting you inspect concrete transports 
   instead of averages.

Reading these together gives both global and local intuition: heatmaps for
structure, histograms for coverage, PCA for clustering, and exemplars for
per-edge meaning.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


def parse_args():
    """
        Parse CLI arguments controlling which plots to generate.
    """
    parser = argparse.ArgumentParser(description="Plot restriction maps stored in restriction_maps.pt")
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="Path to source-causal/model_checkpoint/<dataset>/<model_timestamp> directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the generated figures (defaults to checkpoint_dir/plots)",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="coolwarm",
        help="Matplotlib colormap to use for the heatmaps.",
    )
    parser.add_argument("--no-heatmaps", action="store_true", help="Skip heatmap generation.")
    parser.add_argument("--histograms", action="store_true", help="Generate per-layer histograms/violin plots.")
    parser.add_argument("--clusters", action="store_true", help="Generate PCA scatter plots of edge transports.")
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of exemplar edges to plot (set to 0 to disable).",
    )
    return parser.parse_args()


def ensure_dir(base: Path, name: str) -> Path:
    """
        Create a child directory (e.g., heatmaps/) and return its Path.
    """
    sub = base / name
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def plot_heatmap(layer_name, matrix, out_path, cmap):
    """
        Visualise every edge × dimension entry.

        Interpretation:
            - Horizontal bands with similar hues indicate cohorts of edges sharing transports.
            - Saturated colors (deep reds/blues) reveal near-binary behaviour; pale areas mean weak transports.
            - Abrupt transitions across rows hint at outlier edges that may deserve inspection.
#     """
    matrix_np = matrix.detach().cpu().numpy()
    nrows, ncols = matrix_np.shape

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix_np, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_title(f"{layer_name} | shape {matrix_np.shape}")
    ax.set_xlabel("Restriction dimension")
    ax.set_ylabel("Edge index")
    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels([str(i) for i in range(ncols)])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_histogram(layer_name, matrix, out_path):
    """
        Plot a histogram for each restriction dimension.

        Interpretation:
            - Multi-modal curves suggest the learner split edges into regimes (e.g., positive vs negative).
            - Narrow spikes centred at zero imply that dimension carries little information.
            - Heavy tails may signal instability or the need for stronger regularisation.
    """
    matrix_np = matrix.detach().cpu().numpy()
    dims = matrix_np.shape[1]
    fig, axes = plt.subplots(dims, 1, figsize=(8, 2 * dims), sharex=True)
    if dims == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        ax.hist(matrix_np[:, idx], bins=50, alpha=0.7, color="steelblue")
        ax.set_ylabel(f"dim {idx}")
    axes[-1].set_xlabel("Restriction value")
    fig.suptitle(f"{layer_name} distribution per dimension")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_clusters(layer_name, matrix, out_path, mode='pca'):
    """
        Run PCA on the per-edge transports and scatter in 2-D.

        Interpretation:
            - Well-separated blobs imply the sheaf learned distinct regimes (perhaps correlating with structure).
            - A tight ball around the origin means transports collapsed, potentially underfitting.
            - Coloring by norm helps see whether magnitude drives the separation.
    """
    matrix_np = matrix.detach().cpu().numpy()
    flattened = matrix_np.reshape(matrix_np.shape[0], -1)
    if flattened.shape[0] < 2:
        return
    
    if mode == 'pca':
        model = PCA(n_components=2)
        label = "PC"
    elif mode == 'tsne':
        model = TSNE(n_components=2, random_state=42)
        label = "t-SNE"
    elif mode == 'umap':
        model = UMAP(n_components=2, random_state=42)
        label = "UMAP"
    else:
        raise ValueError(f"Unknown mode {mode} for plot_clusters")

    proj = model.fit_transform(flattened)
    norms = np.linalg.norm(flattened, axis=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=norms, cmap="viridis", s=5, alpha=0.8)
    title = f"{layer_name} {label} scatter"
    if mode == 'pca':
        var = model.explained_variance_ratio_.sum() * 100
        title += f" | Var={var:.1f}%"
    ax.set_title(title)
    ax.set_xlabel(f"{label} 1")
    ax.set_ylabel(f"{label} 2")
    plt.colorbar(scatter, ax=ax, label="Edge transport norm")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_topk(layer_name, matrix, out_dir, k):
    """
        Bar-chart the highest-norm edge transports for manual inspection.

        Interpretation:
            - Confirms what an “extreme” transport looks like per dimension.
            - Lets you see whether the top edges emphasise one dimension or distribute mass evenly.
            - Deviations here can hint at bad edges (e.g., exploding values confined to a few edges).
    """
    matrix_np = matrix.detach().cpu().numpy()
    flattened = matrix_np.reshape(matrix_np.shape[0], -1)
    norms = np.linalg.norm(flattened, axis=1)
    if norms.size == 0:
        return
    # Select the k largest norms (most “extreme” transports).
    top_indices = np.argsort(norms)[-k:][::-1]
    for rank, edge_idx in enumerate(top_indices, start=1):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(np.arange(matrix_np.shape[1]), matrix_np[edge_idx], color="coral")
        ax.set_title(f"{layer_name} | edge {edge_idx} | norm {norms[edge_idx]:.3f}")
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Value")
        fig.tight_layout()
        out_path = out_dir / f"{layer_name}_edge{edge_idx}_rank{rank}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def main():
    """Load checkpoint, fan out the requested plots, and explain what is saved."""
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir
    base_output = args.output_dir or checkpoint_dir / "plots"
    base_output.mkdir(parents=True, exist_ok=True)

    # ---- Load transports ----
    restriction_file = checkpoint_dir / "restriction_maps.pt"
    if not restriction_file.exists():
        raise FileNotFoundError(f"{restriction_file} not found. Did you pass a checkpoint directory?")

    # restriction_maps is an OrderedDict layer_name -> tensor [num_edges, d, ...]
    restriction_maps = torch.load(restriction_file, map_location="cpu")
    if not restriction_maps:
        raise RuntimeError(f"No maps found inside {restriction_file}")

    # Decide upfront which visualisations to produce; each toggles a folder below plots/.
    do_hist = args.histograms
    do_clusters = args.clusters
    do_topk = args.topk > 0
    do_heatmap = not args.no_heatmaps

    heatmap_dir = ensure_dir(base_output, "heatmaps") if do_heatmap else None
    hist_dir = ensure_dir(base_output, "histograms") if do_hist else None
    cluster_dir = ensure_dir(base_output, "clusters") if do_clusters else None
    topk_dir = ensure_dir(base_output, "exemplars") if do_topk else None

    # ---- Dispatch plots per layer ----
    for layer_name, tensor in restriction_maps.items():
        safe_name = layer_name.replace("/", "_")
        if do_heatmap:
            out_file = heatmap_dir / f"{safe_name}.png"
            #print(tensor.shape)
            #break
            plot_heatmap(layer_name, tensor, out_file, args.cmap)
            print(f"Saved heatmap {out_file}")
        if do_hist:
            out_file = hist_dir / f"{safe_name}_hist.png"
            plot_histogram(layer_name, tensor, out_file)
            print(f"Saved histogram {out_file}")
        if do_clusters:
            out_file = cluster_dir / f"{safe_name}_pca.png"
            plot_clusters(layer_name, tensor, out_file, mode='pca')
            print(f"Saved cluster plot {out_file}")
            
            out_file = cluster_dir / f"{safe_name}_tsne.png"
            plot_clusters(layer_name, tensor, out_file, mode='tsne')
            print(f"Saved cluster plot {out_file}")
            
            out_file = cluster_dir / f"{safe_name}_umap.png"
            plot_clusters(layer_name, tensor, out_file, mode='umap')
            print(f"Saved cluster plot {out_file}")
        if do_topk:
            plot_topk(layer_name, tensor, topk_dir, args.topk)


if __name__ == "__main__":
    main()
