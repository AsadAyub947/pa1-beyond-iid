"""
2D projection (t-SNE or UMAP) of clean + transformed features for a single
backbone, fit jointly so both conditions live in the same embedding space.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fit_projection(features: np.ndarray, method: str = "tsne", seed: int = 6304,
                    tsne_perplexity: int = 30, umap_n_neighbors: int = 15,
                    umap_min_dist: float = 0.1) -> np.ndarray:
    """features: (N, D) combined clean+transformed features for ONE backbone.
    Returns (N, 2) projected coordinates. A single projection is fit per
    backbone -- do not compare absolute coordinates across backbones."""
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=seed, perplexity=tsne_perplexity,
                        init="pca")
    elif method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=seed,
                             n_neighbors=umap_n_neighbors, min_dist=umap_min_dist)
    else:
        raise ValueError(f"Unknown method: {method}")
    return reducer.fit_transform(features)


def plot_projection(coords: np.ndarray, class_labels: np.ndarray,
                     condition_labels: np.ndarray, class_names: list,
                     title: str, save_path: str):
    """
    coords: (N, 2)
    class_labels: (N,) int, ground-truth class id -> color
    condition_labels: (N,) str, e.g. "clean" / "grayscale" -> marker style
    """
    unique_conditions = sorted(set(condition_labels))
    markers = ["o", "^", "s", "D", "P", "X", "v"]
    marker_map = {cond: markers[i % len(markers)] for i, cond in enumerate(unique_conditions)}

    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(7, 6))
    for cond in unique_conditions:
        mask = condition_labels == cond
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(c % 10) for c in class_labels[mask]],
                   marker=marker_map[cond], s=18, alpha=0.75,
                   label=cond, edgecolors="none")

    # Class color legend
    class_handles = [plt.Line2D([0], [0], marker="o", color="w",
                                 markerfacecolor=cmap(i % 10), markersize=8,
                                 label=name)
                      for i, name in enumerate(class_names)]
    # Condition marker legend
    cond_handles = [plt.Line2D([0], [0], marker=marker_map[cond], color="gray",
                                linestyle="", markersize=8, label=cond)
                     for cond in unique_conditions]

    leg1 = ax.legend(handles=class_handles, title="Class", bbox_to_anchor=(1.02, 1),
                      loc="upper left", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=cond_handles, title="Condition", bbox_to_anchor=(1.02, 0.4),
              loc="upper left", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
