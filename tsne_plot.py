# tsne_plot.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path

def _run_tsne(features: np.ndarray, perplexity: int, max_iter: int, seed: int) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=max_iter,
        random_state=seed,
        init="pca",
        learning_rate="auto"
    )
    return tsne.fit_transform(features)

def plot_tsne_side_by_side(
    source_feats_1: np.ndarray,
    target_feats_1: np.ndarray,
    source_feats_2: np.ndarray,
    target_feats_2: np.ndarray,
    label_1: str,
    label_2: str,
    out_path: Path,
    perplexity: int = 30,
    max_iter: int = 3000,
    seed: int = 42
):
    """
    Plots two side-by-side t-SNE projections for source vs. target features.

    Parameters:
        source_feats_1 (np.ndarray): shape (N1, D), for first comparison
        target_feats_1 (np.ndarray): shape (M1, D)
        source_feats_2 (np.ndarray): shape (N2, D), for second comparison
        target_feats_2 (np.ndarray): shape (M2, D)
        label_1 (str): e.g., "Before GRL"
        label_2 (str): e.g., "After GRL"
        out_path (Path): Path to save combined PDF figure
    """
    # Combine + t-SNE for first plot
    all_feats_1 = np.vstack([source_feats_1, target_feats_1])
    labels_1 = np.array([0] * len(source_feats_1) + [1] * len(target_feats_1))
    proj_1 = _run_tsne(all_feats_1, perplexity, max_iter, seed)

    # Combine + t-SNE for second plot
    all_feats_2 = np.vstack([source_feats_2, target_feats_2])
    labels_2 = np.array([0] * len(source_feats_2) + [1] * len(target_feats_2))
    proj_2 = _run_tsne(all_feats_2, perplexity, max_iter, seed)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)

    for ax, proj, domain_labels, title in zip(
        axes,
        [proj_1, proj_2],
        [labels_1, labels_2],
        [label_1, label_2]
    ):
        colors = ['tab:blue', 'tab:orange']
        names = ['Source', 'Target']
        for i, name in enumerate(names):
            idxs = domain_labels == i
            ax.scatter(proj[idxs, 0], proj[idxs, 1], s=10, alpha=0.7, label=name, color=colors[i])
        ax.set_title(f"t-SNE: {title}")
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
