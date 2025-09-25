import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple


def plot_reconstruction_error_heatmap(
    original: np.ndarray,
    reconstructed: np.ndarray,
    vmax: float = None,
    title: str = "Reconstruction Error Heatmap",
    save_path: str = None
) -> None:
    """
    Plot a heatmap of per-sample, per-tap reconstruction errors.

    Args:
        original:    np.ndarray of shape (N, L, C) with ground-truth [mag, sin, cos].
        reconstructed: np.ndarray of shape (N, L, C), same shape as original.
        vmax:        Optional float to set maximum color scale (for consistency).
        title:       Title for the plot.
        save_path:   If provided, path to save the figure (PNG).

    This computes error = sqrt(sum_c (recon - orig)^2) for each sample and tap,
    yielding an error matrix of shape (N, L), then displays it as an image.
    """
    # Validate shapes
    assert original.shape == reconstructed.shape, \
        f"Shape mismatch: original {original.shape}, reconstructed {reconstructed.shape}"
    N, L, C = original.shape
    # Compute per-tap error magnitude
    errors = np.linalg.norm(reconstructed - original, axis=2)  # (N, L)

    plt.figure(figsize=(10, N * 0.2 + 2))
    im = plt.imshow(errors, aspect='auto', cmap='viridis', vmax=vmax)
    plt.colorbar(im, label='Reconstruction Error')
    plt.ylabel('Sample Index')
    plt.xlabel('Tap Index')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_embedding_manifold(
    embeddings: Dict[str, np.ndarray],
    perplexity: float = 30,
    random_state: int = 0,
    title: str = "t-SNE Embedding",
    save_path: str = None
) -> None:
    """
    Compute a 2D t-SNE projection of concatenated embeddings from multiple domains
    and plot them with different colors/labels.

    Args:
        embeddings:   Dict mapping label (e.g. 'internal', 'external_pre', 'external_post')
                      to np.ndarray of shape (N_i, D).
        perplexity:   t-SNE perplexity parameter.
        random_state: Random seed for reproducibility.
        title:        Plot title.
        save_path:    If provided, path to save the figure.

    This concatenates all embeddings, runs TSNE(n_components=2), and colors points
    according to their source key.
    """
    # Concatenate embeddings and build labels
    all_embeds = []
    labels = []
    for key, arr in embeddings.items():
        assert arr.ndim == 2, f"Embeddings for {key} must be 2D, got {arr.ndim}D"
        all_embeds.append(arr)
        labels.extend([key] * arr.shape[0])
    X = np.vstack(all_embeds)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X2 = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for key in embeddings:
        mask = np.array(labels) == key
        plt.scatter(X2[mask, 0], X2[mask, 1], label=key, s=10, alpha=0.7)
    plt.legend()
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_correlation_recon_vs_localization(
    recon_losses: np.ndarray,
    loc_errors: np.ndarray,
    title: str = "Reconstruction Loss vs Localization Error",
    save_path: str = None
) -> None:
    """
    Scatter plot of per-sample reconstruction loss vs. point-to-point localization error.
    """
    # Sanity check shapes to prevent silent misalignment
    assert recon_losses.shape == loc_errors.shape, \
        f"Length mismatch: recon {recon_losses.shape}, loc {loc_errors.shape}"

    plt.figure(figsize=(6, 6))
    plt.scatter(recon_losses, loc_errors, alpha=0.6)
    # Trend line
    m, b = np.polyfit(recon_losses, loc_errors, 1)
    xs = np.linspace(recon_losses.min(), recon_losses.max(), 100)
    plt.plot(xs, m * xs + b, linestyle='--', label=f'y = {m:.2f}x + {b:.2f}')
    # More precise axis label matching plotted metric
    plt.xlabel('Per-sample reconstruction MSE')
    plt.ylabel('Localization Error (cm)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()            # Ensure no clipping
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()                   # Prevent figure buildup



def plot_delay_spread_vs_error(
    delay_spreads: np.ndarray,
    loc_errors: np.ndarray,
    bins: int = 10,
    title: str = "Delay Spread vs Localization Error",
    save_path: str = None
) -> None:
    """
    Plot localization error as a function of RMS delay spread using value-based bins.
    """
    assert delay_spreads.shape == loc_errors.shape, \
        f"Length mismatch: delay {delay_spreads.shape}, loc {loc_errors.shape}"

    # Create uniform-width bins over the actual delay-spread range
    bin_edges   = np.linspace(delay_spreads.min(), delay_spreads.max(), bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    med_errors  = []

    # Compute median error in each numeric bin
    for i in range(bins):
        left, right = bin_edges[i], bin_edges[i+1]
        if i < bins - 1:
            mask = (delay_spreads >= left) & (delay_spreads < right)
        else:
            # include the max edge in the last bin
            mask = (delay_spreads >= left) & (delay_spreads <= right)
        if mask.any():
            med_errors.append(np.median(loc_errors[mask]))
        else:
            med_errors.append(np.nan)

    plt.figure(figsize=(8, 5))
    width = bin_edges[1] - bin_edges[0]
    plt.bar(bin_centers, med_errors, width=width, edgecolor='black')
    plt.xlabel('Delay Spread (value-based bins)')
    plt.ylabel('Median Localization Error (cm)')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

    
def plot_error_cdf(errors, title, save_path):
    # errors: 1D array of localization distances in cm
    sorted_e = np.sort(errors)
    p = np.arange(1, len(sorted_e)+1) / len(sorted_e)
    plt.figure()
    plt.plot(sorted_e, p)
    plt.xlabel("Localization error (cm)")
    plt.ylabel("Empirical CDF")
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
