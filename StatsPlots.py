import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp, skew
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# — Imports from your project —
from config_loader import load_config
from utils import ensure_output_dir, get_run_output_dir

def clamp_and_normalize(X_source, X_target):
    Xs = X_source.copy()
    Xt = X_target.copy()

    # Clamp only magnitude channel (ch 0)
    mag = Xs[..., 0]
    q1 = np.percentile(mag, 25, axis=0)
    q3 = np.percentile(mag, 75, axis=0)
    iqr = q3 - q1
    clamp_min = q1 - 1.5 * iqr
    clamp_max = q3 + 1.5 * iqr

    for X in [Xs, Xt]:
        X[..., 0] = np.clip(X[..., 0], clamp_min, clamp_max)

    # Normalize using clamped source stats
    means = Xs.mean(axis=(0, 1))
    stds = Xs.std(axis=(0, 1)) + 1e-6
    Xs = (Xs - means[None, None, :]) / stds[None, None, :]
    Xt = (Xt - means[None, None, :]) / stds[None, None, :]

    return Xs, Xt

def plot_mean_shift(df, save_path="mean_shift_vs_tap.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(df["tap"], df["mean_src"], label="Source Mean")
    plt.plot(df["tap"], df["mean_tgt"], label="Target Mean")
    plt.xlabel("Tap index")
    plt.ylabel("Mean (Normalized Magnitude)")
    plt.title("Mean Feature Shift across Taps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved mean shift plot to: {save_path}")


def summarize_domain_shift(Xs, Xt, feature_index=0):
    L = Xs.shape[1]
    summary_rows = []
    for tap in range(L):
        src = Xs[:, tap, feature_index]
        tgt = Xt[:, tap, feature_index]
        summary_rows.append({
            "tap": tap,
            "mean_src": src.mean(),
            "mean_tgt": tgt.mean(),
            "std_src": src.std(),
            "std_tgt": tgt.std(),
            "skew_src": skew(src),
            "skew_tgt": skew(tgt),
            "emd": wasserstein_distance(src, tgt),
            "ks_stat": ks_2samp(src, tgt).statistic,
            "ks_pval": ks_2samp(src, tgt).pvalue
        })
    return pd.DataFrame(summary_rows)

def plot_emd_vs_tap(df, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(df["tap"], df["emd"], label="EMD (Wasserstein Distance)")
    plt.xlabel("Tap index")
    plt.ylabel("EMD")
    plt.title("Domain Shift across Taps (Magnitude Channel)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f" Saved EMD plot to: {save_path}")
    plt.close()
def plot_combined_shift(df, save_path="domain_shift_combined.png"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    # Left: EMD plot
    axes[0].plot(df["tap"], df["emd"], color="tab:blue")
    axes[0].set_title("Wasserstein Distance (EMD)")
    axes[0].set_xlabel("Tap index")
    axes[0].set_ylabel("Distance")
    axes[0].grid(True)

    # Right: Mean shift plot
    axes[1].plot(df["tap"], df["mean_src"], label="Source Mean", color="tab:blue")
    axes[1].plot(df["tap"], df["mean_tgt"], label="Target Mean", color="tab:orange")
    axes[1].set_title("Mean Feature Shift")
    axes[1].set_xlabel("Tap index")
    axes[1].set_ylabel("Mean (Normalized Magnitude)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f" Saved combined shift plot to: {save_path}")

def plot_combined_shift_dual_regions(df, save_path="domain_shift_combined.pdf", emd_thresh=0.1, mean_thresh=0.1):
    import matplotlib.pyplot as plt
    import numpy as np

    taps = df["tap"].values
    emd_vals = df["emd"].values
    mean_src = df["mean_src"].values
    mean_tgt = df["mean_tgt"].values
    mean_diff = np.abs(mean_src - mean_tgt)

    def get_mask_regions(values, threshold):
        return values >= threshold

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    # Plot 1: EMD with shaded under-curve where EMD >= threshold
    mask_emd = get_mask_regions(emd_vals, emd_thresh)
    axes[0].plot(taps, emd_vals, color="tab:blue", label="EMD")
    axes[0].fill_between(taps, 0, emd_vals, where=mask_emd, color="red", alpha=0.2, label=f"EMD ≥ {emd_thresh}")
    axes[0].set_title("Wasserstein Distance (EMD)")
    axes[0].set_xlabel("Tap index")
    axes[0].set_ylabel("EMD")
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Mean shift with absolute difference curve and shaded region
    mask_mean = get_mask_regions(mean_diff, mean_thresh)
    axes[1].plot(taps, mean_src, label="Source Mean", color="tab:blue")
    axes[1].plot(taps, mean_tgt, label="Target Mean", color="tab:orange")
    axes[1].plot(taps, mean_diff, label="|Δmean|", color="tab:green", linestyle="--")
    axes[1].fill_between(taps, 0, mean_diff, where=mask_mean, color="red", alpha=0.2, label=f"|Δmean| ≥ {mean_thresh}")
    axes[1].set_title("Mean Feature Shift")
    axes[1].set_xlabel("Tap index")
    axes[1].set_ylabel("Mean (Normalized Magnitude)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f" Saved fixed combined plot (PDF) to: {save_path}")





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_outdir = Path(cfg.output.directory)
    out_dir = Path(get_run_output_dir(cfg))
    ensure_output_dir(out_dir)

    # Load baked data (truncated to 100 taps like in training)
    X_source = np.load(base_outdir / "internal_baked.npz")["X"][:, :100, :]
    X_target = np.load(base_outdir / "external_baked.npz")["X"][:, :100, :]

    # Clamp and normalize
    Xs_norm, Xt_norm = clamp_and_normalize(X_source, X_target)

    # Stats
    df_stats = summarize_domain_shift(Xs_norm, Xt_norm, feature_index=0)

    # Save CSV
    stats_path = out_dir / "domain_shift_stats_magnitude.csv"
    df_stats.to_csv(stats_path, index=False)
    print(f" Saved stats to: {stats_path}")

    # Plot EMD
    plot_path_emd = out_dir / "emd_vs_tap.png"
    plot_emd_vs_tap(df_stats, save_path=plot_path_emd)

    # Plot Mean Shift
    plot_path_mean = out_dir / "mean_shift_vs_tap.png"
    plot_mean_shift(df_stats, save_path=plot_path_mean)

    # Plot Combined (Side-by-side)
    plot_path_combined = out_dir / "domain_shift_combined.png"
    plot_combined_shift(df_stats, save_path=plot_path_combined)
        # Final combined plot with both EMD + Mean shift regions
    plot_combined_shift_dual_regions(
        df_stats,
        save_path=out_dir / "domain_shift_combined.pdf"
    )


if __name__ == "__main__":
    main()

