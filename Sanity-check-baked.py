#!/usr/bin/env python3
"""
sanity_check_baked.py

Load the internal and external baked CIR datasets (.npz), verify channel shapes,
reconstruct complex CIR taps, and plot the mean PSD with your cutoff_bin marker.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config_loader import load_config

# --- User parameters: adjust if you want to test different values ---
# time interval between taps (seconds)
time_bin_width = 1e-9
# cutoff frequency bin index
cutoff_bin = 10
# number of samples to use for PSD debug
n_debug_samples = 100

# --- Locate the baked .npz files via config.yaml ---
cfg = load_config("config.yaml")
base_outdir = Path(cfg.output.directory)
int_path = base_outdir / "internal_baked.npz"
ext_path = base_outdir / "external_baked.npz"

if not int_path.exists() or not ext_path.exists():
    raise FileNotFoundError(f"Please ensure both {int_path} and {ext_path} exist.")

# --- Load and sanity-check shapes ---
data_int = np.load(int_path)
data_ext = np.load(ext_path)

for name, data in ("internal", data_int), ("external", data_ext):
    X = data["X"]  # shape (N, L, 3)
    assert X.ndim == 3 and X.shape[2] == 3, (
        f"[{name}] CIR array wrong shape: {X.shape}, expected (N, L, 3)"
    )
    print(f"[{name}] Loaded CIR data: samples={X.shape[0]}, taps={X.shape[1]}, channels={X.shape[2]}")

# --- Reconstruct complex CIR taps from channels ---
# X[...,0] = magnitude, X[...,1] = sin(phase), X[...,2] = cos(phase)
X_debug = data_ext["X"][:n_debug_samples]
mag = X_debug[..., 0]
sin_ph = X_debug[..., 1]
cos_ph = X_debug[..., 2]
# h = mag * (cos + j sin)
tap_matrix = mag * (cos_ph + 1j * sin_ph)  # shape (n_debug_samples, L)

# --- PSD computation ---
M = tap_matrix.shape[1]
nfft = M  # use M-length FFT

# FFT and PSD
taps_to_fft = tap_matrix if tap_matrix.shape[0] <= n_debug_samples else tap_matrix[:n_debug_samples]
H = np.fft.fft(taps_to_fft, n=nfft, axis=1)
psd = np.abs(H) ** 2
mean_psd = psd.mean(axis=0)  # mean over debug samples

# Build frequency axis
dt = time_bin_width  # seconds per tap
fs = 1.0 / dt           # sampling frequency (Hz)
freqs = np.fft.fftfreq(nfft, d=1/fs)
pos = freqs >= 0
freqs_pos = freqs[pos]
psd_pos = mean_psd[pos]

# --- Plot mean PSD ---
plt.figure(figsize=(6, 3))
plt.plot(freqs_pos / 1e6, 10 * np.log10(psd_pos), label='Mean PSD (dB)')
plt.axvline(
    freqs_pos[cutoff_bin] / 1e6,
    color='red', linestyle='--',
    label=f'cutoff_bin={cutoff_bin} (@{freqs_pos[cutoff_bin]/1e6:.1f} MHz)'
)
plt.title('PSD sanity check')
plt.xlabel('Frequency (MHz)')
plt.ylabel('PSD (dB)')
plt.legend(loc='best')
plt.tight_layout()
out_dir = base_outdir / "debug"
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / "psd_sanity_check.png", dpi=300)
plt.show()
