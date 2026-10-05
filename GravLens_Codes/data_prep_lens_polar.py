"""
Polar + angular-harmonic representation of the SIE lens images, as a more
physically-motivated (and far lower-dimensional) alternative to the raw 2D
FFT used in data_prep_lens.py.

Rationale: an Einstein ring is a roughly circular/elliptical feature centered
on the lens. Resampling into polar coordinates (radius, angle) turns "where
is the ring" into "which row is bright" -- and a 1D Fourier transform along
the angular axis decomposes the ring's shape into angular harmonics, where
(per SIE lensing theory) the m=0 harmonic relates to overall brightness/
radius and the m=2 harmonic IS the ellipticity signal directly. Keeping only
the first few harmonics per radius bin gives a small, genuinely complex,
physically interpretable feature set -- and unlike a generic whole-image FFT,
this one also happens to map naturally onto a Quantum Fourier Transform /
amplitude encoding if this is ever ported to a QNN later.

Output:
  prepped_data/lens_cv_harmonic/{train,val,test}_X.npy  -- complex64, (N, N_R, N_HARMONICS)
  prepped_data/lens_cv_harmonic/{train,val,test}_y.npy  -- float32, (N, 4), same normalized targets/split as data_prep_lens.py
"""

import json
import os

import numpy as np
from scipy.ndimage import map_coordinates

IMG_SIZE = 64
IN_DIR = "training_data/lens"
OUT_DIR = "prepped_data/lens_cv_harmonic"

N_R = 32            # radial bins
N_THETA = 128        # angular samples per radius (before FFT)
N_HARMONICS = 9      # keep angular harmonics m = 0..8
R_MAX = 28.0         # stay within the 64x64 frame (center at 31.5, margin for interpolation)


def to_polar(images):
    """images: (N, H, W) real -> polar-resampled (N, N_R, N_THETA) real."""
    N, H, W = images.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

    r = np.linspace(0, R_MAX, N_R)
    theta = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)
    rr, tt = np.meshgrid(r, theta, indexing="ij")  # (N_R, N_THETA)
    xs = cx + rr * np.cos(tt)
    ys = cy + rr * np.sin(tt)

    polar = np.zeros((N, N_R, N_THETA), dtype=np.float32)
    for i in range(N):
        polar[i] = map_coordinates(images[i], [ys.ravel(), xs.ravel()], order=1, mode="constant", cval=0.0).reshape(N_R, N_THETA)
    return polar


def to_harmonics(polar):
    """polar: (N, N_R, N_THETA) real -> (N, N_R, N_HARMONICS) complex angular Fourier coefficients."""
    coeffs = np.fft.rfft(polar, axis=2)  # (N, N_R, N_THETA//2+1)
    return coeffs[:, :, :N_HARMONICS].astype(np.complex64)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    images = np.load(os.path.join(IN_DIR, "images.npy")).astype(np.float32)
    labels = np.load(os.path.join(IN_DIR, "labels.npy")).astype(np.float32)
    print("images shape:", images.shape)

    # Peak-normalize each image first (same convention as data_prep_lens.py)
    peak = images.reshape(len(images), -1).max(axis=1, keepdims=True)
    images_norm = images / np.maximum(peak, 1e-8)[:, None]

    print("Resampling to polar coordinates...")
    polar = to_polar(images_norm)
    print("polar shape:", polar.shape)

    print("Computing angular harmonics...")
    harmonics = to_harmonics(polar)
    print("harmonics shape:", harmonics.shape)

    # Same label normalization + split convention as data_prep_lens.py (identical
    # seed and permutation logic -> identical train/val/test split for fair comparison)
    y_min = labels.min(axis=0)
    y_max = labels.max(axis=0)
    labels_norm = (labels - y_min) / (y_max - y_min)

    rng = np.random.default_rng(42)
    n = len(images)
    perm = rng.permutation(n)
    harmonics, labels_norm = harmonics[perm], labels_norm[perm]

    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    splits = {
        "train": (harmonics[:n_train], labels_norm[:n_train]),
        "val": (harmonics[n_train:n_train + n_val], labels_norm[n_train:n_train + n_val]),
        "test": (harmonics[n_train + n_val:], labels_norm[n_train + n_val:]),
    }

    # Dataset-wide standardization of log-magnitude (computed from TRAIN split
    # only, applied to all splits) -- preserves phase exactly, and unlike
    # per-image peak normalization, keeps the relative brightness/prominence
    # differences between faint and bright lens systems that the network
    # needs to tell them apart.
    train_harm = splits["train"][0]
    train_logmag = np.log1p(np.abs(train_harm))
    cell_mean = train_logmag.mean(axis=0, keepdims=True)
    cell_std = train_logmag.std(axis=0, keepdims=True) + 1e-8

    def standardize(h):
        logmag = np.log1p(np.abs(h))
        z = (logmag - cell_mean) / cell_std
        phase = np.angle(h)
        return (z * np.exp(1j * phase)).astype(np.complex64)

    for name, (X, y) in splits.items():
        Xs = standardize(X)
        print(f"  {name}: X={Xs.shape} y={y.shape}")
        np.save(os.path.join(OUT_DIR, f"{name}_X.npy"), Xs)
        np.save(os.path.join(OUT_DIR, f"{name}_y.npy"), y)

    with open(os.path.join(OUT_DIR, "scaler.json"), "w") as f:
        json.dump({
            "y_min": y_min.tolist(), "y_max": y_max.tolist(),
            "param_names": ["theta_E", "f", "eps_x", "eps_y"],
            "n_r": N_R, "n_harmonics": N_HARMONICS, "r_max": R_MAX,
        }, f, indent=2)

    print("Done. Saved to", OUT_DIR)


if __name__ == "__main__":
    main()
