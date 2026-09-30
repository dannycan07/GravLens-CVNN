"""
Prep the synthetic SIE lens images (training_data/lens/) for both the RVNN
(raw pixel intensities) and CVNN (2D FFT of the image) regressors.

Targets: [theta_E, f, eps_x, eps_y] -- min-max normalized to [0, 1] per
parameter for stable multi-target regression (scaler stats saved so
evaluation scripts can report errors in original physical units).

Output:
  prepped_data/lens_rv/{train_X,train_y,val_X,val_y,test_X,test_y}.npy  -- real, (N,1,H,W) float32
  prepped_data/lens_cv/{train_X,train_y,val_X,val_y,test_X,test_y}.npy  -- complex64, (N,1,H,W)
  prepped_data/lens_scaler.json  -- {"y_min": [...], "y_max": [...], "param_names": [...]}
"""

import json
import os

import numpy as np

IMG_SIZE = 64
PARAM_NAMES = ["theta_E", "f", "eps_x", "eps_y"]
IN_DIR = "training_data/lens"
RV_OUT_DIR = "prepped_data/lens_rv"
CV_OUT_DIR = "prepped_data/lens_cv"


def main():
    os.makedirs(RV_OUT_DIR, exist_ok=True)
    os.makedirs(CV_OUT_DIR, exist_ok=True)

    images = np.load(os.path.join(IN_DIR, "images.npy")).astype(np.float32)  # (N, H, W)
    labels = np.load(os.path.join(IN_DIR, "labels.npy")).astype(np.float32)  # (N, 4)
    print("images shape:", images.shape, "labels shape:", labels.shape)

    # Normalize image intensity per-image to [0, 1] (peak brightness varies a
    # lot with theta_E/amplitude; put every image on the same intensity scale)
    peak = images.reshape(len(images), -1).max(axis=1, keepdims=True)
    images = images / np.maximum(peak[:, None], 1e-8)
    images = images[:, None, :, :]  # (N, 1, H, W)

    # Min-max normalize labels to [0, 1] per parameter
    y_min = labels.min(axis=0)
    y_max = labels.max(axis=0)
    labels_norm = (labels - y_min) / (y_max - y_min)

    # Shuffle, then split 70/15/15 train/val/test
    rng = np.random.default_rng(42)
    n = len(images)
    perm = rng.permutation(n)
    images, labels_norm = images[perm], labels_norm[perm]

    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    splits = {
        "train": (images[:n_train], labels_norm[:n_train]),
        "val": (images[n_train:n_train + n_val], labels_norm[n_train:n_train + n_val]),
        "test": (images[n_train + n_val:], labels_norm[n_train + n_val:]),
    }
    for name, (X, y) in splits.items():
        print(f"  {name}: X={X.shape} y={y.shape}")

    # RVNN: raw pixel intensities
    for name, (X, y) in splits.items():
        np.save(os.path.join(RV_OUT_DIR, f"{name}_X.npy"), X)
        np.save(os.path.join(RV_OUT_DIR, f"{name}_y.npy"), y)

    # CVNN: centered 2D FFT (genuinely complex). These lens images are mostly
    # empty background with a thin, faint ring, so their FFT magnitude is
    # extremely heavy-tailed (huge DC spike, near-nothing elsewhere) -- fed
    # raw, this reliably kills ReLU activations within 1-2 layers (verified:
    # >96% of post-conv units go permanently to zero). Compress the dynamic
    # range with log1p on the magnitude while preserving phase exactly, then
    # renormalize per-image so the log-magnitude peaks at 1 (standard
    # log-spectrogram-style scaling for spectral/FFT inputs to neural nets).
    def to_fft(X):
        spec = np.fft.fft2(X, axes=(-2, -1))
        spec = np.fft.fftshift(spec, axes=(-2, -1))
        mag = np.log1p(np.abs(spec))
        peak = mag.reshape(len(mag), -1).max(axis=1)
        mag = mag / np.maximum(peak[:, None, None, None], 1e-8)
        phase = np.angle(spec)
        return (mag * np.exp(1j * phase)).astype(np.complex64)

    for name, (X, y) in splits.items():
        np.save(os.path.join(CV_OUT_DIR, f"{name}_X.npy"), to_fft(X))
        np.save(os.path.join(CV_OUT_DIR, f"{name}_y.npy"), y)

    with open("prepped_data/lens_scaler.json", "w") as f:
        json.dump({
            "y_min": y_min.tolist(), "y_max": y_max.tolist(),
            "param_names": PARAM_NAMES, "img_size": IMG_SIZE,
        }, f, indent=2)

    print("Done. Saved to", RV_OUT_DIR, "and", CV_OUT_DIR)


if __name__ == "__main__":
    main()
