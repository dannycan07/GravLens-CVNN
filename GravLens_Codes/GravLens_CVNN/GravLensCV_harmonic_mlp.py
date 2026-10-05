"""
CVNN regressor operating on the polar/angular-harmonic representation
(data_prep_lens_polar.py) instead of the raw whole-image 2D FFT. Reuses the
same complex layer primitives and training loop as GravLensCV_cnn.py.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from GravLensCV_cnn import ComplexLinear, CRELU, ComplexBatchNorm, evaluate, train_model, log_history, device

NUM_PARAMS = 4
N_R = 32
N_HARMONICS = 9


class GravLensHarmonicMLP(nn.Module):
    """Small complex MLP on flattened (radius x angular-harmonic) coefficients."""

    def __init__(self, n_r=N_R, n_harmonics=N_HARMONICS, num_params=NUM_PARAMS):
        super().__init__()
        in_features = n_r * n_harmonics
        self.fc1 = ComplexLinear(in_features, 128)
        self.bn1 = ComplexBatchNorm(128, spatial=False)
        self.fc2 = ComplexLinear(128, 64)
        self.bn2 = ComplexBatchNorm(64, spatial=False)
        self.fc3 = ComplexLinear(64, num_params)
        self.act = CRELU()
        # See GravLensCV_cnn.py's GravLensComplexCNN for why sigmoid(x.abs())
        # is wrong (restricts output to [0.5, 1.0]) and this real-valued
        # readout over [real, imag] is used instead.
        self.readout = nn.Linear(num_params * 2, num_params)

    def forward(self, x):
        x = x.flatten(1)
        x = self.act(self.bn1(self.fc1(x)))
        x = self.act(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        logits = self.readout(torch.cat([x.real, x.imag], dim=-1))
        return torch.sigmoid(logits)


if __name__ == "__main__":
    DATA_DIR = "prepped_data/lens_cv_harmonic"
    train_X = torch.from_numpy(np.load(os.path.join(DATA_DIR, "train_X.npy"))).to(torch.complex64)
    train_y = torch.from_numpy(np.load(os.path.join(DATA_DIR, "train_y.npy"))).to(torch.float32)
    val_X = torch.from_numpy(np.load(os.path.join(DATA_DIR, "val_X.npy"))).to(torch.complex64)
    val_y = torch.from_numpy(np.load(os.path.join(DATA_DIR, "val_y.npy"))).to(torch.float32)

    print("train_X shape:", train_X.shape, "val_X shape:", val_X.shape)

    SAVE_DIR = "saved_models/lens_cv_harmonic"
    model = GravLensHarmonicMLP()
    print("Model has", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters.")

    train_model(model, train_X, train_y, val_X, val_y, epochs=300, lr=0.001, save_dir=SAVE_DIR,
                history_path="results/lens_cv_harmonic_loss_history.csv",
                ckpt_prefix="lens_cv_harmonic_model")
