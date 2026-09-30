import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def log_history(path, row):
    """Append one epoch's metrics to a CSV loss-history file (writes header on first call)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)

NUM_PARAMS = 4  # theta_E, f, eps_x, eps_y


### Complex conv/linear layers (same real+imag weight-pair pattern used
### throughout the project's other CVNN scripts)
class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.padding = padding
        self.weight_real = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_imag = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias_real = nn.Parameter(torch.zeros(out_channels))
        self.bias_imag = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))

    def forward(self, x):
        real, imag = x.real, x.imag
        out_real = (F.conv2d(real, self.weight_real, self.bias_real, padding=self.padding)
                    - F.conv2d(imag, self.weight_imag, None, padding=self.padding))
        out_imag = (F.conv2d(real, self.weight_imag, self.bias_imag, padding=self.padding)
                    + F.conv2d(imag, self.weight_real, None, padding=self.padding))
        return torch.complex(out_real, out_imag)


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_real = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_real = nn.Parameter(torch.zeros(out_features))
        self.bias_imag = nn.Parameter(torch.zeros(out_features))
        std = math.sqrt(1.0 / in_features)
        nn.init.uniform_(self.weight_real, -std * math.sqrt(3), std * math.sqrt(3))
        nn.init.uniform_(self.weight_imag, -std * math.sqrt(3), std * math.sqrt(3))

    def forward(self, x):
        real, imag = x.real, x.imag
        out_real = torch.matmul(real, self.weight_real.t()) - torch.matmul(imag, self.weight_imag.t()) + self.bias_real
        out_imag = torch.matmul(real, self.weight_imag.t()) + torch.matmul(imag, self.weight_real.t()) + self.bias_imag
        return torch.complex(out_real, out_imag)


class CRELU(nn.Module):
    def forward(self, x):
        return torch.complex(F.relu(x.real), F.relu(x.imag))


class ComplexAvgPool2d(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        return torch.complex(F.avg_pool2d(x.real, self.kernel_size), F.avg_pool2d(x.imag, self.kernel_size))


class ComplexAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        real = F.adaptive_avg_pool2d(x.real, self.output_size)
        imag = F.adaptive_avg_pool2d(x.imag, self.output_size)
        return torch.complex(real, imag)


### CV-CNN regressor -- same conv/pool schedule as the RVNN counterpart,
### operating on the image's centered 2D FFT (genuinely complex input)
class GravLensComplexCNN(nn.Module):
    def __init__(self, num_params=NUM_PARAMS):
        super().__init__()
        self.features = nn.Sequential(
            ComplexConv2d(1, 16, kernel_size=3, padding=1), CRELU(), ComplexAvgPool2d(2),   # 64 -> 32
            ComplexConv2d(16, 32, kernel_size=3, padding=1), CRELU(), ComplexAvgPool2d(2),  # 32 -> 16
            ComplexConv2d(32, 64, kernel_size=3, padding=1), CRELU(), ComplexAvgPool2d(2),  # 16 -> 8
        )
        self.pool = ComplexAdaptiveAvgPool2d(1)
        self.fc1 = ComplexLinear(64, 64)
        self.act = CRELU()
        self.fc2 = ComplexLinear(64, num_params)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x.abs())  # magnitude of final complex output -> [0,1] normalized targets


def evaluate(model, X, y, batch_size=128):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size].to(device)
            yb = y[i:i + batch_size].to(device)
            pred = model(xb)
            total_loss += criterion(pred, yb).item()
    return total_loss / (len(X) * y.shape[1])


def train_model(model, train_X, train_y, val_X, val_y, epochs, lr,
                 batch_size=64, save_dir="saved_models/lens_cv",
                 start_epoch=0, best_vloss=float("inf")):
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n = len(train_X)
    total_epochs = start_epoch + epochs

    for epoch in range(start_epoch, total_epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        t0 = time.time()
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = train_X[idx].to(device)
            yb = train_y[idx].to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)

        train_loss = total_loss / n
        val_loss = evaluate(model, val_X, val_y)

        print(f"Epoch {epoch+1}/{total_epochs} ({time.time()-t0:.1f}s) "
              f"train_loss={train_loss:.4e} val_loss={val_loss:.4e}")

        log_history("results/lens_cv_loss_history.csv", {
            "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
        })

        if val_loss < best_vloss:
            best_vloss = val_loss
            fname = f"{save_dir}/lens_cv_model-{epoch+1:04d}-vloss-{val_loss:.4e}.pth"
            torch.save(model, fname)
            print(f"  Saved new best model: {fname}")


if __name__ == "__main__":
    DATA_DIR = "prepped_data/lens_cv"
    train_X = torch.from_numpy(np.load(os.path.join(DATA_DIR, "train_X.npy"))).to(torch.complex64)
    train_y = torch.from_numpy(np.load(os.path.join(DATA_DIR, "train_y.npy"))).to(torch.float32)
    val_X = torch.from_numpy(np.load(os.path.join(DATA_DIR, "val_X.npy"))).to(torch.complex64)
    val_y = torch.from_numpy(np.load(os.path.join(DATA_DIR, "val_y.npy"))).to(torch.float32)

    print("train_X shape:", train_X.shape, "val_X shape:", val_X.shape)

    SAVE_DIR = "saved_models/lens_cv"
    import glob, re
    existing = glob.glob(os.path.join(SAVE_DIR, "lens_cv_model-*.pth"))
    if existing:
        def epoch_from_name(p):
            m = re.search(r"model-(\d+)-", p)
            return int(m.group(1)) if m else -1
        def vloss_from_name(p):
            m = re.search(r"vloss-([0-9eE+\-.]+)\.pth$", p)
            return float(m.group(1)) if m else float("inf")
        latest = max(existing, key=epoch_from_name)
        print(f"Resuming from {latest}")
        model = torch.load(latest, map_location=device, weights_only=False)
        start_epoch = epoch_from_name(latest)
        best_vloss = vloss_from_name(latest)
    else:
        model = GravLensComplexCNN()
        start_epoch = 0
        best_vloss = float("inf")
    print("Model has", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters.")

    train_model(model, train_X, train_y, val_X, val_y, epochs=400, lr=0.001,
                save_dir=SAVE_DIR, start_epoch=start_epoch, best_vloss=best_vloss)
