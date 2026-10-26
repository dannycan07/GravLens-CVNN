import glob
import json
import os
import re
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from GravLensCV_harmonic_mlp import GravLensHarmonicMLP
from GravLensCV_cnn import ComplexLinear, CRELU, ComplexBatchNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def find_best_model(save_dir="saved_models/lens_cv_harmonic"):
    files = glob.glob(os.path.join(save_dir, "lens_cv_harmonic_model-*.pth"))
    if not files:
        raise FileNotFoundError(
            f"No trained models found in '{save_dir}/'. Run GravLensCV_harmonic_mlp.py first."
        )
    def loss_from_name(p):
        m = re.search(r"vloss-([0-9eE+\-.]+)\.pth$", p)
        return float(m.group(1)) if m else float("inf")
    return min(files, key=loss_from_name)


MODEL_PATH = os.environ.get("MODEL_PATH") or find_best_model()
print(f"Using model : {MODEL_PATH}")

model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model = model.to(device)
model.eval()

with open("prepped_data/lens_cv_harmonic/scaler.json") as f:
    scaler = json.load(f)
param_names = scaler["param_names"]
y_min = np.array(scaler["y_min"])
y_max = np.array(scaler["y_max"])

test_X = torch.from_numpy(np.load("prepped_data/lens_cv_harmonic/test_X.npy")).to(torch.complex64)
test_y_norm = np.load("prepped_data/lens_cv_harmonic/test_y.npy")
print("Test set shape:", test_X.shape)

with torch.no_grad():
    preds_norm = model(test_X.to(device)).cpu().numpy()

def denorm(y_norm):
    return y_norm * (y_max - y_min) + y_min

y_true = denorm(test_y_norm)
y_pred = denorm(preds_norm)

print("\nPer-parameter performance:")
r2_scores, rel_errors = [], []
for j, name in enumerate(param_names):
    yt, yp = y_true[:, j], y_pred[:, j]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    rel_err = np.median(np.abs((yp - yt) / yp)) * 100
    r2_scores.append(r2)
    rel_errors.append(rel_err)
    print(f"  {name:10s}: R2={r2:.4f}  median relative error={rel_err:.2f}%")

os.makedirs("results", exist_ok=True)
fig, axs = plt.subplots(1, len(param_names), figsize=(4.5 * len(param_names), 4.5))
for j, name in enumerate(param_names):
    ax = axs[j]
    yt, yp = y_true[:, j], y_pred[:, j]
    ax.scatter(yt, yp, s=4, alpha=0.4, color="tab:green")
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, "r-", linewidth=1)
    ax.set_xlabel("True value")
    ax.set_ylabel("Prediction")
    ax.set_title(f"{name}\nR2={r2_scores[j]:.3f}, err={rel_errors[j]:.1f}%")
plt.tight_layout()
out_path = "results/gravlens_cv_harmonic_scatter.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved scatter plot to {out_path}")
