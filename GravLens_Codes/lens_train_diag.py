"""
lens_train_diag.py -- instrumented, seed-controlled training for the
gravitational-lensing CVNN-vs-RVNN comparison.

This is a NEW file; GravLensCV_cnn.py and GravLensRV_cnn.py are untouched.
The complex layers are imported from GravLensCV_cnn.py so the CVNN maths is
identical to the original script.

What it adds over the original scripts
  * --seed               : reproducible runs (torch / numpy / python RNGs)
  * unique run folder    : results/lens_diag/<run_name>/ (no more overwriting)
  * per-epoch log        : train loss (train mode, running average, as before),
                           train loss in EVAL mode on a fixed train subset
                           (same measurement as val -> an honest gap),
                           val loss, per-parameter val R^2 and MAE, grad norm,
                           learning rate, seconds per epoch
  * best-only checkpoint : one best.pt (state_dict) per run, overwritten
  * --sdrop p            : spatial dropout (drops whole feature maps).
                           Complex version applies ONE mask to the real and the
                           imaginary part, so a complex channel is dropped as a unit.
  * --model rv_fft2ch    : real CNN fed the SAME log-magnitude FFT as the CVNN,
                           as 2 real channels (Re, Im). Separates "FFT input"
                           from "complex arithmetic".
  * --bn                 : BatchNorm on/off for every model (the original RVNN
                           had none, the original CVNN had complex BN -> confound)
  * --wd, --sched, --n-train, --max-minutes : regularisation / schedule /
                           data-size / wall-clock controls

Run from the fire198 root, e.g.
  python cvnn/GravLens_Codes/lens_train_diag.py --model cv_fft --bn 1 --seed 0 --epochs 300
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "GravLens_CVNN"))
from GravLensCV_cnn import (ComplexConv2d, ComplexLinear, CRELU,  # noqa: E402
                            ComplexAvgPool2d, ComplexAdaptiveAvgPool2d,
                            ComplexBatchNorm)

PARAM_NAMES = ["theta_E", "f", "eps_x", "eps_y"]
NUM_PARAMS = 4


# --------------------------------------------------------------------------
# Spatial dropout
# --------------------------------------------------------------------------
class ComplexSpatialDropout2d(nn.Module):
    """Zero entire complex feature maps with probability p (train mode only).

    Ordinary dropout zeros single activations; neighbouring pixels in a conv
    feature map are strongly correlated, so the network can 'fill in' a
    dropped pixel from its neighbours and ordinary dropout barely regularises.
    Spatial dropout (Tompson et al. 2015) zeros a whole channel at once.
    For a complex channel we draw one mask and apply it to Re and Im together
    (dropping only Re would rotate the phase, which is not 'removing' it).
    Survivors are scaled by 1/(1-p) so the expected activation is unchanged.
    """

    def __init__(self, p):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        keep = (torch.rand(x.shape[0], x.shape[1], 1, 1, device=x.real.device) > self.p)
        mask = keep.to(x.real.dtype) / (1.0 - self.p)
        return torch.complex(x.real * mask, x.imag * mask)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class LensCVNN(nn.Module):
    """Same layer schedule as GravLensCV_cnn.GravLensComplexCNN, with BN and
    spatial dropout as switches. bn=True, sdrop=0, widths=(16,32,64) reproduces
    the original model exactly."""

    def __init__(self, widths=(16, 32, 64), fc=64, bn=True, sdrop=0.0):
        super().__init__()
        layers, cin = [], 1
        for w in widths:
            layers.append(ComplexConv2d(cin, w, kernel_size=3, padding=1))
            if bn:
                layers.append(ComplexBatchNorm(w, spatial=True))
            layers.append(CRELU())
            if sdrop > 0:
                layers.append(ComplexSpatialDropout2d(sdrop))
            layers.append(ComplexAvgPool2d(2))
            cin = w
        self.features = nn.Sequential(*layers)
        self.pool = ComplexAdaptiveAvgPool2d(1)
        self.fc1 = ComplexLinear(cin, fc)
        self.bn_fc1 = ComplexBatchNorm(fc, spatial=False) if bn else nn.Identity()
        self.act = CRELU()
        self.fc2 = ComplexLinear(fc, NUM_PARAMS)
        self.readout = nn.Linear(NUM_PARAMS * 2, NUM_PARAMS)  # Re/Im split, NOT .abs()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.act(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x)
        return torch.sigmoid(self.readout(torch.cat([x.real, x.imag], dim=-1)))


class LensRVNN(nn.Module):
    """Same layer schedule as GravLensRV_cnn.GravLensRealCNN. in_ch=1, bn=False,
    sdrop=0, widths=(16,32,64) reproduces the original RVNN exactly."""

    def __init__(self, in_ch=1, widths=(16, 32, 64), fc=64, bn=False, sdrop=0.0):
        super().__init__()
        layers, cin = [], in_ch
        for w in widths:
            layers.append(nn.Conv2d(cin, w, kernel_size=3, padding=1))
            if bn:
                layers.append(nn.BatchNorm2d(w))
            layers.append(nn.ReLU())
            if sdrop > 0:
                layers.append(nn.Dropout2d(sdrop))
            layers.append(nn.AvgPool2d(2))
            cin = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        head = [nn.Linear(cin, fc)]
        if bn:
            head.append(nn.BatchNorm1d(fc))
        head += [nn.ReLU(), nn.Linear(fc, NUM_PARAMS), nn.Sigmoid()]
        self.regressor = nn.Sequential(*head)

    def forward(self, x):
        x = self.features(x)
        return self.regressor(self.pool(x).flatten(1))


# --------------------------------------------------------------------------
# "Packed" complex layers: SAME maths, measured ~2x faster on CPU
# (462 vs 925 ms per 64-image fwd+bwd batch, 4 threads, 2026-10-19)
# --------------------------------------------------------------------------
# A complex tensor z = a + i b with C channels is stored as ONE real tensor with
# 2C channels: [a_1..a_C, b_1..b_C]. A complex conv with weight W = Wr + i Wi,
#     Re(out) = Wr*a - Wi*b + br,   Im(out) = Wi*a + Wr*b + bi,
# is then a single ordinary conv with the real block weight
#     [[Wr, -Wi],
#      [Wi,  Wr]].
# Parameter names/shapes are identical to the original complex layers, so
# state_dicts are interchangeable. Checked against GravLensComplexCNN with
# shared weights on 64 training images: max |difference| 2e-7 (outputs),
# 1e-6 (gradients), 3e-8 (BN buffers) -- float32 rounding only. CReLU and average pooling act on Re
# and Im separately, so on the packed tensor they are just F.relu / avg_pool.
class PackedComplexConv2d(ComplexConv2d):
    def forward(self, x):
        w = torch.cat([torch.cat([self.weight_real, -self.weight_imag], 1),
                       torch.cat([self.weight_imag, self.weight_real], 1)], 0)
        b = torch.cat([self.bias_real, self.bias_imag])
        return F.conv2d(x, w, b, padding=self.padding)


class PackedComplexLinear(ComplexLinear):
    def forward(self, x):
        w = torch.cat([torch.cat([self.weight_real, -self.weight_imag], 1),
                       torch.cat([self.weight_imag, self.weight_real], 1)], 0)
        b = torch.cat([self.bias_real, self.bias_imag])
        return F.linear(x, w, b)


class PackedComplexBatchNorm(ComplexBatchNorm):
    """Identical whitening maths to ComplexBatchNorm, on the packed layout."""

    def forward(self, x):
        real, imag = x.chunk(2, dim=1)
        dims, shape = self._dims_and_shape()
        if self.training:
            mean_r, mean_i = real.mean(dim=dims), imag.mean(dim=dims)
            xc, yc = real - mean_r.view(*shape), imag - mean_i.view(*shape)
            Vrr = (xc ** 2).mean(dim=dims) + self.eps
            Vii = (yc ** 2).mean(dim=dims) + self.eps
            Vri = (xc * yc).mean(dim=dims)
            with torch.no_grad():
                m = self.momentum
                self.running_mean_real.mul_(1 - m).add_(m * mean_r)
                self.running_mean_imag.mul_(1 - m).add_(m * mean_i)
                self.running_Vrr.mul_(1 - m).add_(m * Vrr)
                self.running_Vii.mul_(1 - m).add_(m * Vii)
                self.running_Vri.mul_(1 - m).add_(m * Vri)
        else:
            Vrr, Vii, Vri = self.running_Vrr, self.running_Vii, self.running_Vri
            xc = real - self.running_mean_real.view(*shape)
            yc = imag - self.running_mean_imag.view(*shape)
        tau, delta = Vrr + Vii, Vrr * Vii - Vri ** 2
        s = torch.sqrt(torch.clamp(delta, min=self.eps))
        t = torch.sqrt(torch.clamp(tau + 2 * s, min=self.eps))
        rst = 1.0 / (s * t)
        Wrr, Wii, Wri = ((Vii + s) * rst).view(*shape), ((Vrr + s) * rst).view(*shape), (-Vri * rst).view(*shape)
        x_hat, y_hat = Wrr * xc + Wri * yc, Wri * xc + Wii * yc
        g_rr, g_ii, g_ri = self.gamma_rr.view(*shape), self.gamma_ii.view(*shape), self.gamma_ri.view(*shape)
        out_r = g_rr * x_hat + g_ri * y_hat + self.beta_real.view(*shape)
        out_i = g_ri * x_hat + g_ii * y_hat + self.beta_imag.view(*shape)
        return torch.cat([out_r, out_i], dim=1)


class PackedComplexSpatialDropout2d(ComplexSpatialDropout2d):
    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        C = x.shape[1] // 2
        keep = (torch.rand(x.shape[0], C, 1, 1, device=x.device) > self.p).to(x.dtype) / (1.0 - self.p)
        return x * keep.repeat(1, 2, 1, 1)  # same mask for Re and Im


class PackedLensCVNN(nn.Module):
    """Packed-layout twin of LensCVNN (same parameter names -> same state_dict)."""

    def __init__(self, widths=(16, 32, 64), fc=64, bn=True, sdrop=0.0):
        super().__init__()
        layers, cin = [], 1
        for w in widths:
            layers.append(PackedComplexConv2d(cin, w, kernel_size=3, padding=1))
            if bn:
                layers.append(PackedComplexBatchNorm(w, spatial=True))
            layers.append(nn.ReLU())                 # == CRELU on packed layout
            if sdrop > 0:
                layers.append(PackedComplexSpatialDropout2d(sdrop))
            layers.append(nn.AvgPool2d(2))           # == ComplexAvgPool2d
            cin = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = PackedComplexLinear(cin, fc)
        self.bn_fc1 = PackedComplexBatchNorm(fc, spatial=False) if bn else nn.Identity()
        self.act = nn.ReLU()
        self.fc2 = PackedComplexLinear(fc, NUM_PARAMS)
        self.readout = nn.Linear(NUM_PARAMS * 2, NUM_PARAMS)  # input order [Re, Im] as original

    def forward(self, x):  # x: (N, 2, H, W) = [Re, Im] of the FFT
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.act(self.bn_fc1(self.fc1(x)))
        return torch.sigmoid(self.readout(self.fc2(x)))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_split(root, model, name, impl="packed"):
    if model == "rv_pix":
        X = np.load(os.path.join(root, "prepped_data/lens_rv", f"{name}_X.npy")).astype(np.float32)
        X = torch.from_numpy(X)
    else:
        Xc = np.load(os.path.join(root, "prepped_data/lens_cv", f"{name}_X.npy"))
        if model == "cv_fft" and impl == "complex":
            X = torch.from_numpy(Xc.astype(np.complex64))
        else:  # rv_fft2ch, or packed CVNN: identical FFT input as [Re, Im] channels
            X = torch.from_numpy(np.concatenate([Xc.real, Xc.imag], axis=1).astype(np.float32))
    # labels are identical across lens_rv / lens_cv (same permutation in data_prep_lens.py)
    y = np.load(os.path.join(root, "prepped_data/lens_rv", f"{name}_y.npy")).astype(np.float32)
    return X, torch.from_numpy(y)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
@torch.no_grad()
def predict(model, X, batch_size=256):
    model.eval()
    out = [model(X[i:i + batch_size]) for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy()


def metrics(pred, y, y_range):
    """MSE (normalised units, mean over all 4 params) + per-param R^2 and MAE.
    R^2 is unchanged by the min-max scaling; MAE is reported in physical units."""
    y = y.numpy() if torch.is_tensor(y) else y
    res = {"mse": float(np.mean((pred - y) ** 2))}
    for j, p in enumerate(PARAM_NAMES):
        ss_res = np.sum((pred[:, j] - y[:, j]) ** 2)
        ss_tot = np.sum((y[:, j] - y[:, j].mean()) ** 2)
        res[f"r2_{p}"] = float(1 - ss_res / ss_tot)
        res[f"mae_{p}"] = float(np.mean(np.abs(pred[:, j] - y[:, j])) * y_range[j])
    return res


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cv_fft", "rv_pix", "rv_fft2ch"], required=True)
    ap.add_argument("--bn", type=int, default=None, help="1/0; default = original (cv:1, rv:0)")
    ap.add_argument("--sdrop", type=float, default=0.0, help="spatial dropout prob per conv block")
    ap.add_argument("--widths", type=str, default="16,32,64")
    ap.add_argument("--fc", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0, help=">0 switches Adam -> AdamW")
    ap.add_argument("--sched", choices=["none", "cosine", "plateau"], default="none")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=0, help="subsample training set (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-minutes", type=float, default=0, help="stop after this wall time (0 = off)")
    ap.add_argument("--root", type=str, default=".", help="fire198 root")
    ap.add_argument("--out", type=str, default="results/lens_diag")
    ap.add_argument("--run-name", type=str, default="")
    ap.add_argument("--no-test", action="store_true", help="skip final test-set evaluation")
    ap.add_argument("--impl", choices=["packed", "complex"], default="packed",
                    help="CVNN implementation: packed (fast, same maths) or complex (original layers)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    bn = bool(args.bn) if args.bn is not None else (args.model == "cv_fft")
    widths = tuple(int(w) for w in args.widths.split(","))

    tag = f"{args.model}{'' if args.impl == 'packed' or args.model != 'cv_fft' else 'C'}_bn{int(bn)}_sd{args.sdrop:g}_w{widths[0]}_lr{args.lr:g}"
    if args.wd > 0:
        tag += f"_wd{args.wd:g}"
    if args.sched != "none":
        tag += f"_{args.sched}"
    if args.n_train:
        tag += f"_n{args.n_train}"
    run_name = args.run_name or f"{tag}_s{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = os.path.join(args.root, args.out, run_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(args.root, "prepped_data/lens_scaler.json")) as f:
        sc = json.load(f)
    y_range = np.array(sc["y_max"]) - np.array(sc["y_min"])

    trX, trY = load_split(args.root, args.model, "train", args.impl)
    vaX, vaY = load_split(args.root, args.model, "val", args.impl)
    if args.n_train:
        g = torch.Generator().manual_seed(1234)  # same subset for every seed/model
        keep = torch.randperm(len(trX), generator=g)[:args.n_train]
        trX, trY = trX[keep], trY[keep]
    sub = torch.arange(min(900, len(trX)))  # fixed train subset for eval-mode train loss
    trX_sub, trY_sub = trX[sub], trY[sub]

    if args.model == "cv_fft":
        cls = PackedLensCVNN if args.impl == "packed" else LensCVNN
        model = cls(widths, args.fc, bn=bn, sdrop=args.sdrop)
    else:
        model = LensRVNN(in_ch=1 if args.model == "rv_pix" else 2, widths=widths,
                         fc=args.fc, bn=bn, sdrop=args.sdrop)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.wd > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.sched == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    elif args.sched == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=15)
    else:
        sched = None
    crit = nn.MSELoss()

    cfg = dict(vars(args), bn=bn, widths=list(widths), n_params=n_params, run_name=run_name,
               n_train_used=len(trX), torch_version=torch.__version__)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[{run_name}] params={n_params} n_train={len(trX)}", flush=True)

    hist_path = os.path.join(out_dir, "history.csv")
    hist_f = open(hist_path, "w", newline="")
    writer = None
    best_v, best_ep = float("inf"), 0
    t_start = time.time()
    n = len(trX)
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n)
        tot, gn_sum, nb = 0.0, 0.0, 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = crit(model(trX[idx]), trY[idx])
            loss.backward()
            gn_sum += float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")))
            opt.step()
            tot += loss.item() * len(idx)
            nb += 1
        t_train = time.time() - t0
        train_loss = tot / n
        tr_eval = metrics(predict(model, trX_sub), trY_sub, y_range)
        va = metrics(predict(model, vaX), vaY, y_range)
        lr_now = opt.param_groups[0]["lr"]
        if sched is not None:
            sched.step(va["mse"]) if args.sched == "plateau" else sched.step()

        row = {"epoch": ep, "train_loss": train_loss, "train_loss_eval": tr_eval["mse"],
               "val_loss": va["mse"], "gap_eval": va["mse"] - tr_eval["mse"],
               "grad_norm": gn_sum / nb, "lr": lr_now,
               "sec_train": t_train, "sec_epoch": time.time() - t0}
        row.update({f"val_{k}": v for k, v in va.items() if k != "mse"})
        row.update({f"train_r2_{p}": tr_eval[f"r2_{p}"] for p in PARAM_NAMES})
        if writer is None:
            writer = csv.DictWriter(hist_f, fieldnames=list(row.keys()))
            writer.writeheader()
        writer.writerow(row)
        hist_f.flush()

        if va["mse"] < best_v:
            best_v, best_ep = va["mse"], ep
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
        if ep == 1 or ep % 10 == 0:
            print(f"[{run_name}] ep {ep} {row['sec_epoch']:.1f}s train {train_loss:.3e} "
                  f"train_eval {tr_eval['mse']:.3e} val {va['mse']:.3e}", flush=True)
        if args.max_minutes and (time.time() - t_start) > 60 * args.max_minutes:
            print(f"[{run_name}] wall-clock limit reached at epoch {ep}", flush=True)
            break
    hist_f.close()
    total_s = time.time() - t_start

    # Final evaluation of the best-val checkpoint
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt")))
    summary = {"run_name": run_name, "model": args.model, "bn": bn, "sdrop": args.sdrop,
               "seed": args.seed, "lr": args.lr, "wd": args.wd, "sched": args.sched,
               "n_params": n_params, "n_train": len(trX), "epochs_run": ep,
               "best_epoch": best_ep, "total_seconds": total_s, "sec_per_epoch": total_s / ep}
    summary.update({f"val_{k}": v for k, v in metrics(predict(model, vaX), vaY, y_range).items()})
    if not args.no_test:
        teX, teY = load_split(args.root, args.model, "test", args.impl)
        pred = predict(model, teX)
        summary.update({f"test_{k}": v for k, v in metrics(pred, teY, y_range).items()})
        np.savez(os.path.join(out_dir, "test_predictions.npz"), pred=pred, y=teY.numpy())
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{run_name}] done in {total_s/60:.1f} min, best ep {best_ep}, val mse {best_v:.3e}", flush=True)


if __name__ == "__main__":
    main()
