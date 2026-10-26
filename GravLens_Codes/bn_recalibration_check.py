"""
bn_recalibration_check.py -- is the CVNN's train/val gap (and its noisy val
curve) caused by BatchNorm running statistics?

For each finished run folder (from lens_train_diag.py) with BatchNorm:
  1. evaluate best.pt as saved (eval mode, running mean/variance from training)
  2. re-estimate the BN running statistics as an exact average over the whole
     training set (momentum=None, train-mode forward passes, no gradient step)
     and evaluate again
  3. report train-set (eval mode), val and test metrics before and after
     (test is untouched by checkpoint selection -> use it for conclusions)

If step 2 lowers val MSE a lot, the exponential-moving-average BN statistics
were stale/noisy -- a measurement/BN problem, not memorisation.

Run from the fire198 root:
  python cvnn/GravLens_Codes/bn_recalibration_check.py results/lens_diag/main/cv_base_s0 ...
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lens_train_diag as L  # noqa: E402


def build(cfg):
    w = tuple(cfg["widths"])
    if cfg["model"] == "cv_fft":
        cls = L.PackedLensCVNN if cfg.get("impl", "packed") == "packed" else L.LensCVNN
        return cls(w, cfg["fc"], bn=cfg["bn"], sdrop=cfg["sdrop"])
    return L.LensRVNN(1 if cfg["model"] == "rv_pix" else 2, w, cfg["fc"], bn=cfg["bn"], sdrop=cfg["sdrop"])


@torch.no_grad()
def recalibrate(model, X, bs=256):
    bns = [m for m in model.modules() if isinstance(m, (L.ComplexBatchNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
    for m in bns:
        if not isinstance(m, L.ComplexBatchNorm):
            m.momentum = None  # torch BN: cumulative average over all batches
    # ComplexBatchNorm has no cumulative mode -> accumulate exact averages via
    # hooks; its own momentum updates during this pass are overwritten below.
    sums = {}

    def hook(mod, inp, out):
        x = inp[0]
        real, imag = x.chunk(2, dim=1) if not torch.is_complex(x) else (x.real, x.imag)
        dims, _ = mod._dims_and_shape()
        n = real.numel() // real.shape[1]
        mr, mi = real.mean(dims), imag.mean(dims)
        s = sums.setdefault(mod, [0, 0, 0, 0, 0, 0])
        s[0] += n; s[1] += mr * n; s[2] += mi * n
        s[3] += (real ** 2).mean(dims) * n; s[4] += (imag ** 2).mean(dims) * n; s[5] += (real * imag).mean(dims) * n

    hs = [m.register_forward_hook(hook) for m in bns if isinstance(m, L.ComplexBatchNorm)]
    for m in bns:
        if not isinstance(m, L.ComplexBatchNorm):
            m.reset_running_stats()
    model.train()
    for m in model.modules():  # keep dropout OFF during recalibration
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, L.ComplexSpatialDropout2d)):
            m.eval()
    # NOTE: deeper BN layers see inputs normalised by the *batch* statistics of
    # earlier layers during this pass (standard practice, e.g. SWA's bn_update).
    for i in range(0, len(X), bs):
        model(X[i:i + bs])
    for h in hs:
        h.remove()
    for m, s in sums.items():
        n = s[0]
        mr, mi = s[1] / n, s[2] / n
        m.running_mean_real.copy_(mr); m.running_mean_imag.copy_(mi)
        m.running_Vrr.copy_(s[3] / n - mr ** 2 + m.eps)
        m.running_Vii.copy_(s[4] / n - mi ** 2 + m.eps)
        m.running_Vri.copy_(s[5] / n - mr * mi)
    model.eval()


def main(run_dirs):
    rows = []
    for d in run_dirs:
        cfg = json.load(open(os.path.join(d, "config.json")))
        if not cfg["bn"]:
            continue
        torch.set_num_threads(4)
        sc = json.load(open(os.path.join(cfg["root"], "prepped_data/lens_scaler.json")))
        yr = np.array(sc["y_max"]) - np.array(sc["y_min"])
        impl = cfg.get("impl", "packed")
        trX, trY = L.load_split(cfg["root"], cfg["model"], "train", impl)
        vaX, vaY = L.load_split(cfg["root"], cfg["model"], "val", impl)
        teX, teY = L.load_split(cfg["root"], cfg["model"], "test", impl)
        model = build(cfg)
        model.load_state_dict(torch.load(os.path.join(d, "best.pt")))
        before_tr = L.metrics(L.predict(model, trX), trY, yr)["mse"]
        before_va = L.metrics(L.predict(model, vaX), vaY, yr)
        before_te = L.metrics(L.predict(model, teX), teY, yr)
        recalibrate(model, trX)
        after_tr = L.metrics(L.predict(model, trX), trY, yr)["mse"]
        after_va = L.metrics(L.predict(model, vaX), vaY, yr)
        after_te = L.metrics(L.predict(model, teX), teY, yr)
        rows.append({"run": os.path.basename(d), "train_mse_eval_before": before_tr, "val_mse_before": before_va["mse"],
                     "train_mse_eval_after": after_tr, "val_mse_after": after_va["mse"],
                     **{f"val_r2_{p}_before": before_va[f"r2_{p}"] for p in L.PARAM_NAMES},
                     **{f"val_r2_{p}_after": after_va[f"r2_{p}"] for p in L.PARAM_NAMES},
                     "test_mse_before": before_te["mse"], "test_mse_after": after_te["mse"],
                     **{f"test_{k}_{p}_{w}": t[f"{k}_{p}"] for w, t in [("before", before_te), ("after", after_te)]
                        for k in ("r2", "mae") for p in L.PARAM_NAMES}})
        print(json.dumps(rows[-1]), flush=True)
    return rows


if __name__ == "__main__":
    main(sys.argv[1:])
