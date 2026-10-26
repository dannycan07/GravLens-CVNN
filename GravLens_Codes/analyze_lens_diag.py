"""
analyze_lens_diag.py -- collect lens_train_diag.py run folders into tidy CSVs.

Run from the fire198 root:
  python cvnn/GravLens_Codes/analyze_lens_diag.py results/lens_diag/main
Writes into that folder:
  all_epochs.csv        one row per (run, epoch) with config + seed columns
  run_summary.csv       one row per run: best epoch, best-ckpt val/test metrics,
                        s/epoch, final true gap (val - eval-mode train) and
                        naive gap (val - train-mode train)
  matched_epoch.csv     metrics for every run at a COMMON epoch E* (the
                        smallest number of epochs any config completed), averaged
                        over the 10 epochs ending at E*, so configs that ran
                        longer get no advantage
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

PARAMS = ["theta_E", "f", "eps_x", "eps_y"]


def config_of(run):
    return re.sub(r"_s\d+$", "", run)


def main(folder):
    rows, sums = [], []
    for d in sorted(glob.glob(os.path.join(folder, "*", "history.csv"))):
        run = os.path.basename(os.path.dirname(d))
        h = pd.read_csv(d)
        h.insert(0, "run", run)
        h.insert(1, "config", config_of(run))
        h.insert(2, "seed", int(run.rsplit("_s", 1)[1]))
        rows.append(h)
        sp = os.path.join(os.path.dirname(d), "summary.json")
        if os.path.exists(sp):
            s = json.load(open(sp))
            last = h.tail(10).mean(numeric_only=True)
            s.update({"config": config_of(run), "final10_val_loss": last.val_loss,
                      "final10_train_loss_trainmode": last.train_loss,
                      "final10_train_loss_evalmode": last.train_loss_eval,
                      "final10_true_gap": last.gap_eval,
                      "final10_naive_gap": last.val_loss - last.train_loss,
                      "val_rel_jitter": float((h.val_loss.diff().abs() / h.val_loss).tail(30).mean())})
            sums.append(s)
    ep = pd.concat(rows, ignore_index=True)
    ep.to_csv(os.path.join(folder, "all_epochs.csv"), index=False)
    summ = pd.DataFrame(sums)
    summ.to_csv(os.path.join(folder, "run_summary.csv"), index=False)

    estar = int(ep.groupby("run").epoch.max().min())
    win = ep[ep.epoch.between(estar - 9, estar)]
    cols = ["train_loss", "train_loss_eval", "val_loss", "gap_eval"] + \
           [f"val_r2_{p}" for p in PARAMS] + [f"val_mae_{p}" for p in PARAMS]
    m = win.groupby(["config", "seed"])[cols].mean().reset_index()
    m.insert(2, "matched_epoch", estar)
    m.to_csv(os.path.join(folder, "matched_epoch.csv"), index=False)
    print(f"{len(summ)} finished runs; matched epoch E* = {estar}")
    return ep, summ, m, estar


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/lens_diag/main")
