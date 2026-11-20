#!/usr/bin/env python
"""Board probe directions from the ROW-level (state-dependent) ORNL forcing effect (scripts/ornl/ornl_state_force.py).

Same conventions as ornl_dirs.py (C-centred, natural scale, level-free per age x scenario, vegetated rows R > 0.05):
    qforce_<band>_<col>_<grp>   D = C x theta_col x (q(row) - block mean)      theta: agb 1, height --theta-h
    qcontrast_<col>_<grp>       D = +-C x theta_col x (q585 - q126 - mean) / 2  (+ on ssp585 rows, - on ssp126 rows)
bands old | ym | young | mid | all; col a | h; grp out | cl | all. Prints E_all and the correlation with the matching
site-level direction of ornl_dirs.py (work/dir_ornl_gfdl_<name>.npy) where it exists.
    python scripts/ornl/ornl_qdirs.py [--src state_force_GFDL-ESM4_CLIM_CO2_s1] [--tag q1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="state_force_GFDL-ESM4_CLIM_CO2_s1")
    ap.add_argument("--tag", default="q1")
    ap.add_argument("--theta-h", type=float, default=1.0, help="height response per unit log-agb response")
    ap.add_argument("--clip", type=float, default=0.6)
    ap.add_argument("--base", default="results/spatcorr_stack_20261106_s08.csv")
    args = ap.parse_args()
    z = np.load(ROOT / f"work/ornl2522/{args.src}.npz")
    q = np.clip(z["q"], -args.clip, args.clip)
    keys = shift.template_keys(); site = keys["site"].to_numpy(); age = keys["age"].to_numpy(); scen = keys["scenario"].to_numpy()
    band = shift.band_of(age)
    C = shift.read_predictions(ROOT / args.base).astype(float)
    R = shift.read_predictions(ROOT / "results/latent_ref_y0_m114114.csv").astype(float)
    cl = (np.load(ROOT / "work/collapse_cluster_labels.npy") == 0)[site]
    is585 = scen == "ssp585"
    # contrast per row: q585 - q126 of the same site x age
    idx = {(s, a, sc): i for i, (s, a, sc) in enumerate(zip(site, age, scen))}
    partner = np.array([idx[(s, a, "ssp126" if sc == "ssp585" else "ssp585")] for s, a, sc in zip(site, age, scen)])
    con = np.where(is585, q - q[partner], q[partner] - q)          # q585 - q126 on both rows
    okq = np.isfinite(q) & np.isfinite(con)
    theta = {0: args.theta_h, 1: 1.0}
    bands = {"old": band == "old", "ym": band != "old", "young": band == "young", "mid": band == "mid", "all": np.ones(len(keys), bool)}
    groups = {"out": ~cl, "cl": cl, "all": np.ones(len(keys), bool)}
    for bname, bmask in bands.items():
        for col, cname in ((1, "a"), (0, "h")):
            for gname, gmask in groups.items():
                rows = bmask & gmask & okq & (R[:, col] > 0.05)
                D = np.zeros_like(C)
                for a in np.unique(age):
                    for s in ("ssp126", "ssp585"):
                        m = rows & (age == a) & (scen == s)
                        if m.any():
                            x = C[m, col] * theta[col] * q[m]
                            D[m, col] = x - x.mean()
                name = f"qforce_{bname}_{cname}_{gname}"
                np.save(ROOT / f"work/dir_ornl_{args.tag}_{name}.npy", D)
                ref = ROOT / f"work/dir_ornl_gfdl_force_{bname}_{cname}_{gname}.npy"
                cc = ""
                if ref.exists():
                    Dr = np.load(ref); cc = f"  corr with site-level {np.sum(D * Dr) / np.sqrt(np.sum(D ** 2) * np.sum(Dr ** 2) + 1e-30):+.3f} (E {np.mean(Dr ** 2):.5f})"
                print(f"{name:22s} rows {int(rows.sum()):6d}  E_all {np.mean(D ** 2):.5f}{cc}")
    for col, cname in ((1, "a"), (0, "h")):
        for gname, gmask in groups.items():
            rows = gmask & okq & (R[:, col] > 0.05)
            D = np.zeros_like(C)
            for a in np.unique(age):
                m = rows & (age == a)
                if m.any():
                    x = C[m, col] * theta[col] * con[m] / 2
                    mu = x[is585[m]].mean() if (is585[m]).any() else 0.0
                    D[m, col] = np.where(is585[m], x - mu, -(x - mu))
            name = f"qcontrast_{cname}_{gname}"
            np.save(ROOT / f"work/dir_ornl_{args.tag}_{name}.npy", D)
            ref = ROOT / f"work/dir_ornl_gfdl_contrast_{cname}_{gname}.npy"
            cc = ""
            if ref.exists():
                Dr = np.load(ref); cc = f"  corr with site-level {np.sum(D * Dr) / np.sqrt(np.sum(D ** 2) * np.sum(Dr ** 2) + 1e-30):+.3f}"
            print(f"{name:22s} rows {int(rows.sum()):6d}  E_all {np.mean(D ** 2):.5f}{cc}")


if __name__ == "__main__":
    main()
