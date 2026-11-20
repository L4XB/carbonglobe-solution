#!/usr/bin/env python
"""Board probe directions from the ORNL DAAC 2522 regional future-forcing response of existing forests.

Per test site (scripts/ornl/ornl_sites.py): force_<ssp> = log(B2060/B2021)_ssp - log(B2060/B2021)_REF (the effect of
the future forcing over the window years) and contrast = log r_585 - log r_126. --esm takes one ESM or a comma list
(their responses are averaged). Directions are C-centred, natural scale, LEVEL-FREE (block means over the direction's
rows per age x scenario x column removed), vegetated rows only (R > 0.05), sites with forest weight > 0.05:
    force_<band>_<col>_<grp>   D = C x (f_ssp(site) - block mean)          bands old | ym | all; col a | h; grp out | cl | all
    contrast_<col>_<grp>       D = +-C x (contrast(site) - mean) / 2       (+ on ssp585 rows, - on ssp126 rows)
Writes work/dir_ornl_<name>.npy and prints E_all per direction.
    python scripts/ornl/ornl_dirs.py --esm GFDL-ESM4 [--case CLIM_CO2] [--tag gfdl]
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
    ap.add_argument("--esm", default="GFDL-ESM4")
    ap.add_argument("--case", default="CLIM_CO2")
    ap.add_argument("--tag", default="")
    ap.add_argument("--clip", type=float, default=0.5, help="clip |f| (log units) to limit outliers")
    ap.add_argument("--resp-suffix", default="", help="site_response_<esm>_<case><suffix>.npz (e.g. _nx2s05)")
    args = ap.parse_args()
    zs = [np.load(ROOT / f"work/ornl2522/site_response_{e}_{args.case}{args.resp_suffix}.npz") for e in args.esm.split(",")]
    f126 = np.nanmean([z["force_126"] for z in zs], axis=0); f585 = np.nanmean([z["force_585"] for z in zs], axis=0)
    con = np.nanmean([z["contrast"] for z in zs], axis=0); w = np.nanmean([z["w_ref"] for z in zs], axis=0)
    oks = (w > 0.05) & np.isfinite(f126) & np.isfinite(f585) & np.isfinite(con)
    f126, f585, con = (np.clip(np.nan_to_num(v), -args.clip, args.clip) for v in (f126, f585, con))
    keys = shift.template_keys(); site = keys["site"].to_numpy(); age = keys["age"].to_numpy(); scen = keys["scenario"].to_numpy()
    band = shift.band_of(age)
    C = shift.read_predictions(ROOT / "results/spatcorr_stack_20261106_s08.csv").astype(float)
    R = shift.read_predictions(ROOT / "results/latent_ref_y0_m114114.csv").astype(float)
    cl = (np.load(ROOT / "work/collapse_cluster_labels.npy") == 0)[site]
    is585 = scen == "ssp585"
    f_row = np.where(is585, f585[site], f126[site])
    c_row = np.where(is585, 0.5, -0.5) * con[site]
    tag = f"_{args.tag}" if args.tag else ""

    def levelfree(X: np.ndarray, rows: np.ndarray, col: int) -> np.ndarray:
        out = np.zeros_like(X)
        for a in np.unique(age):
            for s in ("ssp126", "ssp585"):
                q = rows & (age == a) & (scen == s)
                if q.any():
                    out[q, col] = X[q, col] - X[q, col].mean()
        return out

    specs = []
    for bname, bmask in (("old", band == "old"), ("ym", band != "old"), ("all", np.ones(len(keys), bool))):
        for col, cname in ((1, "a"), (0, "h")):
            for gname, gmask in (("out", ~cl), ("cl", cl), ("all", np.ones(len(keys), bool))):
                specs.append((f"force_{bname}_{cname}_{gname}", bmask & gmask, col, "force"))
    for col, cname in ((1, "a"), (0, "h")):
        for gname, gmask in (("out", ~cl), ("cl", cl), ("all", np.ones(len(keys), bool))):
            specs.append((f"contrast_{cname}_{gname}", gmask, col, "contrast"))
    for name, mask, col, kind in specs:
        rows = mask & oks[site] & (R[:, col] > 0.05)
        X = np.zeros_like(C)
        X[rows, col] = C[rows, col] * (f_row[rows] if kind == "force" else c_row[rows])
        D = levelfree(X, rows, col) if kind == "force" else X - np.where(rows[:, None], 0.0, 0.0)
        if kind == "contrast":                         # remove the mean contrast per age block (the stack's mean contrast is board-pinned)
            D = np.zeros_like(C)
            for a in np.unique(age):
                q = rows & (age == a)
                if q.any():
                    m = np.mean(X[q & is585, col]) if (q & is585).any() else 0.0
                    D[q & is585, col] = X[q & is585, col] - m
                    D[q & ~is585, col] = X[q & ~is585, col] + m
        np.save(ROOT / f"work/dir_ornl{tag}_{name}.npy", D)
        print(f"{name:22s} rows {int((D[:, col] != 0).sum()):6d}  E_all {np.mean(D ** 2):.5f}  cluster share {np.sum(D[cl] ** 2) / max(np.sum(D ** 2), 1e-30):.2f}")


if __name__ == "__main__":
    main()
