#!/usr/bin/env python
"""Fit the regional residual field of a block to site covariates from the scored spatial probes.

Model: the residual on the block is r(site) x w(row) (w = the probes' row weight). Probe k measures
y_k = E[(T - B) D_k] = <r, a_k> with a_k(site) = sum over the site's block cells of w D_k / N_cells. With site covariates
X (climate, soil, profile, smooth modes), r = X beta is fitted by ridge to the y_k; leave-one-probe-out prediction of
y_k checks whether the covariates extrapolate. Writes the fitted direction r_hat(site) x w(row) as a probe candidate.

    ./.venv/bin/python scripts/regional_field_fit.py --block young --weight growth --base results/stack_20261010.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import spatial_correction  # noqa: E402,F401  (same readout conventions)

M = 1.14114
BLOCKS = {"oa": [("old", 1)], "young": [("young", 0), ("young", 1)], "mid": [("mid", 0), ("mid", 1)], "oh": [("old", 0)]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block", required=True, choices=sorted(BLOCKS))
    ap.add_argument("--weight", default="R", choices=["R", "growth"])
    ap.add_argument("--base", default="results/stack_20261010.csv")
    ap.add_argument("--prefix", default="", help="only probes whose name contains this (e.g. young_)")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    sc = json.load(open(ROOT / "results" / "board_scores.json"))
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    B = shift.read_predictions(ROOT / args.base).astype(np.float64)
    keys = shift.template_keys(); masks = shift.band_masks(keys["age"].to_numpy()); st = keys["site"].to_numpy()
    C = sc["zeros.csv"]; s1, s05 = sc["persistence_shifted.csv"], sc["persistence_s05.csv"]
    ps = shift.read_predictions(ROOT / "results" / "persistence_shifted.csv").astype(np.float64)
    rho = 2 * (s1 - 2 * s05 + C) / float(np.mean(ps ** 2))
    s_R = 0.25171  # anchors + mirror pair (scripts/spatial_correction.py)
    centers = json.loads((ROOT / "work" / "spat_centers.json").read_text())
    cells = np.zeros_like(R, dtype=bool)
    for band, col in BLOCKS[args.block]:
        cells[masks[band], col] = True
    if args.weight == "R":
        w = R.copy()
    else:
        w = np.abs(0.876 * np.load(ROOT / "work" / "ageshift46_dec.npy") - R)
    w = np.where(cells, w, 0.0)
    names = [n for n in sc if n.startswith("spat_") and args.prefix in n and not n.endswith("_mirror.csv")]
    A, y = [], []
    N = R.size
    for n in names:
        cen = centers.get(n, "R")
        Cp = R if cen == "R" else shift.read_predictions(ROOT / cen).astype(np.float64)
        D = shift.read_predictions(ROOT / "results" / n).astype(np.float64) - Cp
        if np.abs(D[cells]).sum() < 0.5 * np.abs(D).sum():
            continue  # the probe lives on another block
        e2 = rho * float(np.mean(D * D))
        if cen == "R":
            Xr = (sc[n] - s_R - e2) / 2
        else:
            Xr = (sc[n] - sc[Path(cen).name] - e2) / 2
        y.append(-Xr - rho * float(np.mean((B - Cp) * D)))
        a = np.zeros(6171); np.add.at(a, st, (w * D).sum(1) * rho / N)
        A.append(a)
    A = np.stack(A); y = np.array(y)
    # site covariates: smooth modes 1-12 plus climate/soil/profile descriptors (standardized)
    modes = np.load(ROOT / "work" / "site_graph_modes240.npy")[:, 1:13]
    t = np.load(ROOT / "work" / "cache" / "test_ssp126_annual.npy", mmap_mode="r")
    clim = np.c_[np.asarray(t[:, :, 0]).mean(1), np.log1p(np.asarray(t[:, :, 1]).mean(1)), np.asarray(t[:, :, 0]).std(1),
                 np.log1p(np.asarray(t[:, :, 1])).std(1), np.asarray(t[:, 0, 129:136])]
    y0 = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    prof = np.zeros((6171, 2)); cnt = np.zeros(6171)
    np.add.at(prof, st, y0); np.add.at(cnt, st, 1); prof /= cnt[:, None]
    X = np.c_[modes, clim, prof]
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    F = A @ X                       # (K probes, P covariates): y = F beta
    beta = np.linalg.solve(F.T @ F + args.ridge * np.eye(F.shape[1]) * np.trace(F.T @ F) / F.shape[1], F.T @ y)
    loo = []
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        b = np.linalg.solve(F[m].T @ F[m] + args.ridge * np.eye(F.shape[1]) * np.trace(F[m].T @ F[m]) / F.shape[1], F[m].T @ y[m])
        loo.append(F[i] @ b)
    loo = np.array(loo)
    print(f"{len(y)} probes on block {args.block}; measured y (x1e4) {np.round(y * 1e4, 2)}")
    print(f"leave-one-probe-out predictions (x1e4) {np.round(loo * 1e4, 2)}; corr {np.corrcoef(y, loo)[0, 1]:.3f}")
    r = X @ beta
    if args.out:
        D = np.where(cells, r[st][:, None] * w, 0.0)
        D *= np.sqrt(0.012 / max(np.mean(D ** 2), 1e-12))
        shift.write_predictions(np.clip(B + D, 0, None), ROOT / args.out)
        print(f"wrote {args.out} (base + fitted field x weight, E[D^2] 0.012)")


if __name__ == "__main__":
    main()
