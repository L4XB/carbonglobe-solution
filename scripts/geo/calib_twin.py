#!/usr/bin/env python
"""Per-channel affine map of the released test drivers against the SAME CELL's training drivers (exact twins).

For every channel c: x = training monthly climatology of the cell over training years --train-years (default the last
10 = 2012-2021), y = released test monthly climatology of the site over test years --test-years (default the first 10
= 2021-2030), both scenarios averaged. OLS fit y = alpha_c * x + beta_c over sites x 12 months, R2, the robust residual
spread, and the per-site offset spread (sd of the site-mean residual). Writes work/cal/calib_twin.npz
(alpha, beta, r2, site_sd, plus the climatologies clim_test (n, 12, 136) and clim_train (n, 12, 136)).
    python scripts/geo/calib_twin.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-years", default="30:40")
    ap.add_argument("--test-years", default="0:10")
    ap.add_argument("--out", default="work/cal/calib_twin.npz")
    args = ap.parse_args()
    t0, t1 = (int(v) for v in args.train_years.split(":")); s0, s1 = (int(v) for v in args.test_years.split(":"))
    ex = np.load(ROOT / "work/test_cell_exact.npz"); ok = ex["cell"] >= 0
    sites = np.where(ok)[0]; cells = ex["cell"][ok]
    order = np.argsort(cells); inv = np.empty_like(order); inv[order] = np.arange(len(order))
    xtr = io.open_train_x()
    ctr = np.empty((len(sites), 12, 136), np.float64)
    for lo in range(0, len(sites), 500):
        idx = cells[order][lo:lo + 500]
        ctr[lo:lo + 500] = np.asarray(xtr[idx, t0:t1], dtype=np.float64).mean(axis=1)
    ctr = ctr[inv]
    cte = np.zeros((len(sites), 12, 136), np.float64)
    for s in io.SCENARIOS:
        xt, _ = io.open_test(s)
        for lo in range(0, len(sites), 500):
            cte[lo:lo + 500] += np.asarray(xt[sites[lo:lo + 500], s0:s1], dtype=np.float64).mean(axis=1) / 2
    alpha = np.ones(136); beta = np.zeros(136); r2 = np.zeros(136); site_sd = np.zeros(136); mad = np.zeros(136)
    for c in range(136):
        x = ctr[:, :, c].ravel(); y = cte[:, :, c].ravel()
        xm, ym = x.mean(), y.mean()
        if x.std() < 1e-12:
            beta[c] = ym - xm; r2[c] = np.nan; continue
        a = np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2); b = ym - a * xm
        r = y - (a * x + b)
        alpha[c], beta[c] = a, b; r2[c] = 1 - r.var() / y.var(); mad[c] = np.median(np.abs(r))
        site_sd[c] = r.reshape(len(sites), 12).mean(1).std()
    np.savez(ROOT / args.out, alpha=alpha, beta=beta, r2=r2, site_sd=site_sd, mad=mad, clim_test=cte.astype(np.float32), clim_train=ctr.astype(np.float32), sites=sites, cells=cells)
    names = {0: "ta_m", 1: "pr", **{2 + k: f"tsl{k + 1}" for k in range(6)}, **{8 + k: f"co2_{k}" for k in range(24)}, 32: "treefall",
             129: "k_sat", 130: "sat_theta", 131: "res_theta", 132: "L", 133: "n", 134: "m", 135: "soil_depth"}
    for h in range(24):
        for j, g in enumerate(("hus", "ta_h", "rsds", "wind")):
            names[33 + 4 * h + j] = f"{g}_{h:02d}"
    np.set_printoptions(linewidth=200, precision=4, suppress=True)
    for c in list(range(0, 8)) + [8, 20] + list(range(33, 45)) + list(range(81, 89)) + list(range(129, 136)):
        print(f"ch {c:3d} {names[c]:10s} alpha {alpha[c]:9.5f} beta {beta[c]:12.5f} R2 {r2[c]:.5f} median |r| {mad[c]:.4g} site-offset sd {site_sd[c]:.4g}  (train sd {ctr[:, :, c].std():.4g})")
    for j, g in enumerate(("hus", "ta_h", "rsds", "wind")):
        ch = 33 + 4 * np.arange(24) + j
        print(f"{g}: alpha by hour {alpha[ch]}\n      R2 by hour {r2[ch]}")


if __name__ == "__main__":
    main()
