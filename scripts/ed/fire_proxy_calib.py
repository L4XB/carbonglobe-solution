#!/usr/bin/env python
"""Calibrate ED's potential-fire knobs (fp1, fire_max_disturbance_rate) on the organisers' training trajectories with
a Python replica of ED's fire rate (no ED runs).

Replica (work/edv3/build_dryfix: read_site_data.cc calcPETMonthly / calcSiteDrynessIndex, fire.cc update_fuel / fire),
per window year with that year's monthly climate (ED reloads the yearly climate in windows):
    PET_m  = 1.62 b_m(lat) (10 T_m / I)^a  cm/month -> mm/yr (x 120); I = sum (T/5)^1.51, a = Thornthwaite cubic
    P_m    = monthly precipitation x 12 (mm/yr)
    DI_m   = accumulated 30 (PET - P) over consecutive months with PET > P (January wraps over December), DI_avg = mean
    ign_m  = fuel fp1 (DI_avg / 30000)^10, zero when P_m > 100
    fire_y = min(cap, sum_m ign_m / 12)
Fuel = the site's y0 agb at the seed age (kgC/m2). Cumulative burnt fraction F = sum_y fire_y over the 40 years; the
model's expected old-stand survival is ~exp(-F * k). The script reports, per (fp1, cap), the rank correlation between
F and the truth's 40-year agb change ratio (old band, forested sites), and the median truth ratio in F bins.
    python scripts/ed/fire_proxy_calib.py --n 6000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io  # noqa: E402

B_TABLE = [  # (|lat| upper bound, 12 factors), read_site_data.cc
    (5.0, [1.04, 0.94, 1.04, 1.01, 1.04, 1.01, 1.04, 1.04, 1.01, 1.04, 1.01, 1.04]),
    (15.0, [1.00, 0.91, 1.03, 1.03, 1.08, 1.06, 1.08, 1.07, 1.02, 1.02, 0.98, 0.99]),
    (25.0, [0.95, 0.90, 1.03, 1.05, 1.13, 1.11, 1.14, 1.11, 1.02, 1.00, 0.93, 0.94]),
    (32.5, [0.90, 0.87, 1.03, 1.08, 1.18, 1.17, 1.20, 1.14, 1.03, 0.98, 0.89, 0.88]),
    (37.5, [0.87, 0.85, 1.03, 1.09, 1.21, 1.21, 1.23, 1.16, 1.03, 0.97, 0.86, 0.85]),
    (42.5, [0.84, 0.83, 1.03, 1.11, 1.24, 1.25, 1.27, 1.18, 1.04, 0.96, 0.83, 0.81]),
    (47.5, [0.80, 0.81, 1.02, 1.13, 1.28, 1.29, 1.31, 1.21, 1.04, 0.94, 0.79, 0.75]),
    (90.0, [0.74, 0.78, 1.02, 1.15, 1.33, 1.36, 1.37, 1.25, 1.06, 0.92, 0.76, 0.70]),
]


def b_factors(lat: float) -> np.ndarray:
    for ub, b in B_TABLE:
        if abs(lat) <= ub:
            return np.array(b)
    return np.array(B_TABLE[-1][1])


def dryness_avg(T: np.ndarray, P: np.ndarray, b: np.ndarray) -> np.ndarray:
    """T (years, 12) deg C, P (years, 12) mm/month -> DI_avg per year (ED units)."""
    Tc = np.maximum(T, 0.0001)
    I = np.sum((Tc / 5.0) ** 1.51, axis=1, keepdims=True)
    a = 67.5e-8 * I ** 3 - 77.1e-6 * I ** 2 + 0.0179 * I + 0.492
    pet = 1.62 * b[None, :] * (10.0 * Tc / I) ** a * 120.0          # mm/yr
    pr = np.maximum(P * 12.0, 1e-9)                                  # mm/yr
    dry = pet / pr > 1.0
    inc = 30.0 * (pet - pr)
    ny = T.shape[0]
    di = np.zeros((ny, 12))
    for y in range(ny):
        d0 = 0.0
        if dry[y, 0]:
            d0 = inc[y, 0]
            for m2 in range(11, 0, -1):          # January also collects the dry months that end the year
                if dry[y, m2]:
                    d0 += inc[y, m2]
                else:
                    break
        di[y, 0] = d0
        for m in range(1, 12):
            di[y, m] = di[y, m - 1] + inc[y, m] if dry[y, m] else 0.0
    return di.mean(axis=1)


def burnt(T: np.ndarray, P: np.ndarray, lat: float, fuel: float, fp1: float, cap: float, p_scale: float = 1.0) -> float:
    P = P * p_scale                                                      # unit hypothesis: what ED reads as "precip"
    dia = dryness_avg(T, P, b_factors(lat))                              # (years,)
    ign = fuel * fp1 * (dia / 30000.0) ** 10                             # per wet/dry month
    nd = np.sum(P * 12.0 <= 100.0, axis=1)                               # months that may ignite
    fire = np.minimum(cap, ign * nd / 12.0)
    return float(fire.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--fp1", default="1,0.1,0.01,1e-3,1e-4,1e-6")
    ap.add_argument("--caps", default="0.1,0.2")
    ap.add_argument("--p-scale", type=float, default=1.0, help="multiply the monthly precipitation (mm/month) before ED's fire code; 1/30.4 = mm/day")
    ap.add_argument("--out", default=str(ROOT / "work" / "edv3" / "fire_proxy_train.csv"))
    ap.add_argument("--sample", default=str(ROOT / "work" / "edv3" / "train_nearbare_sample.csv"))
    args = ap.parse_args()
    s = pd.read_csv(args.sample)
    old = s[(s.age >= 200)].groupby("site").agg(T=("T", "first"), y0a=("y0_a", "median"), y40a=("y40_a", "median"),
                                                 y0h=("y0_h", "median"), y40h=("y40_h", "median"))
    old = old[(old.y0a > 2.0)].copy()                                  # forested old stands
    old["r_a"] = old.y40a / old.y0a
    old = old.iloc[: args.n]
    x = io.open_train_x(); tll = np.load(ROOT / "work" / "edv3" / "train_latlon.npy")
    clim = {}
    for site in old.index:
        xs = np.asarray(x[site, :, :, 0:2], dtype=np.float64)
        clim[site] = (xs[:, :, 0] - 273.15, xs[:, :, 1], float(tll[site, 0]))
    print(f"{len(old)} forested training sites (old-band y0 agb > 2 kgC); truth old agb ratio median {old.r_a.median():.3f}, "
          f"share < 0.6: {np.mean(old.r_a < 0.6):.3f}")
    rows = []
    for fp1 in [float(v) for v in args.fp1.split(",")]:
        for cap in [float(v) for v in args.caps.split(",")]:
            F = np.array([burnt(*clim[st][:2], clim[st][2], float(old.y0a[st]), fp1, cap, args.p_scale) for st in old.index])
            rho = pd.Series(F).corr(pd.Series(-np.log(old.r_a.to_numpy())), method="spearman")
            # survival model fit: log r = c - k F (least squares through the median)
            lr = np.log(old.r_a.to_numpy()); A = np.c_[np.ones_like(F), -F]
            coef, res, *_ = np.linalg.lstsq(A, lr, rcond=None)
            r2 = 1.0 - np.sum((lr - A @ coef) ** 2) / np.sum((lr - lr.mean()) ** 2)
            rows.append(dict(fp1=fp1, cap=cap, share_burning=float(np.mean(F > 0.05)), F_med=float(np.median(F)),
                             F_p90=float(np.quantile(F, 0.9)), spearman=float(rho), R2=float(r2), k=float(coef[1])))
            old[f"F_{fp1:g}_{cap:g}"] = F
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))
    best = t.sort_values("R2").iloc[-1]
    col = f"F_{best.fp1:g}_{best.cap:g}"
    old["Fbin"] = pd.cut(old[col], [-1, 0.05, 0.5, 1.5, 3, 100])
    print(f"best by R2: fp1 {best.fp1:g} cap {best.cap:g}; truth old agb ratio by F bin:")
    print(old.groupby("Fbin", observed=True).agg(n=("r_a", "size"), r_med=("r_a", "median"), T_med=("T", "median")).round(3).to_string())
    old.to_csv(args.out)


if __name__ == "__main__":
    main()
