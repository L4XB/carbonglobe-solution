#!/usr/bin/env python
"""Affine maps released = a * real + b of the monthly test channels against NEX-GDDP-CMIP6 GFDL-ESM4 ssp585 at the EXACT
0.5-deg cell of each matched test site (mean of the 2x2 NEX 0.25-deg cells), years 2021-2030 (pr, tas) and 2021-2022
(huss, rsds, sfcWind where on disk). Robust (trimmed) fits over sites x months; prints R2 and the residual spread, and
the per-site spread of slope / intercept (a single global map should leave none).
    python scripts/geo/calib_nex.py
-> work/cal/nex_cell_series.npz (tas, pr at the exact cells; test ch0 / ch1 for the same months), work/cal/calib_nex.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io  # noqa: E402

N = ROOT / "work/nexgddp"


def cell_mean(arr: np.ndarray, lat: np.ndarray, lon: np.ndarray, clat: np.ndarray, clon: np.ndarray) -> np.ndarray:
    """arr (..., 600, 1440) -> (..., n) mean of the 2x2 NEX cells inside each 0.5-deg cell centred (clat, clon)."""
    parts = []
    for dla in (-0.125, 0.125):
        for dlo in (-0.125, 0.125):
            ri = np.clip(np.round((clat + dla - lat[0]) / 0.25).astype(int), 0, len(lat) - 1)
            ci = np.round(((clon + dlo) % 360 - lon[0]) / 0.25).astype(int) % len(lon)
            parts.append(arr[..., ri, ci])
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.stack(parts, 0), axis=0)          # coastal cells: the land NEX cells only


def robust_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    a, b = np.polyfit(x, y, 1)
    for _ in range(6):
        r = y - (a * x + b); keep = np.abs(r) <= np.quantile(np.abs(r), 0.8)
        a, b = np.polyfit(x[keep], y[keep], 1)
    r = y - (a * x + b)
    return float(a), float(b), float(1 - np.var(r) / np.var(y)), float(np.median(np.abs(r)))


def main() -> None:
    g = np.load(N / "grid.npz"); lat, lon = g["lat"], g["lon"]
    ex = np.load(ROOT / "work/test_cell_exact.npz"); ok = ex["cell"] >= 0
    sites = np.where(ok)[0]; clat, clon = ex["lat"][ok], ex["lon"][ok]
    xt, _ = io.open_test("ssp585")
    res = {}
    series = {"sites": sites}
    for name, fn, ch, years in (("tas", "tas_ssp585_2021_2030_monthly.npy", 0, 10), ("pr", "pr_ssp585_2021_2030_monthly.npy", 1, 10)):
        A = np.load(N / fn, mmap_mode="r")                               # (years, 12, 600, 1440)
        nx = np.stack([cell_mean(np.asarray(A[y]), lat, lon, clat, clon) for y in range(years)], 0)     # (years, 12, n)
        nx = np.transpose(nx, (2, 0, 1))                                  # (n, years, 12)
        te = np.asarray(xt[:, :years, :, ch], dtype=np.float64)[sites]    # (n, years, 12)
        series[name] = nx.astype(np.float32); series[f"test_{name}"] = te.astype(np.float32)
        a, b, r2, mad = robust_fit(nx.ravel(), te.ravel())
        # per-site fits
        good = [i for i in range(0, len(sites), 7) if np.isfinite(nx[i]).all() and nx[i].std() > 1e-6]
        sl = np.array([np.polyfit(nx[i].ravel(), te[i].ravel(), 1) for i in good])
        cc = np.array([np.corrcoef(nx[i].ravel(), te[i].ravel())[0, 1] for i in good])
        res[name] = dict(a=a, b=b, r2=r2, mad=mad)
        print(f"{name}: released = {a:.5f} * NEX + {b:.4f}; R2 {r2:.5f}; median |resid| {mad:.4f}; per-site corr median {np.median(cc):.4f} (p10 {np.percentile(cc, 10):.4f}); "
              f"per-site slope median {np.median(sl[:, 0]):.4f} IQR {np.percentile(sl[:, 0], 25):.4f}-{np.percentile(sl[:, 0], 75):.4f}", flush=True)
    np.savez(ROOT / "work/cal/nex_cell_series.npz", **series)
    (ROOT / "work/cal/calib_nex.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
