#!/usr/bin/env python
"""Geolocate the test sites on the NEX-GDDP-CMIP6 0.25 deg grid by their monthly precipitation fingerprint.

The test forcing derives from NEX-GDDP-CMIP6 GFDL-ESM4 (test year 0 = 2021: max monthly-pr correlation 0.965 vs 0.78
for 2015 / 0.76 for 2022). For each test site (ssp585 forcing, years 0-1, pr minus the 6.3517 release floor) the cell
within --radius deg of the rsds position estimate whose 2021-2022 monthly pr series correlates best is taken; also
the second-best cell outside the best's 3x3 neighbourhood (uniqueness) and the regression slope / intercept at the best.
Writes work/test_latlon_nex.npz (lat, lon, corr, corr2, slope, intercept, dist_deg).
    python scripts/nex/geolocate_nex.py [--radius 5]
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
    ap.add_argument("--radius", type=float, default=5.0)
    args = ap.parse_args()
    g = np.load(ROOT / "work/nexgddp/grid.npz"); lat, lon = g["lat"], g["lon"]
    P = np.concatenate([np.load(ROOT / f"work/nexgddp/pr_GFDL-ESM4_ssp585_{y}_monthly.npy") for y in (2021, 2022)])   # (24, 600, 1440)
    xt, _ = io.open_test("ssp585")
    pos = np.load(ROOT / "work/test_latlon_v2.npy")
    n = xt.shape[0]
    out = {k: np.full(n, np.nan) for k in ("lat", "lon", "corr", "corr2", "slope", "intercept", "dist_deg")}
    for s in range(n):
        p_t = np.asarray(xt[s, 0:2, :, 1], dtype=np.float64).reshape(-1) - 6.3517
        if p_t.std() < 1e-6:
            continue
        la, lo = pos[s]; lo360 = lo % 360
        ri = np.where(np.abs(lat - la) <= args.radius)[0]
        dl = np.abs(((lon - lo360 + 180) % 360) - 180); ci = np.where(dl <= args.radius / max(np.cos(np.radians(la)), 0.2))[0]
        if len(ri) == 0 or len(ci) == 0:
            continue
        Q = P[:, ri][:, :, ci]
        ok = np.all(np.isfinite(Q), axis=0)
        a = p_t - p_t.mean()
        B = Q - Q.mean(0)
        c = np.einsum("t,tij->ij", a, B) / np.sqrt((a @ a) * np.sum(B * B, 0) + 1e-12)
        c = np.where(ok, c, -2.0)
        i, j = np.unravel_index(int(np.argmax(c)), c.shape)
        if c[i, j] < -1.0:
            continue
        c2 = c.copy(); c2[max(0, i - 1):i + 2, max(0, j - 1):j + 2] = -2.0
        q = Q[:, i, j]
        slope, intercept = np.polyfit(q, p_t, 1) if np.all(np.isfinite(q)) and q.std() > 0 else (np.nan, np.nan)
        out["lat"][s], out["lon"][s] = lat[ri[i]], ((lon[ci[j]] + 180) % 360) - 180
        out["corr"][s], out["corr2"][s] = c[i, j], c2.max()
        out["slope"][s], out["intercept"][s] = slope, intercept
        out["dist_deg"][s] = np.hypot(lat[ri[i]] - la, (((lon[ci[j]] - lo360 + 180) % 360) - 180) * np.cos(np.radians(la)))
    np.savez(ROOT / "work/test_latlon_nex.npz", **out)
    ok = np.isfinite(out["corr"])
    print(f"located {ok.sum()} of {n}; best corr median {np.nanmedian(out['corr']):.4f} (p10 {np.nanpercentile(out['corr'], 10):.4f}), "
          f"share > 0.99 {np.nanmean(out['corr'] > 0.99):.2f}, > 0.98 {np.nanmean(out['corr'] > 0.98):.2f}; second-best (outside 3x3) median {np.nanmedian(out['corr2']):.4f}")
    print(f"slope median {np.nanmedian(out['slope']):.4f} (IQR {np.nanpercentile(out['slope'], 25):.3f}-{np.nanpercentile(out['slope'], 75):.3f}), "
          f"intercept median {np.nanmedian(out['intercept']):.3f}; distance to the rsds estimate median {np.nanmedian(out['dist_deg']):.2f} deg (p90 {np.nanpercentile(out['dist_deg'], 90):.2f})")


if __name__ == "__main__":
    main()
