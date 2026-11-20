#!/usr/bin/env python
"""Geolocate test sites on the NEX-GDDP-CMIP6 0.25 deg grid by 120 months (2021-2030) of weather ANOMALIES.

For each site (ssp585 forcing, window years 0-9): monthly anomalies (minus the series' own 10-year monthly
climatology) of pr and ta_m; for every NEX cell within --radius deg of the rsds position estimate the same anomalies
of pr and tas (v2.0); score = mean of the two anomaly correlations. Best cell, second-best outside its 5x5
neighbourhood, and per-variable regression of the raw test series on the raw NEX series at the best cell.
Writes work/test_latlon_nex2.npz.
    python scripts/nex/geolocate_nex2.py [--radius 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io  # noqa: E402


def anom(x: np.ndarray) -> np.ndarray:
    """x: (10, 12, ...) -> anomalies flattened over (120, ...)."""
    a = x - x.mean(axis=0, keepdims=True)
    return a.reshape(120, *x.shape[2:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=5.0)
    args = ap.parse_args()
    g = np.load(ROOT / "work/nexgddp/grid.npz"); lat, lon = g["lat"], g["lon"]
    PR = np.load(ROOT / "work/nexgddp/pr_ssp585_2021_2030_monthly.npy", mmap_mode="r")
    TA = np.load(ROOT / "work/nexgddp/tas_ssp585_2021_2030_monthly.npy", mmap_mode="r")
    xt, _ = io.open_test("ssp585")
    pos = np.load(ROOT / "work/test_latlon_v2.npy")
    n = xt.shape[0]
    keys = ("lat", "lon", "score", "score2", "cpr", "cta", "pr_slope", "pr_icpt", "ta_slope", "ta_icpt", "dist_deg")
    out = {k: np.full(n, np.nan) for k in keys}
    for s in range(n):
        x = np.asarray(xt[s, 0:10], dtype=np.float64)                   # (10, 12, 136)
        tp, tt = x[:, :, 1], x[:, :, 0]
        ap_, at_ = anom(tp), anom(tt)
        if ap_.std() < 1e-9 or at_.std() < 1e-9:
            continue
        la, lo = pos[s]; lo360 = lo % 360
        ri = np.where(np.abs(lat - la) <= args.radius)[0]
        dl = np.abs(((lon - lo360 + 180) % 360) - 180); ci = np.where(dl <= args.radius / max(np.cos(np.radians(la)), 0.2))[0]
        if len(ri) == 0 or len(ci) == 0:
            continue
        P = np.asarray(PR[:, :, ri][:, :, :, ci], dtype=np.float64); T = np.asarray(TA[:, :, ri][:, :, :, ci], dtype=np.float64)
        ok = np.all(np.isfinite(P), axis=(0, 1)) & np.all(np.isfinite(T), axis=(0, 1))
        AP, AT = anom(P), anom(T)

        def corr(a, B):
            a = a - a.mean(); B = B - B.mean(0)
            return np.einsum("t,tij->ij", a, B) / np.sqrt((a @ a) * np.sum(B * B, 0) + 1e-12)
        cp, ct = corr(ap_, AP), corr(at_, AT)
        sc = np.where(ok, 0.5 * (cp + ct), -2.0)
        i, j = np.unravel_index(int(np.argmax(sc)), sc.shape)
        if sc[i, j] < -1:
            continue
        s2 = sc.copy(); s2[max(0, i - 2):i + 3, max(0, j - 2):j + 3] = -2.0
        out["lat"][s], out["lon"][s] = lat[ri[i]], ((lon[ci[j]] + 180) % 360) - 180
        out["score"][s], out["score2"][s], out["cpr"][s], out["cta"][s] = sc[i, j], s2.max(), cp[i, j], ct[i, j]
        for nm, X, y in (("pr", P[:, :, i, j].reshape(-1), tp.reshape(-1)), ("ta", T[:, :, i, j].reshape(-1), tt.reshape(-1))):
            if np.all(np.isfinite(X)) and np.all(np.isfinite(y)) and X.std() > 1e-9:
                try:
                    out[f"{nm}_slope"][s], out[f"{nm}_icpt"][s] = np.polyfit(X, y, 1)
                except np.linalg.LinAlgError:
                    pass
        out["dist_deg"][s] = np.hypot(lat[ri[i]] - la, (((lon[ci[j]] - lo360 + 180) % 360) - 180) * np.cos(np.radians(la)))
        if s % 1000 == 0:
            print(s, round(float(sc[i, j]), 3), flush=True)
    np.savez(ROOT / "work/test_latlon_nex2.npz", **out)
    m = np.isfinite(out["score"])
    pct = lambda k, q: np.nanpercentile(out[k], q)
    print(f"located {m.sum()}; score median {np.nanmedian(out['score']):.3f} (p10 {pct('score', 10):.3f}); second-best median {np.nanmedian(out['score2']):.3f}; "
          f"anomaly corr pr {np.nanmedian(out['cpr']):.3f} ta {np.nanmedian(out['cta']):.3f}")
    print(f"pr: slope {np.nanmedian(out['pr_slope']):.4f} (IQR {pct('pr_slope', 25):.3f}-{pct('pr_slope', 75):.3f}) icpt {np.nanmedian(out['pr_icpt']):.3f}; "
          f"ta: slope {np.nanmedian(out['ta_slope']):.4f} icpt {np.nanmedian(out['ta_icpt']):.3f}; distance to rsds estimate median {np.nanmedian(out['dist_deg']):.2f} deg (p90 {pct('dist_deg', 90):.2f})")


if __name__ == "__main__":
    main()
