#!/usr/bin/env python
"""Geolocate test sites on the NEX-GDDP-CMIP6 0.25-deg grid by 120 months (2021-2030) of pr and ta anomalies, searching
ALL longitudes inside a latitude band around a prior latitude (geolocate_nex2.py searched only 5 deg around the rsds
estimate, whose longitudes are off by ~4-5 deg; the static-map check says nex2 positions are far better than rsds
ones, except where the true cell lay outside that radius).
score = mean of the two anomaly correlations (as in nex2). Prior latitude: --prior (npz with lat/lon, or (n, 2) npy).
    python scripts/nex/geolocate_nex3.py [--band 6] [--prior work/test_latlon_v2.npy]
-> work/test_latlon_nex3.npz (lat, lon, score, score2, cpr, cta, d_nex2_deg)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io  # noqa: E402


def norm_anom(x: np.ndarray) -> np.ndarray:
    """x: (10, 12, ...) -> monthly anomalies (120, ...), zero mean and unit norm along time (nan where undefined)."""
    a = (x - x.mean(axis=0, keepdims=True)).reshape(120, *x.shape[2:])
    a = a - a.mean(axis=0, keepdims=True)
    nrm = np.sqrt((a * a).sum(axis=0, keepdims=True))
    return np.where(nrm > 1e-9, a / np.maximum(nrm, 1e-12), np.nan).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", type=float, default=6.0, help="half-width (deg) of the latitude band around the prior")
    ap.add_argument("--prior", default="work/test_latlon_v2.npy")
    args = ap.parse_args()
    t0 = time.time()
    g = np.load(ROOT / "work/nexgddp/grid.npz"); lat, lon = g["lat"], g["lon"]
    PR = np.load(ROOT / "work/nexgddp/pr_ssp585_2021_2030_monthly.npy").astype(np.float64)
    TA = np.load(ROOT / "work/nexgddp/tas_ssp585_2021_2030_monthly.npy").astype(np.float64)
    AP, AT = norm_anom(PR), norm_anom(TA); del PR, TA
    okc = np.isfinite(AP).all(axis=0) & np.isfinite(AT).all(axis=0)
    AP = np.nan_to_num(AP); AT = np.nan_to_num(AT)
    print(f"NEX anomalies {AP.shape} ready in {time.time() - t0:.0f} s; valid cells {okc.sum()}", flush=True)
    pz = np.load(ROOT / args.prior)
    prior = np.stack([pz["lat"], pz["lon"]], 1) if hasattr(pz, "files") else pz
    z2 = np.load(ROOT / "work/test_latlon_nex2.npz")
    xt, _ = io.open_test("ssp585")
    n = xt.shape[0]
    keys = ("lat", "lon", "score", "score2", "cpr", "cta", "d_nex2_deg")
    out = {k: np.full(n, np.nan) for k in keys}
    lon180 = ((lon + 180) % 360) - 180
    for s in range(n):
        x = np.asarray(xt[s, 0:10], dtype=np.float64)
        a_p = norm_anom(x[:, :, 1]); a_t = norm_anom(x[:, :, 0])
        if not (np.isfinite(a_p).all() and np.isfinite(a_t).all()):
            continue
        la = prior[s, 0]
        if not np.isfinite(la):
            la = z2["lat"][s]
        if not np.isfinite(la):
            continue
        ri = np.where(np.abs(lat - la) <= args.band)[0]
        if len(ri) == 0:
            continue
        cp = np.tensordot(a_p, AP[:, ri, :], axes=(0, 0)); ct = np.tensordot(a_t, AT[:, ri, :], axes=(0, 0))
        sc = np.where(okc[ri], 0.5 * (cp + ct), -2.0)
        i, j = np.unravel_index(int(np.argmax(sc)), sc.shape)
        s2 = sc.copy(); jj = np.arange(j - 2, j + 3) % sc.shape[1]; s2[max(0, i - 2):i + 3][:, jj] = -2.0
        out["lat"][s], out["lon"][s] = lat[ri[i]], lon180[j]
        out["score"][s], out["score2"][s], out["cpr"][s], out["cta"][s] = sc[i, j], s2.max(), cp[i, j], ct[i, j]
        if np.isfinite(z2["lat"][s]):
            out["d_nex2_deg"][s] = np.hypot(lat[ri[i]] - z2["lat"][s], (((lon180[j] - z2["lon"][s] + 180) % 360) - 180) * np.cos(np.radians(la)))
        if s % 500 == 0:
            print(f"  {s}/{n} {time.time() - t0:.0f} s, score {sc[i, j]:.3f}", flush=True)
    np.savez(ROOT / "work/test_latlon_nex3.npz", **out)
    d = out["d_nex2_deg"]
    print(f"located {np.isfinite(out['score']).sum()}; score median {np.nanmedian(out['score']):.3f} (nex2 {np.nanmedian(z2['score']):.3f}); "
          f"moved > 1 deg vs nex2: {np.nanmean(d > 1):.3f}, > 3 deg: {np.nanmean(d > 3):.3f}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
