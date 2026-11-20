#!/usr/bin/env python
"""Does the test's seed-age lookup follow ORNL DAAC 2522's REF new-forest runs (bare ground planted 2020, present-day
forcing repeated)? Compare the decoded test y0 agb at seed ages 1 / 10 / 30 with the ORNL REF new-forest biomass at
2021 / 2030 / 2051 (ages 1 / 10 / 31), Gaussian-weighted around each site's rsds position estimate. Also prints the
same for SSP585 new forest at 2060 (age 40) against the stack's age-1 y40 prediction.
    python scripts/ornl/newforest_vs_y0.py [--sigma 1.0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "ornl"))
from cgb import shift  # noqa: E402
from ornl_sites import load, site_means  # noqa: E402

DIV = np.array([10.90645, 3.39567])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=1.0)
    args = ap.parse_args()
    pos = np.load(ROOT / "work/test_latlon_v2.npy")
    keys = shift.template_keys(); site = keys["site"].to_numpy(); age = keys["age"].to_numpy(); scen = keys["scenario"].to_numpy()
    R = shift.read_predictions(ROOT / "results/latent_ref_y0_m114114.csv").astype(np.float64) * DIV
    C = shift.read_predictions(ROOT / "results/spatcorr_stack_20261106_s08.csv").astype(np.float64) * DIV
    nsite = site.max() + 1
    def per_site(v: np.ndarray, a: int, s: str = "ssp126") -> np.ndarray:
        out = np.full(nsite, np.nan); m = (age == a) & (scen == s); out[site[m]] = v[m]; return out
    pairs = [(1, "Biomass_REF_new_forest_2021_2030", "first"), (10, "Biomass_REF_new_forest_2021_2030", "last"),
             (30, "Biomass_REF_new_forest_2051_2060", "first")]
    for a, stem, band in pairs:
        lat, lon, m, f = load(stem, band)
        b, w = site_means(lat, lon, m, f, pos, args.sigma)
        y = per_site(R[:, 1], a)
        ok = np.isfinite(b) & (w > 0.05) & (y > 0)
        lb, ly = np.log(b[ok] + 0.01), np.log(y[ok] + 0.01)
        print(f"age {a:3d} vs ORNL REF new {stem[-9:]} {band}: n {ok.sum()}, median y0 {np.median(y[ok]):.3f}, median ORNL "
              f"{np.median(b[ok]):.3f}, median ratio y0/ORNL {np.median(y[ok] / np.maximum(b[ok], 1e-3)):.3f}, corr(log) {np.corrcoef(lb, ly)[0, 1]:+.3f}")
    for s in ("ssp126", "ssp585"):
        lat, lon, m, f = load(f"Biomass_GFDL-ESM4_{s.upper()}_CLIM_CO2_new_forest_2051_2060", "last")
        b, w = site_means(lat, lon, m, f, pos, args.sigma)
        y = per_site(C[:, 1], 1, s); y0 = per_site(R[:, 1], 30, s)
        ok = np.isfinite(b) & (w > 0.05) & (y > 0)
        print(f"{s} new forest 2060 vs stack y40(age 1): n {ok.sum()}, median stack {np.median(y[ok]):.3f}, ORNL {np.median(b[ok]):.3f}, "
              f"ratio {np.median(y[ok] / np.maximum(b[ok], 1e-3)):.3f}, corr(log) {np.corrcoef(np.log(b[ok] + .01), np.log(y[ok] + .01))[0, 1]:+.3f}; "
              f"y0(age 30) median {np.nanmedian(y0[ok]):.3f}")


if __name__ == "__main__":
    main()
