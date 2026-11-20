#!/usr/bin/env python
"""Climate tilts of the young scenario contrast (2026-11-07).

young_growth_scen (sgn x (B - R) on ages 1-30 outside the collapse cluster) read +0.254 +- 0.028. Under a CO2
response (Farquhar/Collatz) the fertilisation per unit CO2 grows with leaf temperature; under a warming response the
sign flips between cold and hot sites. The tilt sgn x (B - R) x z(site) with z the standardized window-mean
temperature (or the scenario warming dT = T585 - T126) separates the two. Writes work/dir_young_scen_tilt_<z>.npy.

    ./.venv/bin/python scripts/scen_tilt_dirs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io, shift  # noqa: E402


def main() -> None:
    keys = shift.template_keys(); N = len(keys)
    B = shift.read_predictions(ROOT / "results" / "stack_20261010.csv").astype(np.float64)
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    scen = keys["scenario"].to_numpy(); site = keys["site"].to_numpy(); age = keys["age"].to_numpy()
    sgn = np.where(scen == "ssp585", 1.0, -1.0)
    out = ~np.isin(site, np.where(np.load(ROOT / "work" / "collapse_cluster_labels.npy") == 0)[0])
    young = shift.band_masks(age)["young"] & out
    T = {}
    for s in io.SCENARIOS:
        x, _ = io.open_test(s)                                   # (6171, 40, 12, 136) memory-mapped
        T[s] = np.asarray(x[:, :, :, 0], dtype=np.float64).mean(axis=(1, 2))   # (6171,) window-mean driver 0
    Tm = 0.5 * (T["ssp126"] + T["ssp585"]); dT = T["ssp585"] - T["ssp126"]
    print(f"site mean T: {np.percentile(Tm, [5, 50, 95]).round(2)}  dT: {np.percentile(dT, [5, 50, 95]).round(3)}")
    np.save(ROOT / "work" / "site_Tmean_dT.npy", np.stack([Tm, dT], 1))
    for nm, z in (("T", Tm), ("dT", dT)):
        zz = (z - z.mean()) / z.std()
        D = sgn[:, None] * (B - R) * zz[site][:, None]
        D[~young] = 0
        np.save(ROOT / "work" / f"dir_young_scen_tilt_{nm}.npy", D)
        print(f"young_scen_tilt_{nm}: energy {np.mean(D ** 2):.5f}")
    print("corr(Tm, dT) =", round(float(np.corrcoef(Tm, dT)[0, 1]), 3))


if __name__ == "__main__":
    main()
