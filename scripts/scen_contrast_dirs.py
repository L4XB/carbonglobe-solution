#!/usr/bin/env python
"""Scenario-contrast candidate directions (2026-11-07, after young_growth_scen read +0.254 +- 0.028 on day 44).

The young probe (ssp585 +, ssp126 - times the stack's 40-year growth B - R, young ages outside the collapse cluster)
says the truth separates the scenarios more than the stack (kappa ~0.07 of the growth each way). The test CO2 drivers
are compressed ~0.21x, so every member misses most of a CO2 response; if ED saw the real SSP paths the contrast should
also exist on mid and old stands, where it scales with productivity rather than with net growth. Writes raw
directions (template order, scaled units, (N, 2)) to work/dir_<name>.npy; build probes with `scripts/dir_probe.py`.

    ./.venv/bin/python scripts/scen_contrast_dirs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io, shift  # noqa: E402

H_MEAN, A_MEAN = 10.90645, 3.39567


def main() -> None:
    keys = shift.template_keys(); N = len(keys); n_site = 6171
    B = shift.read_predictions(ROOT / "results" / "stack_20261010.csv").astype(np.float64)
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    scen = keys["scenario"].to_numpy(); site = keys["site"].to_numpy(); age = keys["age"].to_numpy()
    sgn = np.where(scen == "ssp585", 1.0, -1.0)
    cl_sites = np.where(np.load(ROOT / "work" / "collapse_cluster_labels.npy") == 0)[0]
    out = ~np.isin(site, cl_sites)
    bands = shift.band_masks(age)
    # y0 npp per template row: test_y0 (6171, 15, 7) per scenario, template is scenario-major, site, age
    npp = np.zeros(N)
    gpp = np.zeros(N)
    for s in io.SCENARIOS:
        _, y0 = io.open_test(s)
        m = scen == s
        npp[m] = np.asarray(y0[:, :, 5], dtype=np.float64).reshape(-1)
        gpp[m] = np.asarray(y0[:, :, 4], dtype=np.float64).reshape(-1)
    veg = (R[:, 1] > 0.02) | (B[:, 1] > 0.02)
    g = B - R
    dirs: dict[str, np.ndarray] = {}
    # 1. mid analogue of the young probe: growth x scenario sign, both columns, outside the cluster
    D = sgn[:, None] * g; D[~(bands["mid"] & out)] = 0; dirs["mid_growth_scen"] = D
    # 2. productivity-shaped contrast (agb): npp0 x scenario sign; npp clipped at 0, vegetated rows only
    for b in ("mid", "old"):
        D = np.zeros((N, 2)); sel = bands[b] & out & veg
        D[sel, 1] = sgn[sel] * np.maximum(npp[sel], 0); dirs[f"{b}_npp_scen_a"] = D
    # 3. relative-level contrast on old agb (the stack's own old contrast is ~+5%)
    D = np.zeros((N, 2)); sel = bands["old"] & out & veg; D[sel, 1] = sgn[sel] * B[sel, 1]; dirs["old_lev_scen_rel_a"] = D
    # 4. inside the collapse cluster: growth x scenario sign, all ages, both columns
    D = sgn[:, None] * g; D[out] = 0; dirs["cl_growth_scen"] = D
    # 5. young split by column (for the record; the joint young probe is scored)
    for c, cn in ((0, "h"), (1, "a")):
        D = np.zeros((N, 2)); sel = bands["young"] & out; D[sel, c] = sgn[sel] * g[sel, c]; dirs[f"young_growth_scen_{cn}"] = D
    yg = np.load(ROOT / "work" / "dir_young_growth_scen.npy")
    for k, D in dirs.items():
        np.save(ROOT / "work" / f"dir_{k}.npy", D)
        e = float(np.mean(D ** 2)); rows = int((D != 0).any(1).sum())
        print(f"{k:22s} energy {e:.5f} rows {rows:6d}")
    # shape overlaps on shared support
    def corr(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
    print("corr mid_growth vs mid_npp (agb col):", round(corr(dirs["mid_growth_scen"][:, 1], dirs["mid_npp_scen_a"][:, 1]), 3))
    print("corr old_npp vs old_lev_rel:", round(corr(dirs["old_npp_scen_a"], dirs["old_lev_scen_rel_a"]), 3))
    lso = np.load(ROOT / "work" / "dir_lev_scen_oa.npy")
    print("corr old_npp vs lev_scen_oa (const):", round(corr(dirs["old_npp_scen_a"], lso), 3), " old_rel vs const:", round(corr(dirs["old_lev_scen_rel_a"], lso), 3))
    yn = np.zeros((N, 2)); sel = bands["young"] & out & veg; yn[sel, 1] = sgn[sel] * np.maximum(npp[sel], 0)
    print("young: corr(growth_a x sgn, npp x sgn) =", round(corr(dirs["young_growth_scen_a"][:, 1], yn[:, 1]), 3),
          " corr(young probe raw, young growth) =", round(corr(yg.ravel(), (dirs["young_growth_scen_h"] + dirs["young_growth_scen_a"]).ravel()), 3))


if __name__ == "__main__":
    main()
