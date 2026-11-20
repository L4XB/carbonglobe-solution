#!/usr/bin/env python
"""Row-level agb prediction from ORNL DAAC 2522 for ALL seed ages at the exact cell (v2).

    agb40(age) = lookup(age + 40) x exp(q(age)),     q(age) = q_old + w(age) (q_new - q_old)
lookup = the site's y0 agb profile (log-linear in age, flat beyond 500), q_old / q_new as in ornl_rows.py, w(age) fitted on
ED at the fire-never moist / warm sites (scripts/ornl/mid_ages.py: sd(ED - this) 0.02-0.04 at every age).
A second estimate per row uses the cell's pixels matched by their REF growth (q_match, >= 5 pixels); stored for checks.
Cells with only q_new: age-1 rows only. Cells with only q_old: old rows (w = 0 there) only.
Writes work/ornl2522/rows_<tag>.npz: pred, kind (1 old, 2 age 1, 3 intermediate), q, qmatch, npx.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "ornl"))
from cgb import io, shift  # noqa: E402
import ornl_rows as orr  # noqa: E402

W = {1: 0.85, 10: 0.79, 20: 0.76, 30: 0.72, 50: 0.57, 70: 0.45, 100: 0.38, 150: 0.19, 200: 0.0, 250: 0.0, 300: 0.0, 350: 0.0, 400: 0.0, 450: 0.0, 500: 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gmax", type=float, default=0.12)
    ap.add_argument("--min-old", type=int, default=6)
    ap.add_argument("--min-new", type=int, default=6)
    ap.add_argument("--tag", default="v2")
    args = ap.parse_args()
    ex = np.load(ROOT / "work/test_cell_exact.npz"); lat, lon = ex["lat"], ex["lon"]
    p2, cell, profile, AGES = orr.p2, orr.cell, orr.profile, orr.AGES
    R0 = p2("Biomass_REF_existing_forest_2021_2030", 0); R1 = p2("Biomass_REF_existing_forest_2051_2060", 9)
    S1 = {sc: p2(f"Biomass_GFDL-ESM4_SSP{sc}_CLIM_CO2_existing_forest_2051_2060", 9) for sc in ("126", "585")}
    NR = p2("Biomass_REF_new_forest_2051_2060", 9); N1 = {sc: p2(f"Biomass_GFDL-ESM4_SSP{sc}_CLIM_CO2_new_forest_2051_2060", 9) for sc in ("126", "585")}
    _, y1 = io.open_test("ssp126"); _, y5 = io.open_test("ssp585"); FL = shift.FLOORS; M = shift.M_HEIGHT
    Y = {"126": (y1[:, :, 1] - FL[1]) * M * orr.F_AGB, "585": (y5[:, :, 1] - FL[1]) * M * orr.F_AGB}
    keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy(); scen = keys.scenario.to_numpy()
    n = len(keys)
    pred = np.full(n, np.nan); kind = np.zeros(n, dtype=np.int8); qrow = np.full(n, np.nan); qmatch = np.full(n, np.nan); npx = np.zeros(n, dtype=np.int32)
    order = np.argsort(site, kind="stable"); bounds = np.r_[0, np.flatnonzero(np.diff(site[order])) + 1, n]
    cnt = dict(both=0, old_only=0, new_only=0, rej=0)
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        rows = order[b0:b1]; s = int(site[rows[0]])
        if not np.isfinite(lat[s]) or int(ex["cell"][s]) < 0:
            continue
        r0 = cell(R0, lat[s], lon[s]); r1 = cell(R1, lat[s], lon[s])
        okx = np.isfinite(r0) & np.isfinite(r1) & (r0 > 0.3) & (r1 > 0)
        g = np.where(okx, np.log(np.maximum(r1, 1e-6) / np.maximum(r0, 1e-6)), np.nan)
        old = okx & (r0 > 1.0) & (g < args.gmax) & (g > -0.2)
        nr = cell(NR, lat[s], lon[s])
        for sc in ("126", "585"):
            y = Y[sc][s]; yold = float(y[7:].mean())
            s1 = cell(S1[sc], lat[s], lon[s]); n1 = cell(N1[sc], lat[s], lon[s])
            q_old = np.nan; q_new = np.nan
            m = old & np.isfinite(s1) & (s1 > 0)
            if m.sum() >= args.min_old and yold > 1.0:
                if 0.6 <= float(np.median(r0[m])) / yold <= 1.7:
                    q_old = float(np.median(np.log(s1[m] / r1[m]))); n_old = int(m.sum())
                else:
                    cnt["rej"] += 1
            mn = np.isfinite(nr) & np.isfinite(n1) & (nr > 0.3) & (n1 > 0)
            if mn.sum() >= args.min_new:
                q_new = float(np.median(np.log(n1[mn] / nr[mn])))
            if np.isfinite(q_old) and np.isfinite(q_new): cnt["both"] += 1
            elif np.isfinite(q_old): cnt["old_only"] += 1
            elif np.isfinite(q_new): cnt["new_only"] += 1
            mx_ = okx & np.isfinite(s1) & (s1 > 0)
            gx = g[mx_]; qx = np.log(s1[mx_] / r1[mx_])
            for i in rows[scen[rows] == "ssp" + sc]:
                a = int(age[i]); w = W[a]
                if a >= 200 and np.isfinite(q_old):
                    q = q_old
                elif np.isfinite(q_old) and np.isfinite(q_new):
                    q = q_old + w * (q_new - q_old)
                elif a == 1 and np.isfinite(q_new):
                    q = q_new                       # no old-growth pixels: the new-forest effect itself (w would need q_old)
                elif a == 150 and np.isfinite(q_old):
                    q = q_old
                else:
                    continue
                pred[i] = profile(y, a + 40) * np.exp(q); qrow[i] = q
                kind[i] = 2 if a == 1 else (1 if a >= 150 else 3)
                if a > 1 and gx.size:
                    gstar = np.log(profile(y, a + 40) / profile(y, a + 1))
                    sel = np.abs(gx - gstar) <= max(0.25 * abs(gstar), 0.04)
                    npx[i] = int(sel.sum())
                    if sel.sum() >= 5:
                        qmatch[i] = float(np.median(qx[sel]))
    np.savez(ROOT / f"work/ornl2522/rows_{args.tag}.npz", pred=pred, kind=kind, q=qrow, qmatch=qmatch, npx=npx)
    print(f"site x scenario: q_old and q_new {cnt['both']}, q_old only {cnt['old_only']}, q_new only {cnt['new_only']}, old rejected for a state mismatch {cnt['rej']}; "
          f"rows: old {int((kind == 1).sum())}, age 1 {int((kind == 2).sum())}, intermediate {int((kind == 3).sum())} of {n}; with a matched-pixel estimate {int(np.isfinite(qmatch).sum())}")


if __name__ == "__main__":
    main()
