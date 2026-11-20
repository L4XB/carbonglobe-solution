#!/usr/bin/env python
"""Row-level agb prediction from ORNL DAAC 2522 (GFDL-ESM4, CLIM_CO2) at the EXACT 0.5-deg cell of each test site.

  old rows (seed age >= 150):  agb40 = y0 x G_lookup(age) x exp(q_old),
      q_old = median over the cell's old-growth existing-forest pixels of log(B2060_ssp / B2060_REF)
      (old-growth = pixels whose REF run grows less than `gmax` in log from 2021 to 2060; the state must resemble the
       site's: median 2021 biomass of these pixels within [0.6, 1.7] x the site's old y0 agb)
      G_lookup(age) = y0 profile at age + 40 over the profile at age (log-linear in age, 1 beyond 500)
  age-1 rows:                  agb40 = y0 profile at age 41 x exp(q_new),
      q_new = median over the cell's new-forest pixels of log(N2060_ssp / N2060_REF)
Units: ORNL biomass = decoded test agb x 1.055 (REF new forest 2060 vs the lookup at age 41: ratio 0.99).
Writes work/ornl2522/rows_<tag>.npz: pred (n_rows,) physical agb (NaN = no prediction), kind (0 none, 1 old, 2 age 1), q, npx.
    python scripts/ornl/ornl_rows.py [--gmax 0.12] [--tag v1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io, shift  # noqa: E402

D = ROOT / "work" / "ornl2522"; RES = 0.04; F_AGB = 1.055
AGES = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500])


def p2(stem: str, band: int) -> np.ndarray:
    a = np.load(D / f"p2_{stem}_b{band}.npy")
    return np.where(a == -32768, np.nan, a.astype(np.float32) * np.float32(0.01))


def cell(a: np.ndarray, la: float, lo: float) -> np.ndarray:
    r0 = int(round((80.0 - (la + 0.25)) / RES)); r1 = int(round((80.0 - (la - 0.25)) / RES))
    c0 = int(round((lo - 0.25 + 180.0) / RES)); c1 = int(round((lo + 0.25 + 180.0) / RES))
    return a[max(r0, 0):r1, c0:c1].ravel()


def profile(y: np.ndarray, age: float) -> float:
    """y0 agb profile (15,) at a stand age, log-linear in age; flat beyond 500."""
    return float(np.exp(np.interp(np.log(min(max(age, 1.0), 500.0)), np.log(AGES), np.log(np.maximum(y, 1e-3)))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gmax", type=float, default=0.12)
    ap.add_argument("--min-old", type=int, default=6)
    ap.add_argument("--min-new", type=int, default=6)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    ex = np.load(ROOT / "work/test_cell_exact.npz"); lat, lon = ex["lat"], ex["lon"]
    R0 = p2("Biomass_REF_existing_forest_2021_2030", 0); R1 = p2("Biomass_REF_existing_forest_2051_2060", 9)
    S1 = {sc: p2(f"Biomass_GFDL-ESM4_SSP{sc}_CLIM_CO2_existing_forest_2051_2060", 9) for sc in ("126", "585")}
    NR = p2("Biomass_REF_new_forest_2051_2060", 9); N1 = {sc: p2(f"Biomass_GFDL-ESM4_SSP{sc}_CLIM_CO2_new_forest_2051_2060", 9) for sc in ("126", "585")}
    _, y1 = io.open_test("ssp126"); _, y5 = io.open_test("ssp585"); FL = shift.FLOORS; M = shift.M_HEIGHT
    Y = {"126": (y1[:, :, 1] - FL[1]) * M * F_AGB, "585": (y5[:, :, 1] - FL[1]) * M * F_AGB}
    keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy(); scen = keys.scenario.to_numpy()
    n = len(keys)
    pred = np.full(n, np.nan); kind = np.zeros(n, dtype=np.int8); qrow = np.full(n, np.nan); npx = np.zeros(n, dtype=np.int32)
    rows_of: dict[int, np.ndarray] = {}
    order = np.argsort(site, kind="stable")
    bounds = np.r_[0, np.flatnonzero(np.diff(site[order])) + 1, n]
    stats = dict(old=0, new=0, old_rej_state=0)
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        rows = order[b0:b1]; s = int(site[rows[0]])
        if not np.isfinite(lat[s]) or int(ex["cell"][s]) < 0:
            continue
        r0 = cell(R0, lat[s], lon[s]); r1 = cell(R1, lat[s], lon[s])
        ok = np.isfinite(r0) & np.isfinite(r1) & (r0 > 1.0) & (r1 > 0)
        g = np.where(ok, np.log(np.maximum(r1, 1e-6) / np.maximum(r0, 1e-6)), np.nan)
        old = ok & (g < args.gmax) & (g > -0.2)
        nr = cell(NR, lat[s], lon[s])
        for sc in ("126", "585"):
            y = Y[sc][s]; yold = float(y[7:].mean())
            # ---- old rows
            s1 = cell(S1[sc], lat[s], lon[s]); m = old & np.isfinite(s1) & (s1 > 0)
            if m.sum() >= args.min_old and yold > 1.0:
                bmed = float(np.median(r0[m]))
                if 0.6 <= bmed / yold <= 1.7:
                    q = float(np.median(np.log(s1[m] / r1[m])))
                    for i in rows[(scen[rows] == "ssp" + sc) & (age[rows] >= 150)]:
                        a = int(age[i]); j = int(np.where(AGES == a)[0][0])
                        G = profile(y, a + 40) / profile(y, a)
                        pred[i] = y[j] * G * np.exp(q); kind[i] = 1; qrow[i] = q; npx[i] = int(m.sum())
                    stats["old"] += 1
                else:
                    stats["old_rej_state"] += 1
            # ---- age-1 rows
            n1 = cell(N1[sc], lat[s], lon[s]); m = np.isfinite(nr) & np.isfinite(n1) & (nr > 0.3) & (n1 > 0)
            if m.sum() >= args.min_new:
                q = float(np.median(np.log(n1[m] / nr[m])))
                for i in rows[(scen[rows] == "ssp" + sc) & (age[rows] == 1)]:
                    pred[i] = profile(y, 41) * np.exp(q); kind[i] = 2; qrow[i] = q; npx[i] = int(m.sum())
                stats["new"] += 1
    np.savez(D / f"rows_{args.tag}.npz", pred=pred, kind=kind, q=qrow, npx=npx)
    print(f"site x scenario with an old-row prediction {stats['old']} (rejected for a state mismatch {stats['old_rej_state']}), with an age-1 prediction {stats['new']}; "
          f"rows: old {int((kind == 1).sum())}, age 1 {int((kind == 2).sum())} of {n}")


if __name__ == "__main__":
    main()
