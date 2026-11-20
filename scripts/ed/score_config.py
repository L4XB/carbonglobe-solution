#!/usr/bin/env python
"""Score an ED configuration run (a --root of run_site.py) against test_y0 and the board's block facts.

(i)   y0 fit: RMS of log(ED_y0est / y0_decoded) over height, agb (x1.14), lai, 15 ages, both scenarios
      (vegetated cells: y0 > 0.05); also the scenario difference at ages 1-30 (ED vs y0).
(ii)  block slopes of the ED prediction on decoded persistence R in truth units, outside the cluster, both
      scenarios: targets old h 0.891, old agb 0.888, mid h 0.941, mid agb ~1.05. Two forms: level-factor
      (assemble_ed.py: ED40 x clip(y0/spin)) and pure ratio (R x ED40/ED_spin).
(iii) young growth contrast (T-R)_s / (B-R)_s - 1 per scenario (targets +7% ssp585, -7% ssp126) and the
      old-agb contrast mean(T585 - T126) in truth units (target about +0.08).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

MH, MA, HS = 10.90645, 3.395674, 0.876


def load(root: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(root / "csv" / "site*.csv")))
    if not files:
        raise SystemExit(f"no site CSVs under {root}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def slope(t: np.ndarray, r: np.ndarray) -> float:
    m = np.isfinite(t) & np.isfinite(r) & (r > 0)
    return float((t[m] * r[m]).sum() / (r[m] ** 2).sum()) if m.sum() else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--base", default=str(ROOT / "results" / "stack_20261010.csv"), help="B for the growth contrast (10-10 stack)")
    ap.add_argument("--stack", default=str(ROOT / "results" / "spatcorr_stack_20261106_s08.csv"))
    ap.add_argument("--common", action="store_true", help="restrict every root to the sites finished in all roots")
    args = ap.parse_args()
    common: set[int] | None = None
    if args.common:
        for root in args.roots:
            s = {int(Path(f).stem[4:]) for f in glob.glob(str(Path(root) / "csv" / "site*.csv"))}
            common = s if common is None else common & s
        print(f"common sites across roots: {len(common or [])}")
    keys = shift.template_keys()
    B = shift.read_predictions(args.base); C = shift.read_predictions(args.stack); R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv")
    idx = pd.Series(np.arange(len(keys)), index=pd.MultiIndex.from_arrays([keys["scenario"], keys["site"], keys["age"]]))
    cl = np.load(ROOT / "work" / "collapse_cluster_labels.npy") == 0
    bands = shift.band_masks(keys["age"].to_numpy())
    for root in args.roots:
        root = Path(root); df = load(root)
        if common is not None:
            df = df[df.site.isin(common)].reset_index(drop=True)
        sites = sorted(df.site.unique())
        print(f"\n=== {root} : {len(sites)} sites ===")
        # (i) y0 fit
        out = []
        for var, dec in (("height", 1.0), ("agb", 1.0), ("lai", 1.0)):
            est = df[f"y0est_{'height_m' if var == 'height' else 'agb_kgC' if var == 'agb' else 'lai'}"].to_numpy()
            y0 = df[f"y0_{var}"].to_numpy()
            m = (y0 > 0.05) & (est > 0.01)
            lr = np.log(est[m] / y0[m])
            out.append((var, np.sqrt((lr ** 2).mean()), lr.mean(), m.sum()))
        print("  (i) y0 fit, RMS log ratio (bias): " + "; ".join(f"{v} {r:.3f} ({b:+.3f}, n={n})" for v, r, b, n in out))
        for band, ages in (("young", [1, 10, 20, 30]), ("mid", [50, 70, 100]), ("old", [150, 200, 250, 300, 350, 400, 450, 500])):
            sub = df[df.age.isin(ages)]
            s = []
            for var, col in (("height", "y0est_height_m"), ("agb", "y0est_agb_kgC")):
                est, y0 = sub[col].to_numpy(), sub[f"y0_{var}"].to_numpy(); m = (y0 > 0.05) & (est > 0.01)
                lr = np.log(est[m] / y0[m]); s.append(f"{var} rms {np.sqrt((lr**2).mean()):.3f} bias {lr.mean():+.3f}")
            print(f"      {band:5s}: " + ", ".join(s))
        # scenario difference at ages 1-30
        y = df[df.age <= 30].pivot_table(index=["site", "age"], columns="scenario", values=["y0est_agb_kgC", "y0_agb", "y0est_height_m", "y0_height"])
        for var in ("agb", "height"):
            e = y[(f"y0est_{'agb_kgC' if var == 'agb' else 'height_m'}", "ssp585")] / y[(f"y0est_{'agb_kgC' if var == 'agb' else 'height_m'}", "ssp126")]
            o = y[(f"y0_{var}", "ssp585")] / y[(f"y0_{var}", "ssp126")]
            m = np.isfinite(e) & np.isfinite(o) & (y[(f"y0_{var}", "ssp126")] > 0.05)
            print(f"      scenario ratio 585/126 at ages 1-30, {var}: ED mean {np.nanmean(e[m]):.3f} sd {np.nanstd(e[m]):.3f} | y0 mean {np.nanmean(o[m]):.3f} sd {np.nanstd(o[m]):.3f} | corr {np.corrcoef(np.log(e[m]), np.log(o[m]))[0,1]:+.2f}")
        # (ii) predictions in truth units, two forms
        rows = idx.loc[list(zip(df.scenario, df.site, df.age))].to_numpy()
        sp_h, sp_a = df.spin_height_m.to_numpy(), df.spin_agb_kgC.to_numpy()
        y0_h, y0_a = R[rows, 0] * MH, R[rows, 1] * MA
        fac_h = np.where(sp_h > 0.05, np.clip(y0_h / np.maximum(sp_h, 1e-9), 0.5, 2.0), 1.0)
        fac_a = np.where(sp_a > 0.05, np.clip(y0_a / np.maximum(sp_a, 1e-9), 0.5, 2.0), 1.0)
        T_lf = np.column_stack([HS * df.ed40_height_m.to_numpy() * fac_h / MH, df.ed40_agb_kgC.to_numpy() * fac_a / MA])
        with np.errstate(divide="ignore", invalid="ignore"):
            T_ratio = np.column_stack([HS * R[rows, 0] * df.ed40_height_m.to_numpy() / sp_h, R[rows, 1] * df.ed40_agb_kgC.to_numpy() / sp_a])
        outc = ~cl[keys["site"].to_numpy()[rows]]
        veg = R[rows, 1] > 0.02
        print("  (ii) through-origin slopes on R, outside cluster, vegetated (targets old h 0.891 / a 0.888, mid h 0.941 / a ~1.05):")
        for band in ("young", "mid", "old"):
            m = bands[band][rows] & outc & veg
            print(f"      {band:5s} n={m.sum():5d}  level-factor h {slope(T_lf[m,0], R[rows[m],0]):.3f} a {slope(T_lf[m,1], R[rows[m],1]):.3f} | pure ratio h {slope(T_ratio[m,0], R[rows[m],0]):.3f} a {slope(T_ratio[m,1], R[rows[m],1]):.3f} | stack C h {slope(C[rows[m],0], R[rows[m],0]):.3f} a {slope(C[rows[m],1], R[rows[m],1]):.3f}")
        # (iii) contrasts
        print("  (iii) young growth vs the 10-10 stack B, (T-R)/(B-R) - 1 (targets +7% ssp585, -7% ssp126); old-agb contrast mean(T585-T126) (target ~+0.08):")
        s = df.scenario.to_numpy()
        for sc in ("ssp126", "ssp585"):
            m = bands["young"][rows] & outc & veg & (s == sc)
            g_t = (T_lf[m] - R[rows[m]]).sum(0); g_b = (B[rows[m]] - R[rows[m]]).sum(0)
            print(f"      {sc}: young growth ratio h {g_t[0] / g_b[0] - 1:+.3f} a {g_t[1] / g_b[1] - 1:+.3f}  (stack C: h {((C[rows[m]] - R[rows[m]]).sum(0)[0] / g_b[0]) - 1:+.3f} a {((C[rows[m]] - R[rows[m]]).sum(0)[1] / g_b[1]) - 1:+.3f})")
        piv = pd.DataFrame({"site": df.site, "age": df.age, "scenario": s, "Ta": T_lf[:, 1], "Th": T_lf[:, 0], "Ca": C[rows, 1], "out": outc, "veg": veg}).query("out and veg and age >= 150")
        if len(piv) and piv.scenario.nunique() == 2:
            p = piv.pivot_table(index=["site", "age"], columns="scenario", values=["Ta", "Ca", "Th"])
            d_ed = (p[("Ta", "ssp585")] - p[("Ta", "ssp126")]).mean(); d_c = (p[("Ca", "ssp585")] - p[("Ca", "ssp126")]).mean(); d_h = (p[("Th", "ssp585")] - p[("Th", "ssp126")]).mean()
            print(f"      old agb contrast 585-126: ED {d_ed:+.3f} (height {d_h:+.3f}) | stack C {d_c:+.3f}  (n pairs {len(p)})")
        else:
            print("      (no outside-cluster old rows in this set)")


if __name__ == "__main__":
    main()
