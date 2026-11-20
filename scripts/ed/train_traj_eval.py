#!/usr/bin/env python
"""Yearly trajectories on TRAINING cells: does ED's year-to-year height / agb growth follow the organisers', and with the same temperature sensitivity?

Needs a training root run with ED_SAVE_TRAJ=1 (traj/<cell>_<scenario>_a<age>.csv, December rows h_m12 / agb_m12 of window years 0..40).
Per cell x seed age: yearly log increments of the truth (io.open_train_y, December) and of ED; corr and slope over the 40 years; the regression
of both on the year's temperature anomaly (mean monthly air temperature of the months above 0 C, training X channel 0) - a truth-based check of
the WARMING response that the 40-year totals cannot give (the lookup cycles the same 40 years, so the window total is a pure CO2 contrast).

    python scripts/ed/train_traj_eval.py work/edv3/train_hv2 [--cells work/edv3/train_hv_cells.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io  # noqa: E402

AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
BANDS = (("10-30", 10, 30), ("50-100", 50, 100), ("150-500", 150, 500))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--cells", default="work/edv3/train_hv_cells.json")
    args = ap.parse_args()
    root = Path(args.root)
    zone = {}
    if Path(args.cells).exists():
        for st, lst in json.load(open(args.cells)).items():
            for _, c in lst:
                zone[int(c)] = st
    X = io.open_train_x()
    Y = {a: io.open_train_y(a) for a in AGES}
    rows = []
    for f in sorted((root / "traj").glob("*_ssp126_a*.csv")):
        cell = int(f.name.split("_")[0]); a = int(f.stem.split("_a")[1])
        t = pd.read_csv(f)
        if len(t) < 41 or a not in Y:
            continue
        tm = np.asarray(X[cell, :, :, 0], dtype=np.float64)                     # (40, 12) monthly mean air temperature
        tm = tm - 273.15 if np.nanmean(tm) > 100 else tm
        warm = tm.mean(axis=0) > 0.0                                            # months above 0 C in the 40-year mean
        if warm.sum() == 0:
            continue
        tg = tm[:, warm].mean(axis=1); dT = tg - tg.mean()                      # growing-season temperature anomaly per window year (K)
        rec = dict(cell=cell, age=a, zone=zone.get(cell, "other"), sdT=float(dT.std()))
        for c, col, v, eps in (("h", "h_m12", 0, 0.1), ("a", "agb_m12", 1, 0.01)):
            yt = np.asarray(Y[a][cell, :41, 11, v], dtype=np.float64); ye = t[col].to_numpy(dtype=float)[:41]
            dt_ = np.diff(np.log(yt + eps)); de_ = np.diff(np.log(np.maximum(ye, 0.0) + eps))       # increments of years 1..40
            n = min(len(dt_), len(dT))
            dt_, de_, x = dt_[:n], de_[:n], dT[:n]
            ok = np.isfinite(dt_) & np.isfinite(de_)
            if ok.sum() < 30 or dt_[ok].std() == 0 or de_[ok].std() == 0:
                continue
            rec[f"corr_{c}"] = float(np.corrcoef(dt_[ok], de_[ok])[0, 1]); rec[f"sd_t_{c}"] = float(dt_[ok].std()); rec[f"sd_e_{c}"] = float(de_[ok].std())
            rec[f"sT_t_{c}"] = float(np.polyfit(x[ok], dt_[ok], 1)[0]); rec[f"sT_e_{c}"] = float(np.polyfit(x[ok], de_[ok], 1)[0])   # log growth per K
            rec[f"tot_t_{c}"] = float(np.log((yt[40] + eps) / (yt[0] + eps))); rec[f"tot_e_{c}"] = float(np.log((max(ye[40], 0.0) + eps) / (max(ye[0], 0.0) + eps)))
        rows.append(rec)
    d = pd.DataFrame(rows)
    if d.empty:
        print("no trajectories"); return
    d.to_csv(root / "traj_eval_rows.csv", index=False)
    print(f"{root}: {d.cell.nunique()} cells, {len(d)} cell x age trajectories")
    for c, cn in (("h", "HEIGHT"), ("a", "AGB")):
        print(f"\n{cn}: yearly log increments, truth against ED (medians over cell x age)")
        for z in sorted(d.zone.unique()):
            for bn, lo, hi in BANDS:
                g = d[(d.zone == z) & (d.age >= lo) & (d.age <= hi)].dropna(subset=[f"corr_{c}"])
                if len(g) < 5:
                    continue
                st, se = g[f"sT_t_{c}"], g[f"sT_e_{c}"]
                cs = np.corrcoef(st, se)[0, 1] if len(g) > 5 else float("nan")
                print(f"   {z:6s} {bn:8s} n {len(g):4d} | yearly corr {g[f'corr_{c}'].median():+.2f} | sd of increments truth {g[f'sd_t_{c}'].median():.4f} ED {g[f'sd_e_{c}'].median():.4f} | "
                      f"growth per K of a warm year: truth {st.median():+.4f} ED {se.median():+.4f} (means {st.mean():+.4f} {se.mean():+.4f}; corr across rows {cs:+.2f}; "
                      f"slope ED->truth {np.polyfit(se, st, 1)[0]:+.2f}) | 40-year total truth {g[f'tot_t_{c}'].median():+.3f} ED {g[f'tot_e_{c}'].median():+.3f}")


if __name__ == "__main__":
    main()
