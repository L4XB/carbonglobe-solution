#!/usr/bin/env python
"""Training-world check of the chronosequence decode (scripts/ed/dec2.py) for HEIGHT and AGB.

The test pipeline's ED configuration (cycled 40-year lookup at 415 ppm, no fire) is run on TRAINING cells
(scripts/ed/train_site.py with CG_OVERRIDE, see notes 11-10 23:50 UTC); there the organisers' year-40 state is known.
Per cell x seed age (December rows, one scenario - both are identical inputs):

    truth response  r_t = log(truth40 / truth-y0 profile at age + 40)
    ED response     r_e = log(ED40 / ED first-record profile at age + 40)
    decode error    r_t - r_e  (the decode predicts truth40 = y0 profile at age + 40 x exp(r_e))

Printed per zone x age band: medians of r_t and r_e, corr, slope of r_t on r_e, rms of the decode error against the rms of r_t
(= the error of the chronosequence alone), with and without the seed gates of the test pipeline.

    python scripts/ed/train_height_eval.py work/edv3/train_hv1 [--cells work/edv3/train_hv_cells.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AGES = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]); LA = np.log(AGES)
EPS = {"h": 0.1, "a": 0.01}
BANDS = (("age 1", 1, 1), ("10-30", 10, 30), ("50-100", 50, 100), ("150-500", 150, 500))


def load(root: Path) -> pd.DataFrame:
    """One row per cell x seed age with truth (y0, y40) and ED (e0 = first record, e40) for height and agb, physical units."""
    rows = []
    for f in sorted((root / "csv").glob("site*.csv")):
        t = pd.read_csv(f)
        t = t[t.scenario == t.scenario.iloc[0]].drop_duplicates("age").set_index("age").reindex(AGES)
        cell = int(f.stem[4:])
        tf = root / "truth" / f"{cell}.csv"
        if not tf.exists() or not np.isfinite(t[["end_h_m12", "end_agb_m12", "pre_h_m12", "pre_agb_m12"]].to_numpy(dtype=float)).all():
            continue
        y = pd.read_csv(tf).set_index("age").reindex(AGES)
        d = pd.DataFrame(dict(cell=cell, age=AGES, y0_h=y.y0_height.to_numpy(), y40_h=y.y40_height.to_numpy(), y0_a=y.y0_agb.to_numpy(), y40_a=y.y40_agb.to_numpy(),
                              e0_h=t.pre_h_m12.to_numpy(), e40_h=t.end_h_m12.to_numpy(), e0_a=t.pre_agb_m12.to_numpy(), e40_a=t.end_agb_m12.to_numpy()))
        x = np.log(np.minimum(AGES + 40, 500))
        for c in ("h", "a"):
            e = EPS[c]
            yc = np.exp(np.interp(x, LA, np.log(d[f"y0_{c}"].to_numpy() + e))); ec = np.exp(np.interp(x, LA, np.log(np.maximum(d[f"e0_{c}"].to_numpy(), 0.0) + e)))
            d[f"rt_{c}"] = np.log((d[f"y40_{c}"].to_numpy() + e) / yc); d[f"re_{c}"] = np.log((np.maximum(d[f"e40_{c}"].to_numpy(), 0.0) + e) / ec)
            mis = np.abs(np.log((d[f"y0_{c}"].to_numpy() + e) / (np.maximum(d[f"e0_{c}"].to_numpy(), 0.0) + e)))
            d[f"mis_{c}"] = mis
            jb = np.clip(np.searchsorted(LA, x, side="right") - 1, 0, len(LA) - 2)
            d[f"mis2_{c}"] = np.maximum(mis[jb], mis[jb + 1])
            d[f"chrono_{c}"] = yc - e
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def stats(d: pd.DataFrame, c: str) -> str:
    rt, re = d[f"rt_{c}"].to_numpy(), d[f"re_{c}"].to_numpy()
    if len(d) < 8:
        return f"n {len(d):4d}"
    cc = np.corrcoef(rt, re)[0, 1] if rt.std() > 0 and re.std() > 0 else float("nan")
    sl = float(np.polyfit(re, rt, 1)[0]) if re.std() > 0 else float("nan")
    return (f"n {len(d):4d} cells {d.cell.nunique():3d} | truth {np.median(rt):+.3f} ED {np.median(re):+.3f} (means {rt.mean():+.3f} {re.mean():+.3f}) | corr {cc:+.2f} slope {sl:+.2f} | "
            f"rms chrono alone {np.sqrt(np.mean(rt ** 2)):.3f} decode {np.sqrt(np.mean((rt - re) ** 2)):.3f} decode at half the ED response {np.sqrt(np.mean((rt - 0.5 * re) ** 2)):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--cells", default="work/edv3/train_hv_cells.json")
    ap.add_argument("--gate-h", type=float, default=0.12)
    ap.add_argument("--gate-h2", type=float, default=0.15)
    args = ap.parse_args()
    d = load(Path(args.root))
    if d.empty:
        print("no finished cells"); return
    zone = {}
    if Path(args.cells).exists():
        for st, lst in json.load(open(args.cells)).items():
            for _, c in lst:
                zone[int(c)] = st
    d["zone"] = [zone.get(int(c), "other") for c in d.cell]
    print(f"{args.root}: {d.cell.nunique()} cells; by zone: " + ", ".join(f"{k} {v}" for k, v in d.groupby('zone').cell.nunique().items()))
    print(f"seed match (first record against the truth's y0), median |log ratio| by zone: " +
          "; ".join(f"{z}: h {g.mis_h.median():.3f} a {g.mis_a.median():.3f}" for z, g in d.groupby("zone")))
    for c, cn in (("h", "HEIGHT"), ("a", "AGB")):
        print(f"\n{cn}: response against the own chronosequence at age + 40")
        for z in sorted(d.zone.unique()):
            for bn, lo, hi in BANDS:
                g = d[(d.zone == z) & (d.age >= lo) & (d.age <= hi)]
                gg = g[(g.mis_h < args.gate_h) & (g.mis2_h < args.gate_h2)] if c == "h" else g[g.mis_a < np.where(g.age >= 200, 0.25, 0.12)]
                if bn == "age 1":
                    gg = g
                print(f"   {z:6s} {bn:8s} all:   {stats(g, c)}")
                print(f"   {z:6s} {bn:8s} gated: {stats(gg, c)}")
    # height response per unit of agb response (mature rows), truth against ED
    print("\nmature rows (seed ages 150-500): height response per agb response, per zone (ratio of medians; slope across rows)")
    for z in sorted(d.zone.unique()):
        g = d[(d.zone == z) & (d.age >= 150) & (d.mis_h < args.gate_h) & (d.mis2_h < args.gate_h2)]
        if len(g) < 8:
            continue
        print(f"   {z:6s} n {len(g):4d}: truth h {np.median(g.rt_h):+.4f} / a {np.median(g.rt_a):+.4f} = {np.median(g.rt_h) / np.median(g.rt_a):+.2f}; ED h {np.median(g.re_h):+.4f} / a {np.median(g.re_a):+.4f} = "
              f"{np.median(g.re_h) / np.median(g.re_a):+.2f}; slope h on a: truth {np.polyfit(g.rt_a, g.rt_h, 1)[0]:+.2f}, ED {np.polyfit(g.re_a, g.re_h, 1)[0]:+.2f}")
    d.to_csv(Path(args.root) / "height_eval_rows.csv", index=False)


if __name__ == "__main__":
    main()
