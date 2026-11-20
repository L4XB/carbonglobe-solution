#!/usr/bin/env python
"""Board calibration of a file in ONE call (post step of scripts/design_run.py): first the height-level step of scripts/apply_hlevel.py, then a
list of block level factors with the semantics of scripts/apply_level.py.

    --remove cold=0.8,b=0.35,pos=0.15          share of the block level of the height changes against the stack that is taken back (apply_hlevel)
    --levels "h:0.972:never,b:warm,moist,dry;a:0.971:pos:*:0.02:150-500;a:1.026:never:boreal,cool:0.02"
              items col:factor:classes:strata[:min[:agelo-agehi]]   ('*' = all classes / all zones)
              h: heights x factor on cells above min (default 0.3), capped at 35 m;  a: (coded agb + floor) x factor on cells with coded + floor above min

    python scripts/apply_cal.py --inp results/x.csv --out results/y.csv --remove cold=0.8 --levels "h:0.972:never,b:warm,moist,dry"
The values come from the joint fit of all scored board files (scripts/diag/height_level_fit.py, 11-12).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "ed"))
from cgb import shift  # noqa: E402
import dec_eval as de  # noqa: E402

HCAP = 35.0 / 1.14114 / 10.90645
HCAP_LEVEL = 2.8122                       # the cap value used by apply_level.py
FL = 0.401412 / 3.39567
GROUPS = ("age1", "cold", "warm", "b", "pos")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--remove", default="", help="group=share,... (groups: age1, cold, warm, b, pos)")
    ap.add_argument("--levels", default="", help="col:factor:classes:strata[:min[:agelo-agehi]];...")
    ap.add_argument("--stack", default="results/spatcorr_stack_20261110_s08.csv")
    args = ap.parse_args()
    P = shift.read_predictions(ROOT / args.inp).astype(float); S = shift.read_predictions(ROOT / args.stack).astype(float)
    keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy(); scen = keys.scenario.to_numpy()
    cl = (np.load(ROOT / "work/collapse_cluster_labels.npy") == 0)[site]; mxf = np.load(ROOT / "work/fire/max_factor_rule833.npy")[site]
    cls = np.where(cl, "cluster", np.where(mxf >= 1e3, "pos", np.where(mxf >= 30, "b", "never"))); strat = np.asarray(de.strata_of_sites())[site]
    Q = P.copy(); msg = []
    # 1. height level (identical to apply_hlevel.py)
    rem = {g: 0.0 for g in GROUPS}
    for it in args.remove.split(","):
        if it:
            k, v = it.split("="); assert k in rem, f"unknown group {k}"; rem[k] = float(v)
    if any(v != 0.0 for v in rem.values()):
        band = np.where(age == 1, "1", np.where(age <= 30, "y", np.where(age <= 100, "m", "o")))
        d = P[:, 0] - S[:, 0]; ch = np.abs(d) > 1e-9
        blk = (pd.Series(cls.astype(str)) + "|" + pd.Series(strat.astype(str)) + "|" + pd.Series(band) + "|" + pd.Series(scen.astype(str))).to_numpy()
        level = pd.Series(np.where(ch, d, np.nan)).groupby(blk).transform("mean").to_numpy(); level = np.where(ch, np.nan_to_num(level), 0.0)
        cold = np.isin(strat, ["boreal", "cool"]); nev = cls == "never"; nb = np.isin(cls, ["never", "b"]); a1 = band == "1"
        G = {"age1": nb & a1, "cold": nev & cold & ~a1, "warm": nev & ~cold & ~a1, "b": (cls == "b") & ~a1, "pos": cls == "pos"}
        for g in GROUPS:
            m = G[g] & ch
            if rem[g] == 0.0 or not m.any():
                continue
            Q[m, 0] = np.clip(P[m, 0] - rem[g] * level[m], 0.0, HCAP)
            msg.append(f"h level {g} x {rem[g]:g}: {int(m.sum())} cells, energy {np.sum((Q[m, 0] - P[m, 0]) ** 2):.0f}")
    # 2. block level factors (identical to successive calls of apply_level.py)
    for it in [x for x in args.levels.split(";") if x]:
        f = it.split(":"); col = f[0]; fac = float(f[1])
        mc = np.ones(len(site), bool) if f[2] == "*" else np.isin(cls, f[2].split(","))
        ms = np.ones(len(site), bool) if f[3] == "*" else np.isin(strat, f[3].split(","))
        mn = float(f[4]) if len(f) > 4 and f[4] != "" else 0.3
        sel = np.ones(len(site), bool)
        if len(f) > 5 and f[5]:
            lo, hi = (int(x) for x in f[5].split("-")); sel = (age >= lo) & (age <= hi)
        before = Q.copy()
        if col == "a":
            m = mc & ms & (Q[:, 1] + FL > mn) & sel
            Q[m, 1] = (Q[m, 1] + FL) * fac - FL
        else:
            m = mc & ms & (Q[:, 0] > mn) & sel
            Q[m, 0] = np.minimum(Q[m, 0] * fac, HCAP_LEVEL)
        msg.append(f"{col} x {fac:g} [{f[2]} | {f[3]}{' | ages ' + f[5] if len(f) > 5 and f[5] else ''}]: {int(m.sum())} cells, energy {np.sum((Q - before) ** 2):.0f}")
    shift.write_predictions(Q, ROOT / args.out)
    print(f"{args.inp} -> {args.out}: calibration | " + " | ".join(msg) + f" | total energy {np.sum((Q - P) ** 2):.0f}, column means {P[:, 0].mean():.4f} {P[:, 1].mean():.4f} -> {Q[:, 0].mean():.4f} {Q[:, 1].mean():.4f}")


if __name__ == "__main__":
    main()
