#!/usr/bin/env python
"""Scale the SCENARIO CONTRAST of a block: values x (1 + s) on ssp585 rows and x (1 - s) on ssp126 rows (post step of scripts/design_run.py; probes).

A probe with s reads delta = half the relative error of the file's ssp585 - ssp126 difference on the block: S_probe - S_base = E (1 - 2 delta / s) with
E = s^2 x sum of squares of the block (heights: coded value on cells above --min; agb: coded + floor). Old probes leave the contrast of the
fire-possible heights (+1.4 +- 2.3 %) and of the mature fire-never heights (+0.4 +- 2.3 %) open (scripts/diag/level_vs_probes.py --blocks scen).

    python scripts/apply_contrast.py --inp results/x.csv --out results/y.csv --col h --s 0.08 --classes pos [--strata warm,moist,dry,cool] [--ages 150:500]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "ed"))
from cgb import shift  # noqa: E402
import dec_eval as de  # noqa: E402

HCAP = 35.0 / 1.14114 / 10.90645
FL = 0.401412 / 3.39567


def pair_mask(m: np.ndarray, val: np.ndarray, site: np.ndarray, age: np.ndarray, scen: np.ndarray, vmax: float) -> np.ndarray:
    """Cells of mask m whose partner row (same site and seed age, other scenario) is in m too, both values <= vmax."""
    key = site.astype(np.int64) * 1000 + age.astype(np.int64)
    i5 = np.where(scen == "ssp585")[0]; i1 = np.where(scen != "ssp585")[0]
    i5 = i5[np.argsort(key[i5], kind="stable")]; i1 = i1[np.argsort(key[i1], kind="stable")]
    assert len(i5) == len(i1) and np.array_equal(key[i5], key[i1]), "scenario rows do not pair"
    ok = m[i5] & m[i1] & (val[i5] <= vmax) & (val[i1] <= vmax)
    out = np.zeros(len(m), bool); out[i5[ok]] = True; out[i1[ok]] = True
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--col", default="h", choices=("h", "a")); ap.add_argument("--s", type=float, required=True)
    ap.add_argument("--classes", default="never,b,pos,cluster"); ap.add_argument("--strata", default="")
    ap.add_argument("--ages", default="", help="seed-age range lo:hi (default all)"); ap.add_argument("--min", type=float, default=-1.0)
    ap.add_argument("--pair-max", type=float, default=0.0, help="keep a cell only if it AND its partner (same site and seed age, other scenario) are in the "
                    "block and both values are <= this (e.g. 35 m cap / (1 + s): no cell can reach the cap, the probe stays a pure contrast)")
    args = ap.parse_args()
    P = shift.read_predictions(ROOT / args.inp).astype(float)
    keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy(); scen = keys.scenario.to_numpy()
    cl = (np.load(ROOT / "work/collapse_cluster_labels.npy") == 0)[site]; mxf = np.load(ROOT / "work/fire/max_factor_rule833.npy")[site]
    cls = np.where(cl, "cluster", np.where(mxf >= 1e3, "pos", np.where(mxf >= 30, "b", "never"))); strat = np.asarray(de.strata_of_sites())[site]
    m = np.isin(cls, args.classes.split(","))
    if args.strata:
        m &= np.isin(strat, args.strata.split(","))
    if args.ages:
        lo, hi = (int(x) for x in args.ages.split(":")); m &= (age >= lo) & (age <= hi)
    sgn = np.where(scen == "ssp585", 1.0, -1.0); Q = P.copy()
    if args.col == "h":
        mn = 0.3 if args.min < 0 else args.min; m &= P[:, 0] > mn; c = 0
    else:
        mn = 0.02 if args.min < 0 else args.min; m &= (P[:, 1] + FL) > mn; c = 1
    if args.pair_max > 0:
        m = pair_mask(m, P[:, c] + (FL if c == 1 else 0.0), site, age, scen, args.pair_max)
    if c == 0:
        Q[m, 0] = np.clip(P[m, 0] * (1.0 + args.s * sgn[m]), 0.0, np.maximum(HCAP, P[m, 0]))
    else:
        Q[m, 1] = (P[m, 1] + FL) * (1.0 + args.s * sgn[m]) - FL
    shift.write_predictions(Q, ROOT / args.out)
    val = P[m, c] + (FL if c == 1 else 0.0)
    print(f"{args.inp} -> {args.out}: scenario contrast {'heights' if c == 0 else 'agb'} x (1 +- {args.s:g}) on {int(m.sum())} cells [{args.classes} | {args.strata or 'all zones'}"
          f"{' | ages ' + args.ages if args.ages else ''}], energy {np.sum((Q - P) ** 2):.0f} (s^2 S2 = {args.s ** 2 * np.sum(val ** 2):.0f}), column means {P[:, c].mean():.4f} -> {Q[:, c].mean():.4f}")


if __name__ == "__main__":
    main()
