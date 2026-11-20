#!/usr/bin/env python
"""Scale the vegetated heights of a file in chosen classes / climate zones by one factor (a level step, e.g. x 0.976 outside the boreal zone).

    python scripts/apply_level.py --inp results/x.csv --out results/y.csv --factor 0.976 [--classes never,b,pos,cluster] [--strata warm,moist,dry,cool,cluster]
Only cells above --min (0.3 solution units = 3.7 m) are touched; the 35 m cap (2.8122) is kept.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--factor", type=float, required=True)
    ap.add_argument("--classes", default="never,b,pos,cluster")
    ap.add_argument("--strata", default="warm,moist,dry,cool,cluster")
    ap.add_argument("--min", type=float, default=0.3)
    ap.add_argument("--ages", default="", help="seed-age range lo:hi of the rows that are touched (default all)")
    ap.add_argument("--col", default="h", choices=("h", "a"), help="h = heights (default); a = agb: (coded value + floor 0.1182) x factor on cells with coded + floor above --min")
    args = ap.parse_args()
    P = shift.read_predictions(ROOT / args.inp).astype(float)
    keys = shift.template_keys(); site = keys.site.to_numpy()
    cl = (np.load(ROOT / "work/collapse_cluster_labels.npy") == 0)[site]; mxf = np.load(ROOT / "work/fire/max_factor_rule833.npy")[site]
    cls = np.where(cl, "cluster", np.where(mxf >= 1e3, "pos", np.where(mxf >= 30, "b", "never"))); strat = de.strata_of_sites()[site]
    FL = 0.401412 / 3.39567
    sel = np.ones(len(site), bool)
    if args.ages:
        lo, hi = (int(x) for x in args.ages.split(":")); age = keys.age.to_numpy(); sel = (age >= lo) & (age <= hi)
    if args.col == "a":
        m = np.isin(cls, args.classes.split(",")) & np.isin(strat, args.strata.split(",")) & (P[:, 1] + FL > args.min) & sel
        Q = P.copy(); Q[m, 1] = (P[m, 1] + FL) * args.factor - FL
        shift.write_predictions(Q, ROOT / args.out)
        print(f"{args.inp} -> {args.out}: agb x {args.factor} on {int(m.sum())} cells, energy {np.sum((Q[:, 1] - P[:, 1]) ** 2):.0f}, column mean {P[:, 1].mean():.4f} -> {Q[:, 1].mean():.4f}")
        return
    m = np.isin(cls, args.classes.split(",")) & np.isin(strat, args.strata.split(",")) & (P[:, 0] > args.min) & sel
    Q = P.copy(); Q[m, 0] = np.minimum(P[m, 0] * args.factor, 2.8122)
    shift.write_predictions(Q, ROOT / args.out)
    print(f"{args.inp} -> {args.out}: heights x {args.factor} on {int(m.sum())} cells, energy {np.sum((Q[:, 0] - P[:, 0]) ** 2):.0f}, column mean {P[:, 0].mean():.4f} -> {Q[:, 0].mean():.4f}")


if __name__ == "__main__":
    main()
