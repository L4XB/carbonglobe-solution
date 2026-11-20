#!/usr/bin/env python
"""Scale the values of a GROUP of site x scenarios by one factor (level probe on an arbitrary group; same cell rules as scripts/apply_level.py).

    python scripts/apply_group_level.py --inp results/x.csv --out results/y.csv --groups work/fire/groups_regime.csv --group "pos|low" --col a --factor 0.8
        [--ages 50:500] [--min 0.02]
The groups file has columns site, sc (ssp126 / ssp585), group. Heights: cells above --min (default 0.3) x factor, capped at 35 m (2.8122);
agb: (coded agb + floor) x factor - floor on cells with coded + floor above --min (default 0.02).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

HCAP_LEVEL = 2.8122
FL = 0.401412 / 3.39567


def group_mask(groups: str, group: str, ages: str = "") -> np.ndarray:
    """Rows (template order) of the site x scenarios that carry `group` in the csv, optionally restricted to a seed-age range lo:hi."""
    keys = shift.template_keys(); site = keys.site.to_numpy(); scen = keys.scenario.to_numpy(); age = keys.age.to_numpy()
    g = pd.read_csv(ROOT / groups); g = g[g.group == group]
    assert len(g) > 0, f"group {group!r} not in {groups}"
    code = set(zip(g.site.astype(int), g.sc.astype(str)))
    m = np.fromiter(((int(s), sc) in code for s, sc in zip(site, scen)), bool, len(site))
    if ages:
        lo, hi = (int(x) for x in ages.split(":")); m &= (age >= lo) & (age <= hi)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--groups", required=True); ap.add_argument("--group", required=True)
    ap.add_argument("--col", default="a", choices=("h", "a")); ap.add_argument("--factor", type=float, required=True)
    ap.add_argument("--ages", default=""); ap.add_argument("--min", type=float, default=-1.0)
    args = ap.parse_args()
    P = shift.read_predictions(ROOT / args.inp).astype(float); Q = P.copy()
    m = group_mask(args.groups, args.group, args.ages)
    if args.col == "h":
        mn = 0.3 if args.min < 0 else args.min; m &= P[:, 0] > mn
        Q[m, 0] = np.minimum(P[m, 0] * args.factor, HCAP_LEVEL); c = 0; val = P[m, 0]
    else:
        mn = 0.02 if args.min < 0 else args.min; m &= (P[:, 1] + FL) > mn
        Q[m, 1] = (P[m, 1] + FL) * args.factor - FL; c = 1; val = P[m, 1] + FL
    shift.write_predictions(Q, ROOT / args.out)
    print(f"{args.inp} -> {args.out}: group {args.group} [{args.groups}{' | ages ' + args.ages if args.ages else ''}] {'heights' if c == 0 else 'agb'} x {args.factor:g} on {int(m.sum())} cells, "
          f"S2 {np.sum(val ** 2):.0f}, energy {np.sum((Q - P) ** 2):.0f}, column mean {P[:, c].mean():.4f} -> {Q[:, c].mean():.4f}")


if __name__ == "__main__":
    main()
