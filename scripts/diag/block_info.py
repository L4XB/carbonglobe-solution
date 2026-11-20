#!/usr/bin/env python
"""Block table for level / contrast probes: S2 of the block (all rows), eps (sampling error of the block's PUBLIC energy for a random half of the
sites), reading error of ONE probe at its step s and at the best step, and the value of the reading for a prior sd of the block's level error.

    python scripts/diag/block_info.py results/cand50_A.csv "lvl:a:0.85:never:warm" "lvl:h:0.85:pos:dry" "con:h:0.08:never+b:*:50-500" [--prior 0.015]

Names as the generic block probes of scripts/make_spec_next.py: lvl|con : h|a : factor (lvl) or s (con) : classes : zones [: agelo-agehi];
'+' separates classes / zones, '*' = all. Level probe: values x factor (s = 1 - factor); contrast probe: x (1 + s) on ssp585, x (1 - s) on ssp126 rows.
Reading error on delta = 151 / (2 s S2) (rounding of two scores) (+) (s / 2) eps (public energy); best s = sqrt(151 / (S2 eps)).
Value = S2 x (prior^2 - posterior^2) squared units; add `--json` to print entries for work/logs/dayNN_probes.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "ed"))
from cgb import shift  # noqa: E402
import dec_eval as de  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from apply_contrast import pair_mask  # noqa: E402

FL = 0.401412 / 3.39567
RND = 151.0                                   # sd of the difference of two rounded scores, squared units


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cand"); ap.add_argument("names", nargs="+")
    ap.add_argument("--prior", type=float, default=0.015); ap.add_argument("--json", action="store_true")
    ap.add_argument("--boards", default="", help="null boards whose truth base is <cand> (scripts/sim/synth_board.py), comma list: adds sigma_h = rms of "
                    "the block's EXACT public reading of pure site noise (what a reading carries that the private half does not share) and the "
                    "private value S2 tau^4 / (tau^2 + sigma_h^2 + err^2)")
    args = ap.parse_args()
    A = shift.read_predictions(ROOT / args.cand).astype(np.float64)
    boards = []
    for b in [x for x in args.boards.split(",") if x]:
        pub = np.load(ROOT / (b + "_public.npy")); T = np.load(ROOT / (b + "_truth.npy")).astype(np.float64)
        boards.append((T - A, pub if pub.shape[0] == A.shape[0] else pub[shift.template_keys().site.to_numpy()]))
    keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy()
    cl = np.load(ROOT / "work/collapse_cluster_labels.npy") == 0; mxf = np.load(ROOT / "work/fire/max_factor_rule833.npy")
    cls = np.where(cl, "cluster", np.where(mxf >= 1e3, "pos", np.where(mxf >= 30, "b", "never")))[site]
    strat = np.asarray(de.strata_of_sites())[site]
    out = {}
    print(f"{'probe':34s} {'sites':>5s} {'cells':>6s} {'S2':>8s} {'eps %':>6s} | {'s':>5s} {'energy':>7s} {'err %':>6s} | best s {'err %':>6s} | value at prior {100 * args.prior:.1f} %")
    for nm in args.names:
        f = nm.split(":"); assert f[0] in ("lvl", "con") and len(f) in (5, 6), nm
        col = 0 if f[1] == "h" else 1; s = abs(1.0 - float(f[2])) if f[0] == "lvl" else float(f[2])
        m = np.ones(len(A), bool)
        if f[3] != "*":
            m &= np.isin(cls, f[3].split("+"))
        if f[4] != "*":
            m &= np.isin(strat, f[4].split("+"))
        if len(f) == 6:
            lo, hi = (int(x) for x in f[5].split("-")); m &= (age >= lo) & (age <= hi)
        val = A[:, 0] if col == 0 else A[:, 1] + FL
        m &= val > (0.3 if col == 0 else 0.02)
        if f[0] == "con" and col == 0:                     # height contrast probes use only pairs below the 35 m cap (scripts/apply_contrast.py --pair-max)
            m = pair_mask(m, val, site, age, keys.scenario.to_numpy(), round(35.0 / 1.14114 / 10.90645 / (1.0 + s) - 1e-6, 6))
        e = np.bincount(site[m], weights=val[m] ** 2, minlength=6171); e = e[e > 0]
        S2 = float(e.sum()); eps = float(np.sqrt(np.sum(e ** 2)) / S2)
        err = float(np.hypot(RND / (2 * s * S2), s / 2 * eps)); sb = float(np.sqrt(RND / (S2 * eps))); errb = float(np.hypot(RND / (2 * sb * S2), sb / 2 * eps))
        post2 = 1.0 / (1.0 / args.prior ** 2 + 1.0 / err ** 2); value = S2 * (args.prior ** 2 - post2)
        line = f"{nm:34s} {len(e):5d} {int(m.sum()):6d} {S2:8.0f} {100 * eps:6.2f} | {s:5.2f} {s * s * S2:7.0f} {100 * err:6.2f} | {sb:6.2f} {100 * errb:6.2f} | {value:6.1f}"
        if boards:
            sgn = -np.ones(len(A)) if f[0] == "lvl" else np.where(keys.scenario.to_numpy() == "ssp585", 1.0, -1.0)
            u = np.zeros(len(A)); u[m] = sgn[m] * val[m]
            rd_ = [float(np.sum(u[pb] * R[pb, col]) / np.sum(u[pb] ** 2)) for R, pb in boards]       # exact public reading of the noise
            sh = float(np.sqrt(np.mean(np.square(rd_)))); tau2 = args.prior ** 2
            line += f" | sigma_h {100 * sh:5.2f} %  private value {S2 * tau2 * tau2 / (tau2 + sh * sh + err * err):6.1f}"
        print(line)
        safe = nm.replace(":", "_").replace("+", "-").replace("*", "all").replace(".", "")
        out[safe] = {"type": "level" if f[0] == "lvl" else "contrast", "s": round(s, 4), "block": safe, "prior": [0.0, args.prior], "eps": round(eps, 4)}
    if args.json:
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
