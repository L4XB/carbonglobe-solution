#!/usr/bin/env python
"""Take back a share of the block LEVEL of a file's height changes against the stack; the within-block pattern stays.

Background (11-12, scripts/diag/height_level_fit.py): the 11-05 level / scale probes pin the first moment and the scale of the truth's heights
on the board; the ED-based height parts lifted the file's vegetated heights by 5700 solution units in total (+3.6 % of the column) and every
scored file together says that most of the block-mean lift of the mature rows is not there (cold zones: share to take back 0.76 +- 0.41), while the
site-to-site pattern of the same parts is right (direct contrasts CY / CM / H).

    d      = file height - stack height on the changed cells (|d| > 1e-9)
    block  = class | climate zone | seed-age band (1, 10-30, 50-100, 150-500) | scenario;   level = block mean of d
    file  <- file - remove[group] x level
    groups: age1 (fire-never / class b, seed age 1), cold (fire-never boreal + cool, ages >= 10), warm (fire-never warm / moist / dry, ages >= 10),
            b (class b, ages >= 10), pos (fire-possible sites = the forced-fire heights)

    python scripts/apply_hlevel.py --inp results/x.csv --out results/y.csv --remove cold=0.75,b=0.27,pos=0.15
A remove value above 1 or below 0 extrapolates (probes). Heights stay within [0, 35 m].
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
GROUPS = ("age1", "cold", "warm", "b", "pos")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--remove", required=True, help="group=share,... (groups: age1, cold, warm, b, pos; missing groups 0)")
    ap.add_argument("--stack", default="results/spatcorr_stack_20261110_s08.csv")
    args = ap.parse_args()
    rem = {g: 0.0 for g in GROUPS}
    for it in args.remove.split(","):
        if it:
            k, v = it.split("="); assert k in rem, f"unknown group {k}"; rem[k] = float(v)
    P = shift.read_predictions(ROOT / args.inp).astype(float); S = shift.read_predictions(ROOT / args.stack).astype(float)
    keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy(); scen = keys.scenario.to_numpy()
    cl = (np.load(ROOT / "work/collapse_cluster_labels.npy") == 0)[site]; mxf = np.load(ROOT / "work/fire/max_factor_rule833.npy")[site]
    cls = np.where(cl, "cluster", np.where(mxf >= 1e3, "pos", np.where(mxf >= 30, "b", "never"))); strat = np.asarray(de.strata_of_sites())[site]
    band = np.where(age == 1, "1", np.where(age <= 30, "y", np.where(age <= 100, "m", "o")))
    d = P[:, 0] - S[:, 0]; ch = np.abs(d) > 1e-9
    blk = (pd.Series(cls.astype(str)) + "|" + pd.Series(strat.astype(str)) + "|" + pd.Series(band) + "|" + pd.Series(scen.astype(str))).to_numpy()
    level = pd.Series(np.where(ch, d, np.nan)).groupby(blk).transform("mean").to_numpy(); level = np.where(ch, np.nan_to_num(level), 0.0)
    cold = np.isin(strat, ["boreal", "cool"]); nev = cls == "never"; nb = np.isin(cls, ["never", "b"]); a1 = band == "1"
    G = {"age1": nb & a1, "cold": nev & cold & ~a1, "warm": nev & ~cold & ~a1, "b": (cls == "b") & ~a1, "pos": cls == "pos"}
    Q = P.copy(); msg = []
    for g in GROUPS:
        m = G[g] & ch
        if rem[g] == 0.0 or not m.any():
            continue
        Q[m, 0] = np.clip(P[m, 0] - rem[g] * level[m], 0.0, HCAP)
        msg.append(f"{g} x {rem[g]:g}: {int(m.sum())} cells, level energy {np.sum(level[m] ** 2):.0f}, applied {np.sum((Q[m, 0] - P[m, 0]) ** 2):.0f}, sum {np.sum(Q[m, 0] - P[m, 0]):+.0f}")
    shift.write_predictions(Q, ROOT / args.out)
    print(f"{args.inp} -> {args.out}: height level taken back | " + " | ".join(msg) + f" | energy {np.sum((Q[:, 0] - P[:, 0]) ** 2):.0f}, column mean {P[:, 0].mean():.4f} -> {Q[:, 0].mean():.4f}")


if __name__ == "__main__":
    main()
