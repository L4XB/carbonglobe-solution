#!/usr/bin/env python
"""Level of the truth on GROUPS of site x scenarios, read from all scored files, with the same fit on null boards.

For every group g (column `group` of a csv with site, sc) the direction is -value of the candidate on the group's cells (agb + floor above 0.02
or heights above 0.3, seed ages in a range), i.e. truth = candidate x (1 - delta) on the group. Printed: S2, the single-direction fit of delta on
the real board (nominal error = rounding only) and the same fit on null boards (truth = candidate + site noise, true delta 0): the scatter there
is what the nominal error misses. A group is interesting when the real delta stands clear of the null values.

    python scripts/diag/group_levels.py results/cand50_A.csv work/fire/groups_x.csv [--col a] [--ages 50-500] [--null work/sim/synth5,work/sim/synth6,...]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "fire"))
from cgb import shift  # noqa: E402

FL = 0.401412 / 3.39567
SD = 0.00032


def fit_all(dirs: dict[str, np.ndarray], cand: np.ndarray, scores: str | None) -> dict[str, tuple[float, float]]:
    if scores:
        os.environ["CA_SCORES"] = scores
    else:
        os.environ.pop("CA_SCORES", None)
    import importlib
    import consistency_all as ca
    importlib.reload(ca)
    L = ca.load(); names, sc, cen, D, CC, rho = (L[k] for k in ("names", "sc", "cen", "D", "CC", "rho"))
    nrm = rho / (D.shape[1] * 2); wrow = ca.row_weights(L); DW = D * wrow
    E = np.einsum("nrc,nrc->n", DW, D) * nrm; base = np.einsum("nrc,nrc->n", DW, CC) * nrm
    s_p = np.array([sc[n] for n in names]); s_c = np.array([sc[Path(cen[n]).name] for n in names])
    resid = (E - (s_p - s_c)) / 2 - base; ok = E > 2e-5
    C = shift.read_predictions(ROOT / ca.C_PATH).astype(np.float64)
    r = (resid - np.einsum("nrc,rc->n", DW, (cand - C).astype(np.float32)) * nrm)[ok]
    out = {}
    for k, d in dirs.items():
        x = (np.einsum("nrc,rc->n", DW, d.astype(np.float32)) * nrm)[ok]; xx = float(x @ x)
        out[k] = (float(x @ r) / xx, SD / np.sqrt(xx)) if xx > 0 else (np.nan, np.nan)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cand"); ap.add_argument("groups"); ap.add_argument("--col", default="a", choices=("a", "h")); ap.add_argument("--ages", default="50-500")
    ap.add_argument("--null", default="work/sim/synth5,work/sim/synth6,work/sim/synth7,work/sim/synth8")
    args = ap.parse_args()
    A = shift.read_predictions(ROOT / args.cand).astype(np.float64)
    keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy(); scen = keys.scenario.to_numpy()
    g = pd.read_csv(ROOT / args.groups); lo, hi = (int(x) for x in args.ages.split("-"))
    col = 0 if args.col == "h" else 1; val = A[:, 0] if col == 0 else A[:, 1] + FL
    base = (age >= lo) & (age <= hi) & (val > (0.3 if col == 0 else 0.02))
    code = {(int(s), sc): gr for s, sc, gr in zip(g.site, g.sc, g.group)}
    grp = np.array([code.get((int(s), sc), "") for s, sc in zip(site, scen)])
    dirs = {}; info = {}
    for gr in sorted(set(g.group)):
        m = base & (grp == gr)
        if m.sum() < 20:
            continue
        d = np.zeros_like(A); d[m, col] = -val[m]; dirs[gr] = d
        e = np.bincount(site[m], weights=val[m] ** 2, minlength=6171); e = e[e > 0]
        info[gr] = (int(m.sum()), len(e), float(np.sum(val[m] ** 2)), float(val[m].mean()))
    real = fit_all(dirs, A, None)
    nulls = [fit_all(dirs, A, b + "_scores.json") for b in args.null.split(",") if b]
    # exact noise projections on the null boards' public halves (what a direct probe would read there)
    exact = {k: [] for k in dirs}
    for b in [x for x in args.null.split(",") if x]:
        T = np.load(ROOT / (b + "_truth.npy")).astype(np.float64); pub = np.load(ROOT / (b + "_public.npy")); pub = pub if pub.shape[0] == A.shape[0] else pub[site]
        for k, d in dirs.items():
            u = -d[:, col]; exact[k].append(float(np.sum((u * (A[:, col] - T[:, col]))[pub]) / np.sum((u * u)[pub])))
    print(f"{'group':28s} {'cells':>6s} {'sites':>5s} {'S2':>8s} {'mean':>6s} | real delta +- nominal | old-file fit on null boards | exact public delta of the noise (rms)")
    for k in dirs:
        n, ns, S2, mean = info[k]; d_, se = real[k]
        nl = " ".join(f"{100 * x[k][0]:+6.1f}" for x in nulls); ex = float(np.sqrt(np.mean(np.square(exact[k]))))
        print(f"{k:28s} {n:6d} {ns:5d} {S2:8.0f} {mean:6.2f} | {100 * d_:+7.1f} % +- {100 * se:4.1f} | {nl} | {100 * ex:5.1f} %   gain if real {S2 * d_ * d_:6.0f} u")


if __name__ == "__main__":
    main()
