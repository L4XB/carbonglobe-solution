#!/usr/bin/env python
"""Row-level agb prediction from ORNL DAAC 2522 for all seed ages, v6 = v4 with the NODE-centred old-growth effect.

11-11 04:00-05:00 UTC (scripts/ornl/grid_edges.py, node_q.py, node_hole_test.py; notes): the 0.01-deg product is a bilinear interpolation between
0.5-deg lookup-table nodes at the centres of the competition's cells, so the cell median of v1-v5 mixes the node's own response (56 %) with its eight
neighbours'. At one and the same place all existing-forest pixels show the strict old-growth effect (pixels within 0.05 deg of the node, fire-possible
sites: corr 0.996, robust sd 0.009), i.e. the "proxy" error of v4 (rms 0.07-0.10) came from WHERE the pixels sit, not from what they are. v6 takes

    q_old = median log ratio of the pixels within +-0.05 deg of the node (own weight >= 0.81): strict old-growth pixels, else relaxed old-growth
            pixels, else all existing-forest pixels (>= 8 pixels each);
            else the centre value of the 9-node tent fit over node +- 0.5 deg (plain fit, averaged with the biomass-weighted fit where that one is
            well determined; hold-out test with a +-0.1 / 0.2 deg hole: rms 0.037-0.043 against 0.071-0.089 of the cell median) for the same three
            pixel groups, with n_eff >= 20 and se < 0.03;
            else the v4 value (cell median / proxies / regression on q_new).

The qsrc codes of v4 are kept (1 strict, 7 strict with a failed state check, 4-6 proxies; a node value where v4 had only the regression or nothing
gets 4), so the builder's part labels do not change; nsrc says where q_old comes from (11-13 central old / relaxed / all, 14-16 fits, 0 = v4 value)
and dq_node is the change of q_old against v4. q_new (age-1 / young rows) stays the cell median: against ED's age-1 effect at fire-never sites the
node value is no better (corr 0.81 against 0.86; the new-forest ratio carries regeneration noise that the neighbours average out).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "ornl")); sys.path.insert(0, str(ROOT / "scripts" / "ed"))
from cgb import io, shift  # noqa: E402
import ornl_rows as orr  # noqa: E402
import dec_eval as de  # noqa: E402

W = {1: 0.85, 10: 0.79, 20: 0.76, 30: 0.72, 50: 0.57, 70: 0.45, 100: 0.38, 150: 0.19, 200: 0.0, 250: 0.0, 300: 0.0, 350: 0.0, 400: 0.0, 450: 0.0, 500: 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gmax", type=float, default=0.12)
    ap.add_argument("--min-old", type=int, default=12)
    ap.add_argument("--min-new", type=int, default=12)
    ap.add_argument("--tag", default="v6")
    args = ap.parse_args()
    z = np.load(ROOT / "work/ornl2522/cells_full.npz")
    # v5: REGIONAL strict old-growth effect (scripts/ornl/regional_q.py: the same pixel definition in the 8 / the next 16 cells around the site; corr with the
    # own-cell strict effect 0.94 / 0.84 at fire-possible sites, residual rms 0.14 / 0.23) for burnable cells without old-growth pixels of their own. It takes
    # precedence over the own-cell proxies (relaxed / top quartile / all pixels: corr with the regional strict effect 0.82-0.87, rms difference 0.21-0.31,
    # board kappa of the proxy old rows 0.6). qsrc 8 = ring 1, 9 = ring 2, 10 = regional pixels whose REF 2021 biomass fails the state check.
    zn = np.load(ROOT / "work/ornl2522/q_node.npz"); zm = np.load(ROOT / "work/ornl2522/q_node_mat.npz"); npos = {int(s): i for i, s in enumerate(zn["sites"])}
    zj = np.load(ROOT / "work/ornl2522/q_node_joint.npz")
    ALLMAP = {"pos": (0.94, -0.032), "cluster": (0.88, -0.034), "b": (0.93, -0.024), "never": (0.84, 0.002)}
    cl_ = np.load(ROOT / "work/collapse_cluster_labels.npy") == 0; mx_s = np.load(ROOT / "work/fire/max_factor_rule833.npy")

    def key_of(i: int) -> str:
        s_ = int(zn["sites"][i])
        return "cluster" if cl_[s_] else ("pos" if mx_s[s_] >= 1e3 else ("b" if mx_s[s_] >= 30 else "never"))

    def node_value(i: int, j: int, src_v4: int) -> tuple[float, int]:
        def central(zz, grp):
            return float(zz[f"{grp}_central"][i, j]) if zz[f"{grp}_n_cen"][i, j] >= 8 and np.isfinite(zz[f"{grp}_central"][i, j]) else float("nan")

        def fit(zz, grp):
            pl = zz[f"{grp}_plain"][i, j]
            if not (np.isfinite(pl) and zz[f"{grp}_neff"][i, j] >= 20 and zz[f"{grp}_plain_se"][i, j] < 0.03):
                return float("nan")
            rt = zz[f"{grp}_rat"][i, j] if f"{grp}_rat" in zz.files else np.nan
            return float(0.5 * (pl + rt)) if np.isfinite(rt) and zz[f"{grp}_rat_se"][i, j] < 0.03 and abs(rt - pl) < 0.15 else float(pl)

        # The pixel POPULATION of v4 is kept and only moved to the node (judged by exact-input ED on old rows, scripts/diag/v6_vs_ed.py: where strict pixels
        # replace the strict cell median the change is real - slope of (ED - v4) on (v6 - v4) +0.67 +- 0.11 central, +0.74 +- 0.11 fitted, class b +0.87 +- 0.13 -
        # where relaxed pixels near the node replace a strict cell median it is not: +0.05 +- 0.10; state purity counts more than position):
        #   v4 strict (1, 7)        -> strict pixels at the node (11), else their tent fit (14), else the v4 value
        #   v4 relaxed proxy (4)    -> relaxed pixels at the node (12), else their tent fit (15), else the joint fit (17)
        #   v4 top quartile / all pixels (5, 6), regression or nothing -> joint fit (17: field from all pixels, state offset for the non-mature ones;
        #                              hold-out with the mature pixels removed within 0.1 / 0.25 deg of the node: rms 0.036 / 0.045), else all pixels (13, 16)
        if src_v4 in (1, 7):
            order = ((11, central, zn, "old"), (14, fit, zn, "old"))
        elif src_v4 == 4:
            order = ((12, central, zm, "mat"), (15, fit, zm, "mat"))
        else:
            order = ()
        for code, fn, zz, grp in order:
            v = fn(zz, grp)
            if np.isfinite(v):
                return v, code
        if src_v4 in (1, 7):
            return float("nan"), 0
        if np.isfinite(zj["lam"][i, j]) and zj["se"][i, j] < 0.03 and zj["neff"][i, j] >= 20:
            return float(zj["lam"][i, j]), 17
        if src_v4 == 4:
            return float("nan"), 0
        for code, fn in ((13, central), (16, fit)):
            v = fn(zn, "all")
            if np.isfinite(v):
                a_, b_ = ALLMAP[key_of(i)]
                return a_ * v + b_, code
        return float("nan"), 0

    pos = {int(s): i for i, s in enumerate(z["sites"])}
    L = {k: z[k] for k in ("r0", "r1", "s126", "s585", "nr", "n126", "n585")}
    f = lambda a: np.where(a == -32768, np.nan, a.astype(np.float32) * np.float32(0.01))
    profile, AGES = orr.profile, orr.AGES
    _, y1 = io.open_test("ssp126"); _, y5 = io.open_test("ssp585"); FL = shift.FLOORS; M = shift.M_HEIGHT
    Y = {"126": (y1[:, :, 1] - FL[1]) * M * orr.F_AGB, "585": (y5[:, :, 1] - FL[1]) * M * orr.F_AGB}
    keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy(); scen = keys.scenario.to_numpy()
    cl = np.load(ROOT / "work/collapse_cluster_labels.npy") == 0; mx = np.load(ROOT / "work/fire/max_factor_rule833.npy"); strat = de.strata_of_sites()
    n = len(keys)
    pred = np.full(n, np.nan); kind = np.zeros(n, dtype=np.int8); qrow = np.full(n, np.nan); qmatch = np.full(n, np.nan)
    npx = np.zeros(n, dtype=np.int32); qsrc = np.zeros(n, dtype=np.int8); rat = np.full(n, np.nan)
    nsrc = np.zeros(n, dtype=np.int8); dqn = np.full(n, np.nan); qold_row = np.full(n, np.nan)
    order = np.argsort(site, kind="stable"); bounds = np.r_[0, np.flatnonzero(np.diff(site[order])) + 1, n]
    cnt = dict(old=0, old_reg=0, new_only=0, rej=0, none=0)
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        rows = order[b0:b1]; s = int(site[rows[0]])
        if s not in pos:
            continue
        i0 = pos[s]
        r0 = f(L["r0"][i0]); r1 = f(L["r1"][i0]); nr = f(L["nr"][i0])
        okx = np.isfinite(r0) & np.isfinite(r1) & (r0 > 0.3) & (r1 > 0)
        g = np.where(okx, np.log(np.maximum(r1, 1e-6) / np.maximum(r0, 1e-6)), np.nan)
        old = okx & (r0 > 1.0) & (g < args.gmax) & (g > -0.2)
        burnable = bool(cl[s]) or mx[s] >= 30
        for sc in ("126", "585"):
            y = Y[sc][s]; yold = float(y[7:].mean())
            s1 = f(L["s" + sc][i0]); n1 = f(L["n" + sc][i0])
            q_old = np.nan; q_new = np.nan; src = 0; ratio = np.nan
            m = old & np.isfinite(s1) & (s1 > 0)
            if m.sum() >= args.min_old and yold > 1.0:
                ratio = float(np.median(r0[m])) / yold
                q_old = float(np.median(np.log(s1[m] / r1[m])))
                if (not burnable) or (0.6 <= ratio <= 1.7):
                    src = 1
                else:
                    src = 7; cnt["rej"] += 1
            if src == 0 and yold > 1.0:          # proxies from the existing-forest pixels (linear maps of q_proxies.py, per fire class)
                mex = okx & np.isfinite(s1) & (s1 > 0)
                lq = np.log(np.maximum(s1, 1e-6) / np.maximum(r1, 1e-6))
                key = "cluster" if cl[s] else ("pos" if mx[s] >= 1e3 else ("b" if mx[s] >= 30 else "never"))
                mr = mex & (r0 > 0.5) & (g < 0.2) & (g > -0.3)
                if mr.sum() >= 6:
                    a_, b_ = {"pos": (0.99, -0.009), "cluster": (0.98, -0.005), "b": (1.01, -0.009), "never": (0.95, -0.001)}[key]
                    q_old = a_ * float(np.median(lq[mr])) + b_; src = 4; ratio = float(np.median(r0[mr])) / yold
                elif mex.sum() >= 24:
                    thr = np.quantile(r1[mex], 0.75); mt = mex & (r1 >= thr)
                    a_, b_ = {"pos": (0.98, -0.012), "cluster": (0.94, -0.021), "b": (0.97, -0.011), "never": (0.96, 0.0)}[key]
                    q_old = a_ * float(np.median(lq[mt])) + b_; src = 5; ratio = float(np.median(r0[mt])) / yold
                elif mex.sum() >= 12:
                    a_, b_ = {"pos": (0.94, -0.032), "cluster": (0.88, -0.034), "b": (0.93, -0.024), "never": (0.84, 0.002)}[key]
                    q_old = a_ * float(np.median(lq[mex])) + b_; src = 6; ratio = float(np.median(r0[mex])) / yold
            j_ = 0 if sc == "126" else 1; q_v4 = q_old; ns_ = 0
            if yold > 1.0 and s in npos:
                qn_, ns_ = node_value(npos[s], j_, src)
                if np.isfinite(qn_):
                    q_old = qn_
                    if src in (0, 2):
                        src = 4
                else:
                    ns_ = 0
            mn = np.isfinite(nr) & np.isfinite(n1) & (nr > 0.3) & (n1 > 0)
            if mn.sum() >= args.min_new:
                q_new = float(np.median(np.log(n1[mn] / nr[mn])))
            if not np.isfinite(q_old) and np.isfinite(q_new) and yold > 1.0:
                if cl[s]:
                    q_old = -0.056 + 0.716 * q_new; src = 2
                elif mx[s] >= 1e3:
                    q_old = -0.098 + 0.635 * q_new; src = 2
                elif mx[s] < 30 and strat[s] in ("warm", "moist"):
                    q_old = 0.51 * q_new; src = 2
            if src == 1: cnt["old"] += 1
            elif src in (4, 5, 6, 7): cnt["proxy"] = cnt.get("proxy", 0) + 1
            elif src in (8, 9, 10): cnt["regional"] = cnt.get("regional", 0) + 1
            elif src == 2: cnt["old_reg"] += 1
            elif np.isfinite(q_new): cnt["new_only"] += 1
            else: cnt["none"] += 1
            mx_ = okx & np.isfinite(s1) & (s1 > 0)
            gx = g[mx_]; qx = np.log(s1[mx_] / r1[mx_])
            for i in rows[scen[rows] == "ssp" + sc]:
                a = int(age[i]); w = W[a]
                if a == 1:
                    if not np.isfinite(q_new):
                        continue
                    q = q_old + w * (q_new - q_old) if np.isfinite(q_old) else q_new
                    rsrc = src if np.isfinite(q_old) else 3
                elif np.isfinite(q_old) and (a >= 200 or np.isfinite(q_new)):
                    q = q_old if a >= 200 else q_old + w * (q_new - q_old); rsrc = src
                elif np.isfinite(q_old) and a == 150:
                    q = q_old; rsrc = src
                else:
                    continue
                pred[i] = profile(y, a + 40) * np.exp(q); qrow[i] = q; qsrc[i] = rsrc; rat[i] = ratio
                nsrc[i] = ns_; qold_row[i] = q_old; dqn[i] = (q_old - q_v4) if np.isfinite(q_v4) else np.nan
                kind[i] = 2 if a == 1 else (1 if a >= 150 else 3)
                if a > 1 and gx.size:
                    gstar = np.log(profile(y, a + 40) / profile(y, a + 1))
                    sel = np.abs(gx - gstar) <= max(0.25 * abs(gstar), 0.04)
                    npx[i] = int(sel.sum())
                    if sel.sum() >= 8:
                        qmatch[i] = float(np.median(qx[sel]))
    np.savez(ROOT / f"work/ornl2522/rows_{args.tag}.npz", pred=pred, kind=kind, q=qrow, qmatch=qmatch, npx=npx, qsrc=qsrc, ratio=rat, nsrc=nsrc, dq_node=dqn, q_old=qold_row)
    o = kind == 1
    print("old rows by node source: " + ", ".join(f"{k}: {int((o & (nsrc == k)).sum())}" for k in (0, 11, 12, 14, 15, 17, 13, 16)) + f"; rms change of q_old against v4 where both exist: {np.sqrt(np.nanmean(dqn[o & (nsrc > 0)] ** 2)):.3f}")
    print(f"site x scenario: measured q_old {cnt['old']}, q_old by regression {cnt['old_reg']}, q_new only {cnt['new_only']}, rejected (state) {cnt['rej']}, nothing {cnt['none']}; "
          f"proxies / failed state check {cnt.get('proxy', 0)}; regional strict {cnt.get('regional', 0)}; rows by qsrc: {dict(zip(*np.unique(qsrc[qsrc > 0], return_counts=True)))}; rows: old {int((kind == 1).sum())}, age 1 {int((kind == 2).sum())}, intermediate {int((kind == 3).sum())} of {n}; from the regression {int((qsrc == 2).sum())}; matched-pixel estimates {int(np.isfinite(qmatch).sum())}")


if __name__ == "__main__":
    main()
