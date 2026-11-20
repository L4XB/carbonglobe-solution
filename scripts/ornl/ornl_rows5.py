#!/usr/bin/env python
"""Row-level agb prediction from ORNL DAAC 2522 for all seed ages (v5 = v4 + the REGIONAL strict effect before the proxies; v4 = v3 + proxies of q_old where no strict old-growth pixels exist,
scripts/ornl/q_proxies.py: relaxed old-growth pixels r 1.00 / rms 0.03, top quartile by biomass r 0.98 / rms 0.08, all existing pixels r 0.97 / rms 0.10 at
fire-possible sites, against r 0.80 / rms 0.25 of the regression on q_new). FULL-resolution pixels of the exact cell
(work/ornl2522/cells_full.npz, 2500 per cell) and fallbacks for cells without old-growth pixels.

    agb40(age) = lookup(age + 40) x exp(q(age)),   q(age) = q_old + w(age) (q_new - q_old)   (w fitted on ED, 0 from age 200)
  q_old: median log(B2060_ssp / B2060_REF) over old-growth pixels (REF growth < gmax, B2021 > 1); at burnable sites
         (collapse cluster or dryness factor >= 30) only if their 2021 biomass is within [0.6, 1.7] x the site's old y0 agb.
         Missing q_old with q_new present: regression on q_new (fitted where both exist, page 2):
         cluster q_old = -0.056 + 0.716 q_new (R2 0.97); fire-possible -0.098 + 0.635 q_new (R2 0.65);
         fire-never warm / moist 0.0 + 0.51 q_new (R2 0.4-0.6); nothing for class b, cool, boreal.  flag src = 2
  q_new: median log(N2060_ssp / N2060_REF) over new-forest pixels with REF > 0.3.
  qmatch: median q of the existing-forest pixels whose REF growth matches the row's lookup growth (>= 8 pixels).
Writes work/ornl2522/rows_<tag>.npz: pred, kind (1 old, 2 age 1, 3 intermediate), q, qmatch, npx, qsrc (1 measured q_old, 2 regression on q_new, 3 q_new only,
4 relaxed old-growth pixels, 5 top quartile, 6 all existing pixels, 7 strict old-growth pixels that fail the state check at a burnable site), ratio (old pixels' 2021 biomass / site's old y0 agb).
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
    ap.add_argument("--tag", default="v5")
    args = ap.parse_args()
    z = np.load(ROOT / "work/ornl2522/cells_full.npz")
    # v5: REGIONAL strict old-growth effect (scripts/ornl/regional_q.py: the same pixel definition in the 8 / the next 16 cells around the site; corr with the
    # own-cell strict effect 0.94 / 0.84 at fire-possible sites, residual rms 0.14 / 0.23) for burnable cells without old-growth pixels of their own. It takes
    # precedence over the own-cell proxies (relaxed / top quartile / all pixels: corr with the regional strict effect 0.82-0.87, rms difference 0.21-0.31,
    # board kappa of the proxy old rows 0.6). qsrc 8 = ring 1, 9 = ring 2, 10 = regional pixels whose REF 2021 biomass fails the state check.
    zr = np.load(ROOT / "work/ornl2522/q_regional.npz"); rpos = {int(s): i for i, s in enumerate(zr["sites"])}
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
            reg_fail = None
            if src == 0 and yold > 1.0 and s in rpos:
                j_ = 0 if sc == "126" else 1
                for ring_, code_ in ((1, 8), (2, 9)):
                    qr_ = float(zr["q"][rpos[s], j_, ring_])
                    if np.isfinite(qr_):
                        rt_ = float(zr["b0"][rpos[s], j_, ring_]) / yold
                        if 0.6 <= rt_ <= 1.7:        # regional old-growth forest of the site's own biomass level
                            q_old = qr_; ratio = rt_; src = code_
                            break
                        if reg_fail is None:
                            reg_fail = (qr_, rt_)     # kept as a last resort (after the own-cell proxies, before the regression on q_new)
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
            if src == 0 and yold > 1.0 and reg_fail is not None:
                q_old, ratio = reg_fail; src = 10
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
                kind[i] = 2 if a == 1 else (1 if a >= 150 else 3)
                if a > 1 and gx.size:
                    gstar = np.log(profile(y, a + 40) / profile(y, a + 1))
                    sel = np.abs(gx - gstar) <= max(0.25 * abs(gstar), 0.04)
                    npx[i] = int(sel.sum())
                    if sel.sum() >= 8:
                        qmatch[i] = float(np.median(qx[sel]))
    np.savez(ROOT / f"work/ornl2522/rows_{args.tag}.npz", pred=pred, kind=kind, q=qrow, qmatch=qmatch, npx=npx, qsrc=qsrc, ratio=rat)
    print(f"site x scenario: measured q_old {cnt['old']}, q_old by regression {cnt['old_reg']}, q_new only {cnt['new_only']}, rejected (state) {cnt['rej']}, nothing {cnt['none']}; "
          f"proxies / failed state check {cnt.get('proxy', 0)}; regional strict {cnt.get('regional', 0)}; rows by qsrc: {dict(zip(*np.unique(qsrc[qsrc > 0], return_counts=True)))}; rows: old {int((kind == 1).sum())}, age 1 {int((kind == 2).sum())}, intermediate {int((kind == 3).sum())} of {n}; from the regression {int((qsrc == 2).sum())}; matched-pixel estimates {int(np.isfinite(qmatch).sum())}")


if __name__ == "__main__":
    main()
