#!/usr/bin/env python
"""Candidate files after the 11-10 findings: the solution's agb keeps the y0 floor (0.401412 / 3.39567 = 0.1182 in solution
units; heights have no offset on vegetated rows), exact-input ED and ORNL DAAC 2522 at the exact cells agree row by row.

Components on top of a base file (default: the latest stack), each optional:
  low   agb cells of the base at or below `--low-thr` (default 0.02; bare and near-bare rows): base - 0.1182
        (board moments: the truth's mean over these cells is -0.114 [-0.168, -0.058])
  ed    rows with a well-seeded ED prediction at sites outside the collapse cluster: agb = a x hybrid - 0.1182 (not
        clipped), height = a x December height of year 40 (raw)
  ornl  rows with an ORNL prediction (work/ornl2522/rows_<tag>.npz: old and age-1 rows): agb = prediction - 0.1182;
        where ED exists too (non-cluster) the two are averaged, in the cluster ORNL alone counts
Writes results/<out>.csv (negative agb allowed) and work/dir_<out>.npy (file - C, for the screens).
    python scripts/build_floor_candidate.py --out cand_x --parts low,ed,ornl
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

FL_A = 0.401412 / 3.39567
C_FILE = "results/spatcorr_stack_20261106_s08.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/spatcorr_stack_20261110_s08.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="low,ed,ornl")
    ap.add_argument("--low-thr", type=float, default=0.02)
    ap.add_argument("--ed-roots", default="work/edv3/full_cx16+work/edv3/fire_cx17b")
    ap.add_argument("--gate", type=float, default=0.07)
    ap.add_argument("--ornl-tag", default="v2")
    ap.add_argument("--ornl-steps", default="", help="e.g. cl_old=0.8,pos_old=0.6,cold_old=0.5 (defaults as listed)")
    ap.add_argument("--ornl-match", type=int, default=1)
    ap.add_argument("--ed-max-factor", type=float, default=1e3)
    ap.add_argument("--midlow", type=float, default=0.06)
    ap.add_argument("--ed-decode", type=int, default=2, help="2 = chronosequence decode (dec2.py), 1 = the standard decode with --gate")
    ap.add_argument("--ed-weights", default="0.1,0.35,0.67", help="ED weight in the ED / ORNL average: age 1, ages 10-150, ages >= 200")
    ap.add_argument("--posed-tol", type=float, default=0.06)
    ap.add_argument("--hgt-steps", default="", help="e.g. boreal=0.7,cool=1.0")
    ap.add_argument("--bareh", type=float, default=0.900947 / 10.90645)
    ap.add_argument("--ornl-qmax", type=float, default=0.5, help="at sites that can burn (factor >= 30 or cluster) an ORNL effect above this (log, vs REF) means REF itself burned: row left at the base")
    ap.add_argument("--ornl-ratio", type=float, default=2.5, help="ORNL-only rows: the new value stays within [1 / r, r] x max(base, 0.15)")
    ap.add_argument("--ed-heights", type=int, default=1)
    ap.add_argument("--ed-sites", default="", help="optional file: restrict the ED component to these sites")
    ap.add_argument("--ed-exclude", default="", help="optional file: sites whose ED rows are not used")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    parts = set(args.parts.split(","))
    keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy()
    S = shift.read_predictions(ROOT / args.base).astype(np.float64); C = shift.read_predictions(ROOT / C_FILE).astype(np.float64)
    cl = (np.load(ROOT / "work/collapse_cluster_labels.npy") == 0)[site]
    P = S.copy(); src = np.zeros(S.shape, dtype=np.int8)        # 0 base, 1 low, 2 ED, 3 ORNL, 4 ED + ORNL
    if "low" in parts:
        m = S[:, 1] <= args.low_thr
        P[m, 1] = S[m, 1] - FL_A; src[m, 1] = 1
    ed_a = np.full(len(S), np.nan); ed_h = np.full(len(S), np.nan)
    if "ed" in parts:
        tabs = []; seen: set[int] = set()
        for r in args.ed_roots.split("+"):
            q = de.load_root(ROOT / r, "dec")
            if len(q) == 0:
                continue
            q = q[~q.site.isin(seen)]; seen |= set(q.site.unique()); tabs.append(q)
        t = pd.concat(tabs, ignore_index=True)
        if args.ed_sites:
            keep = {int(x) for x in open(ROOT / args.ed_sites).read().split()}; t = t[t.site.isin(keep)]
        if args.ed_exclude:
            drop = {int(x) for x in open(ROOT / args.ed_exclude).read().split()}; t = t[~t.site.isin(drop)]
        if args.ed_decode == 2:      # chronosequence decode (scripts/ed/dec2.py): level from the y0 profile at age + 40, ED supplies the scenario effect
            import dec2
            r2 = dec2.predict2(t, keys)
            ed_a = np.where(cl, np.nan, r2["a"]); ed_h = np.where(cl, np.nan, r2["h"])
        else:
            Pe, sim = de.predict(t, keys, C, "raw", args.gate, 0.0, 1.055)
            ok = sim & ~cl[:, None]
            ed_a[ok[:, 1]] = Pe[ok[:, 1], 1] - FL_A
            ed_h[ok[:, 0]] = Pe[ok[:, 0], 0]
        # ED stands alone only where its (weak) fire cannot matter: largest yearly dryness factor below --ed-max-factor
        # (fire-never and class-b sites; ED = ORNL there, corr 0.76-0.90). Elsewhere it only enters the average with ORNL.
        mxs = np.load(ROOT / "work/fire/max_factor_rule833.npy")[site]
        ed_alone = mxs < args.ed_max_factor
        m = np.isfinite(ed_a) & ed_alone; P[m, 1] = ed_a[m]; src[m, 1] = 2
        if args.ed_heights:
            m = np.isfinite(ed_h) & ed_alone; P[m, 0] = ed_h[m]; src[m, 0] = 2
    if "ornl" in parts:
        z = np.load(ROOT / f"work/ornl2522/rows_{args.ornl_tag}.npz"); kind = z["kind"]
        orn = z["pred"] * de.A / 1.055 / de.DIV[1] - FL_A
        if "qmatch" in z.files and args.ornl_match:       # cluster: average the age interpolation with the matched-pixel estimate
            qm, q = z["qmatch"], z["q"]
            adj = np.where(np.isfinite(qm) & cl & (kind == 3), np.exp(0.5 * (qm - q)), 1.0)
            orn = (orn + FL_A) * adj - FL_A
        strat = de.strata_of_sites()[site]; mxf = np.load(ROOT / "work/fire/max_factor_rule833.npy")[site]
        steps = dict(x.split("=") for x in args.ornl_steps.split(",")) if args.ornl_steps else {}
        st = lambda k, d: float(steps.get(k, d))
        # step toward ORNL where it stands alone (free screen 11-10: cluster old kappa 0.78, fire-possible old 0.32 +- 0.32, cool / boreal old ~0.1-0.3 +- 0.7, every other part >= 1)
        step = np.ones(len(S))
        step[cl & (kind == 1)] = st("cl_old", 0.8)
        step[~cl & (kind == 1) & (mxf >= 1e3)] = st("pos_old", 0.6)
        step[~cl & (kind == 1) & (mxf < 30) & np.isin(strat, ["cool", "boreal"])] = st("cold_old", 0.5)
        if "qsrc" in z.files:            # rows whose q_old comes from the regression on q_new (rows v3)
            qs = z["qsrc"]; reg = qs == 2
            step[reg & cl] = st("reg_cl", 1.0)
            step[reg & ~cl & (mxf < 30)] = st("reg_nev", 0.75)
            step[reg & ~cl & (mxf >= 1e3) & (z["q"] < -0.15)] = st("reg_pos_neg", 0.75)
            step[reg & ~cl & (mxf >= 1e3) & (z["q"] >= -0.15)] = st("reg_pos_pos", 0.0)     # free screen 11-10: rejected (chi2 +34)
        qv = z["q"]
        burnable = cl | (mxf >= 30)
        m = np.isfinite(orn) & ~(burnable & (qv > args.ornl_qmax))
        both = m & np.isfinite(ed_a)
        only = m & ~np.isfinite(ed_a)
        ref = np.maximum(S[:, 1], 0.15)
        tgt = np.clip(orn, ref / args.ornl_ratio - FL_A * (S[:, 1] < 0.15), ref * args.ornl_ratio)
        only &= step > 0
        P[only, 1] = S[only, 1] + step[only] * (tgt[only] - S[only, 1]); src[only, 1] = 3
        # weight of ED in the average with ORNL by age band (implied errors of the three-cornered hat, 11-10: age 1 ED 0.018 / ORNL 0.002,
        # young + mid 0.011 / 0.006, old 0.008 / 0.016); --ed-weights 0.5,0.5,0.5 restores the plain mean
        w1, w2, w3 = (float(x) for x in args.ed_weights.split(","))
        wed = np.where(age == 1, w1, np.where(age < 200, w2, w3))
        P[both, 1] = wed[both] * ed_a[both] + (1 - wed[both]) * orn[both]; src[both, 1] = 4
        if "posed" in parts:       # fire-possible sites whose (weak-fire) ED agrees with ORNL on the old rows of that scenario: no fire acted -> ED counts as at fire-never sites
            scen_ = keys.scenario.to_numpy(); posrow = ~cl & (mxf >= 1e3)
            chk = posrow & (age >= 200) & np.isfinite(ed_a) & np.isfinite(orn) & (z["qsrc"] == 1) & (S[:, 1] > 0.3)
            dl = np.full(len(S), np.nan); dl[chk] = np.log((ed_a[chk] + FL_A) / (orn[chk] + FL_A))
            import pandas as _pd
            g = _pd.DataFrame(dict(site=site[chk], scen=scen_[chk], d=dl[chk])).groupby(["site", "scen"]).d.agg(["median", "size"])
            good = {k for k, r in g.iterrows() if r["size"] >= 3 and abs(r["median"] + 0.02) <= args.posed_tol}
            okrow = np.array([(int(s_), sc_) in good for s_, sc_ in zip(site, scen_)]) & posrow
            m = okrow & np.isfinite(ed_a) & ~np.isfinite(orn); P[m, 1] = ed_a[m]; src[m, 1] = 2
            m = okrow & np.isfinite(ed_h); P[m, 0] = ed_h[m]; src[m, 0] = 2
            print(f"   fire-possible site x scenario with an ED / ORNL check: {len(g)}; agree within {args.posed_tol}: {len(good)}; ED heights on {int(m.sum())} cells")
        print(f"   ORNL rows dropped for q > {args.ornl_qmax} at burnable sites: {int((np.isfinite(orn) & burnable & (qv > args.ornl_qmax)).sum())}; clipped by the ratio bound: {int((only & (tgt != orn)).sum())}")
    if "midlow" in parts:      # stack agb between the low threshold and 0.4: the stack sits ~0.06 above floor-coded ED there (11-10)
        m = (src[:, 1] == 0) & (S[:, 1] > args.low_thr) & (S[:, 1] < 0.4)
        d = np.where(S[:, 1] <= 0.3, args.midlow, args.midlow * (0.4 - S[:, 1]) / 0.1)
        P[m, 1] = S[m, 1] - d[m]; src[m, 1] = 5
    if "hgt" in parts:         # heights without ED: the site's y0 height profile at age + 40 x exp(c + k q_ORNL), fitted on ED heights (scripts/diag/height_fit.py)
        from cgb import io
        z = np.load(ROOT / f"work/ornl2522/rows_{args.ornl_tag}.npz"); qv = z["q"]
        strat = de.strata_of_sites()[site]; mxf = np.load(ROOT / "work/fire/max_factor_rule833.npy")[site]
        scen = keys.scenario.to_numpy(); AG = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500])
        CK = {1: (0.036, 0.22), 10: (-0.002, 0.209), 20: (-0.001, 0.217), 30: (0.001, 0.186), 50: (0.003, 0.179), 70: (0.002, 0.198), 100: (0.003, 0.207), 150: (0.001, 0.247),
              200: (0.01, 0.193), 250: (0.006, 0.231), 300: (0.009, 0.203), 350: (0.011, 0.187), 400: (0.005, 0.238), 450: (0.004, 0.268), 500: (0.005, 0.254)}
        CK_COOL = {"age1": (0.083, 0.43), "young": (0.067, 0.34), "mid": (0.023, 0.32), "old": (0.030, 0.34)}
        CK_BOR = {"age1": (0.05, 0.5), "young": (0.0, 0.5), "mid": (0.0, 0.5), "old": (0.023, 0.27)}
        hstep = dict(x.split("=") for x in args.hgt_steps.split(",")) if args.hgt_steps else {}
        m = (src[:, 0] == 0) & np.isfinite(qv) & ~cl & (mxf < 1e3) & (S[:, 0] > 0.02)
        Yh = {}
        for sc in ("ssp126", "ssp585"):
            y0 = np.asarray(io.open_test(sc)[1][:, :, 0], dtype=np.float64)
            Yh[sc] = np.maximum((y0 - shift.FLOORS[0]) * shift.M_HEIGHT, 0.3)
        la = np.log(AG)
        for i in np.where(m)[0]:
            a = int(age[i]); st_ = strat[i]
            hc = float(np.exp(np.interp(np.log(min(a + 40, 500)), la, np.log(Yh[scen[i]][site[i]]))))
            grp = "age1" if a == 1 else ("young" if a < 50 else ("mid" if a < 150 else "old"))
            c_, k_ = CK[a] if st_ in ("moist", "warm", "dry") else (CK_COOL[grp] if st_ == "cool" else CK_BOR[grp])
            tgt = hc * np.exp(c_ + k_ * float(np.clip(qv[i], -1.0, 1.0))) * de.A / de.DIV[0]
            stp = float(hstep.get("boreal", 0.7)) if st_ == "boreal" else float(hstep.get(st_, 1.0))
            P[i, 0] = S[i, 0] + stp * (float(np.clip(tgt, S[i, 0] / 1.6, S[i, 0] * 1.6)) - S[i, 0]); src[i, 0] = 6
    if "bareh" in parts:       # bare height cells at the y0 height floor (board moments: -0.081 [-0.21, +0.044]; a test)
        m = (S[:, 0] <= 1e-9) & (src[:, 0] == 0)
        P[m, 0] = -args.bareh; src[m, 0] = 7
    P[:, 0] = np.maximum(P[:, 0], -0.0827 if "bareh" in parts else 0.0)
    P[:, 1] = np.maximum(P[:, 1], -FL_A)
    n_all = S.size
    names = {1: "low cells", 2: "ED", 3: "ORNL", 4: "ED + ORNL", 5: "mid-low", 6: "chrono x q", 7: "bare floor"}
    print(f"{args.out}: base {Path(args.base).name}; parts {sorted(parts)}")
    for c, cn in ((0, "height"), (1, "agb")):
        for k, nm in names.items():
            m = src[:, c] == k
            if m.sum():
                print(f"   {cn:6s} {nm:10s}: {int(m.sum()):6d} cells; mean change {np.mean(P[m, c] - S[m, c]):+.4f}; energy vs base {np.sum((P[m, c] - S[m, c]) ** 2) / n_all:.5f}, vs C {np.sum((P[m, c] - C[m, c]) ** 2) / n_all:.5f}")
    print(f"   total energy vs base {np.sum((P - S) ** 2) / n_all:.5f}; vs C {np.sum((P - C) ** 2) / n_all:.5f}; agb min {P[:, 1].min():.4f}, cells below 0: {int((P[:, 1] < 0).sum())}; "
          f"max |change| {np.abs(P - S).max():.2f}")
    np.save(ROOT / f"work/dir_{args.out}.npy", (P - C).astype(np.float32))
    np.save(ROOT / f"work/src_{args.out}.npy", src)
    if not args.no_write:
        shift.write_predictions(P, ROOT / "results" / f"{args.out}.csv")
        print(f"   wrote results/{args.out}.csv")


if __name__ == "__main__":
    main()
