#!/usr/bin/env python
"""Composite candidate, version 2 (11-10 afternoon): every changed cell carries a PART label and each part has its own step.

    P = S + step[part] x (target - S)                                   S = the 11-10 stack

Targets (solution units; agb floor-coded: physical / 1.204 / 3.39567 - 0.1182):
  low        agb cells with S <= --low-thr: S - 0.1182
  ED         exact-input ED rows, chronosequence decode (scripts/ed/dec2.py); never used in the collapse cluster
  ORNL       rows of work/ornl2522/rows_<tag>.npz (lookup(age + 40) x exp(q)), clipped to [1 / r, r] x max(S, 0.15)
  ED + ORNL  weighted mean where both exist (ED weight by age band, --ed-weights; class b --ed-weight-b)
  hgt        heights without ED: y0 height profile at age + 40 x exp(c + k q_ORNL)
  midlow     agb cells with low-thr < S < 0.4 without another source: S - 0.06 (fading out at 0.4)
  bareh      bare height cells at the y0 height floor (-0.0826)
At sites that can burn (dryness factor >= 30, non-cluster) ED is checked against ORNL per site x scenario on the old rows
(median log ratio within --tol of -0.02): 'agree' -> ED counts as at fire-never sites; 'disagree' -> ED unused;
unchecked: class b uses ED (heights at the step of edh_b_unchk), fire-possible sites do not.

Part labels (agb): low, midlow, ed_nev, ed_b, ed_pos, both_nev, both_b, both_pos, and for ORNL alone cl_old, cl_mid, cl_age1,
reg_cl, pos_old, pos_mid, pos_age1, reg_pos_neg, reg_pos_pos, o_b, nev_cold_old, nev_cold_ym, nev_warm_old, nev_warm_ym,
reg_nev; the cluster / fire-possible / class-b parts with q < -0.10 carry the suffix _m (cl_old_m, pos_old_m, o_b_m, ...).  (heights): edh_nev (non-boreal), edh_nev_bor, edh_age1, edh_b, edh_b_unchk, edh_pos, edh_pos_m, hgt_boreal, hgt_cool, hgt_warm, bareh.
    --steps part=value,...   overrides DEFAULT_STEPS;   --dump-parts <prefix>  writes work/dirx_<prefix>_<part>.npy
    (full-step increments target - S per part) and work/dirx_<prefix>__base.npy (S - C) for the joint ridge screen.
Writes results/<out>.csv, work/dir_<out>.npy (file - C), work/part_<out>.npy (labels).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "ed"))
from cgb import io, shift  # noqa: E402
import dec_eval as de  # noqa: E402
import dec2  # noqa: E402

FL_A = 0.401412 / 3.39567
C_FILE = "results/spatcorr_stack_20261106_s08.csv"
AGES = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500])
# posterior steps of the joint ridge screens of 11-10 14:30 - 15:10 UTC (scripts/fire/joint_ridge.py on the --dump-parts directions; 120 scored files;
# priors 1.0 +- 0.3 ED parts, 0.8 +- 0.5 measured ORNL parts, 1.0 +- 0.6 loss parts (_m), 0.5 +- 0.5 regression parts; low pinned at 1)
DEFAULT_STEPS = {
    "low": 1.0, "midlow": 0.6, "bareh": 1.0,
    "ed_nev": 1.0, "ed_b": 0.85, "ed_pos": 0.8, "both_nev": 0.95, "both_b": 1.0, "both_pos": 0.85,
    "cl_old_m": 0.65, "cl_old": 0.3, "cl_mid_m": 0.75, "cl_mid": 0.9, "cl_age1": 0.75, "px_cl": 0.0, "ps_cl": 0.4, "reg_cl": 0.2,
    # fire-possible sites: the steps below belong to the amplified targets (--amp pos=1.4:0.05); loss rows (q < -0.1) carry _m
    "pos_old_m": 1.3, "pos_old": 0.8, "pos_mid_m": 1.3, "pos_mid": 0.8, "pos_age1": 0.8,
    "px_pos_old": 0.5, "px_pos_old_m": -1.0, "px_pos_old_g": -1.0, "px_pos_ym": 0.2, "px_pos_ym_m": -1.0, "ps_pos_old": 0.3, "ps_pos_old_u": -1.0, "ps_pos_ym": 0.0, "ps_pos_ym_m": -1.0, "reg_pos_neg": 0.7, "reg_pos_pos": 0.0,
    "o_b_m": 1.3, "o_b": 0.7, "px_b": 0.0, "ps_b": 0.1,
    "nev_cold_old": 0.6, "nev_cold_ym": 1.1, "nev_warm_old": 0.9, "nev_warm_ym": 0.85, "px_nev_cold": 0.25, "px_nev_warm": 0.2, "reg_nev": 0.0,
    "edh_nev": 0.6, "edh_nev_bor": 0.6, "edh_b": 0.6, "edh_b_unchk": 0.4, "edh_pos": 0.0, "edh_pos_m": 0.0, "edh_pos_age1": 0.0, "edh_age1": 0.3, "edh_age1_bor": 0.3,
    "hgt_boreal": 0.6, "hgt_cool": 0.6, "hgt_warm": 0.6, "hcoup": 1.0,
    "edf": 1.0, "edf_m": 1.0, "edf_age1": 0.5, "edfh": 1.0, "edfh_m": 1.0, "edfh_age1": 0.5, "pe_pos_old": 0.8, "pe_pos_ym": 0.8, "hcol_cl": 1.0, "hcol_cl_tall": 1.0, "hcol_pos": 1.0, "hcol_pos_tall": 1.0,
    "chp_pos_old": 0.6, "chp_pos_ym": 0.6, "edh_fill": 0.8,
    "ffh_age1": 1.0, "ffh_ym": 1.0, "ffh_old": 1.0, "ffa_age1": 0.0, "ffa_ym": 1.0, "both_posq": 1.0,
    "pr_pos_old": 0.6, "pr_pos_ym": 0.2, "pr_b": 0.0, "pr_cl": 0.0,      # regional strict rows (rows v5): defaults = the proxy steps they replace
}
# height predictor refitted 11-10 14:25 UTC on 1033 ED sites (scripts/diag/height_fit.py; boreal sample of 157 sites)
CK = {1: (0.034, 0.33), 10: (-0.001, 0.303), 20: (-0.001, 0.283), 30: (0.0, 0.218), 50: (0.004, 0.202), 70: (0.003, 0.203), 100: (0.004, 0.218), 150: (0.003, 0.25),
      200: (0.011, 0.221), 250: (0.008, 0.255), 300: (0.009, 0.252), 350: (0.012, 0.223), 400: (0.007, 0.256), 450: (0.006, 0.276), 500: (0.008, 0.263)}
CK_COOL = {"age1": (0.067, 0.456), "young": (0.040, 0.381), "mid": (0.003, 0.359), "old": (0.012, 0.401)}
CK_BOR = {"age1": (0.162, 0.511), "young": (-0.028, 0.628), "mid": (-0.027, 0.511), "old": (0.015, 0.348)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/spatcorr_stack_20261110_s08.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="low,ed,ornl", help="components: low, ed, ornl, edh (ED heights), hgt, midlow, bareh, hcoup (heights follow an agb collapse)")
    ap.add_argument("--steps", default="", help="part=value,... overrides of DEFAULT_STEPS")
    ap.add_argument("--low-thr", type=float, default=0.02)
    ap.add_argument("--low-lookup", type=int, default=1, help="1: low cells whose lookup level at age + 40 is below 0.3 kgC get the floor-coded lookup value instead of the floor")
    ap.add_argument("--ed-roots", default="work/edv3/full_cx16+work/edv3/fire_cx17b+work/edv3/fire_cx17b_srv")      # _srv: the home-server run (Linux build; agrees with the Mac to 0.1-0.5 % median)
    ap.add_argument("--ornl-tag", default="v4")
    ap.add_argument("--pos-ed-quiet", type=float, default=0.0, help="ED weight in an ED / ORNL blend on fire-possible rows that are quiet in both (label both_posq); 0 = off")
    ap.add_argument("--ff-rows", default="", help="row predictions of the forced-fire ED runs (scripts/ed/ff_rows.py), components ffh (heights) / ffa (agb of ages < 150)")
    ap.add_argument("--chp-type-levels", action="store_true", help="part chp: block level of the stack per row type (burned state / forest / low) as well")
    ap.add_argument("--gate-h", type=float, default=0.25, help="ED heights: largest height seed mismatch (log) of the row")
    ap.add_argument("--gate-h2", type=float, default=0.0, help="ED heights: largest height seed mismatch of the two seed-age snapshots that bracket age + 40 (0 = no check)")
    ap.add_argument("--h-ages", default="1:500", help="component edh: seed-age range lo:hi of the rows that take ED heights")
    ap.add_argument("--block-levels", default="", help="log level shifts applied to the finished file per block, e.g. pos_h=-0.023,pos_moist_old_a=-0.024,b_a=-0.03,cl_old_a=-0.012 (blocks: pos_h, b_h, cl_h, pos_a, pos_moist_old_a, b_a, cl_old_a, nev_ym_a; from the level scan of the joint ridge)")
    ap.add_argument("--decode", default="chrono", choices=["chrono", "memory"], help="ED decode: chrono (y0 chronosequence at age + 40 x ED's effect on its own chronosequence) or memory (ED year 40 x (y0 / first record)^m with the training-world memory m; cool / warm / moist / dry strata only, boreal keeps chrono)")
    ap.add_argument("--h-site-median", type=int, default=0, help="ED heights: with at least this many usable mature rows (seed ages >= 150) every mature row of a site x scenario takes the site's median ED height response (0 = off)")
    ap.add_argument("--h-strata", default="", help="component edh: comma list of strata (boreal, cool, warm, moist, dry) that take ED heights; empty = all")
    ap.add_argument("--h-strata-pos", default="same", help="component edh: strata of the FIRE-POSSIBLE sites that take ED heights ('same' = --h-strata, 'all', 'none' or a comma list)")
    ap.add_argument("--amp-ed-qmin", type=float, default=-9.0, help="ED is averaged into q only where the ORNL effect is at least this (-0.1 = quiet rows)")
    ap.add_argument("--amp-ed-edmin", type=float, default=-9.0, help="... and only where ED's own effect is at least this (-0.15 = ED quiet)")
    ap.add_argument("--amp-ed-src", default="all", choices=("all", "nonstrict"), help="rows that take the ED averaging: all usable ones or only those without a strict ORNL value")
    ap.add_argument("--amp-ed-plain", action="store_true", help="apply the ED averaging also to fire-possible rows outside the amplified range (plain ORNL target)")
    ap.add_argument("--amp-ed-weight", type=float, default=0.0, help="weight of ED's own effect in q of the amplified model where the ED row is usable")
    ap.add_argument("--q-nb-src", default="all", choices=["all", "nonstrict", "strict", "proxy"], help="rows that take the neighbour mean with --q-nb-weight: all old rows, only proxy / failed-state rows (qsrc 4-7), only strict rows (qsrc 1) or only proxy rows (qsrc 4-6)")
    ap.add_argument("--q-nb-file", default="work/ornl2522/qnb300_row.npy", help="row file of the regional effect for --q-nb-weight: qnb300_row.npy = mean q_old of the test sites within 300 km; qreg_row.npy = STRICT old-growth pixels of the 8 (else 24) cells around the site (scripts/ornl/regional_q.py; corr 0.94 with the own-cell strict effect)")
    ap.add_argument("--q-nb-weight", type=float, default=0.0, help="amplified model at fire-possible sites, OLD rows: weight of the neighbours' mean q_old (all site x scenario within 300 km, work/ornl2522/qnb300_row.npy) in q; the ORNL effect is regionally coherent (corr of a site's q with its neighbours' mean 0.65-0.77)")
    ap.add_argument("--hcol-file", default="work/heights/hcond_A.npy", help="conditional heights (solution units) for the component hcol")
    ap.add_argument("--hcol-da", type=float, default=0.5, help="component hcol: rows whose agb falls below exp(-this) of the chronosequence")
    ap.add_argument("--pos-h-agb-gate", type=float, default=0.25, help="ED heights at fire-possible sites only on rows whose agb seed mismatch is below this")
    ap.add_argument("--amp-qmax", type=float, default=9.0, help="the amplified model applies only to rows with an ORNL effect below this (e.g. -0.10 = loss rows only; the others keep the plain ORNL target)")
    ap.add_argument("--edf-roots", default="", help="ED runs WITH window fire at fire-possible sites (components edf = agb, edfh = heights), e.g. work/edv3/fp_b+work/edv3/fp_b_srv")
    ap.add_argument("--amp", default="", help="amplified ORNL effect per class, e.g. pos=1.4:0.05,b=1.0:0.1 (target = lookup x exp(a (q - q0)); regression rows excluded); "
                    "optional third / fourth fields: the class's own qmax / qmin of the ORNL effect (pos=1.4:0.05:-0.10 = loss rows only, pos=1.4:0.05:9:-0.10 = quiet rows only)")
    ap.add_argument("--ornl-qmax", type=float, default=0.5)
    ap.add_argument("--ornl-ratio", type=float, default=2.5)
    ap.add_argument("--ornl-ratio-neg", type=float, default=4.0, help="lower clip 1 / r for rows with q < -0.3")
    ap.add_argument("--h-level", type=float, default=0.25, help="share of the block-level change (stratum x age band x scenario) kept in the ED / predictor heights; the pattern is kept in full (joint ridge 11-10: levels 0.0-0.3, patterns ~0.5-0.6 +- 0.55)")
    ap.add_argument("--ed-weights", default="0.1,0.45,0.7", help="ED weight in the ED / ORNL mean: age 1, ages 10-150, ages >= 200")
    ap.add_argument("--ed-weight-b", type=float, default=0.2, help="ED weight in the mean at class-b sites (ED error 0.027 vs ORNL 0.003 there, 11-10)")
    ap.add_argument("--tol", type=float, default=0.06)
    ap.add_argument("--hcoup", type=float, default=0.2, help="component hcoup: height log change per unit of agb log loss against the stack (training truth -0.195)")
    ap.add_argument("--robust", type=float, default=0.3, help="ED / ORNL rows that differ by more than this (solution units) take the median of ED, ORNL and the stack")
    ap.add_argument("--midlow", type=float, default=0.085)
    ap.add_argument("--dump-parts", default="")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    comp = set(args.parts.split(","))
    steps = dict(DEFAULT_STEPS)
    if args.steps:
        for x in args.steps.split(","):
            k, v = x.split("="); assert k in steps, f"unknown part {k}"; steps[k] = float(v)
    if steps["px_pos_old_m"] < 0:          # proxy loss rows follow the proxy step unless set (keeps every older recipe unchanged)
        steps["px_pos_old_m"] = steps["px_pos_old"]
    if steps["ps_pos_old_u"] < 0:
        steps["ps_pos_old_u"] = steps["ps_pos_old"]
    if steps["px_pos_old_g"] < 0:
        steps["px_pos_old_g"] = steps["px_pos_old"]
    for _lab, _par in (("px_pos_ym_m", "px_pos_ym"), ("ps_pos_ym_m", "ps_pos_ym")):      # young / mid LOSS rows (q < -0.1) of proxy / failed-state sites: follow the parent unless given
        if steps.get(_lab, -1.0) < 0:
            steps[_lab] = steps[_par]
    keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy(); scen = keys.scenario.to_numpy()
    S = shift.read_predictions(ROOT / args.base).astype(np.float64); C = shift.read_predictions(ROOT / C_FILE).astype(np.float64)
    n = len(S)
    cl = (np.load(ROOT / "work/collapse_cluster_labels.npy") == 0)[site]; mxf = np.load(ROOT / "work/fire/max_factor_rule833.npy")[site]
    strat = de.strata_of_sites()[site]; cold = np.isin(strat, ["cool", "boreal"])
    nev = ~cl & (mxf < 30); cb = ~cl & (mxf >= 30) & (mxf < 1e3); pos = ~cl & (mxf >= 1e3)
    T = S.copy(); part = np.full(S.shape, "", dtype=object)

    def put(mask: np.ndarray, col: int, target: np.ndarray, label: str) -> None:
        mask = mask & (part[:, col] == "")
        T[mask, col] = target[mask]; part[mask, col] = label

    lowm = S[:, 1] <= args.low_thr
    if "low" in comp:
        tl = S[:, 1] - FL_A
        if args.low_lookup:      # the lookup's own level at age + 40 (kgC; scripts/diag/low_cells_lookup.py): sparse grass / shrub rows (0.01-0.3 kgC) sit
            # between the floor and 0, the stack carries them as ~0 (it runs ~0.085 above the floor-coded lookup for every level below 2 kgC)
            Lk = np.load(ROOT / "work/lookup_agb40_kgC.npy").astype(np.float64)
            coded = Lk / 1.204 / 3.39567 - FL_A
            tl = np.where(Lk < 0.3, np.maximum(coded, -FL_A), tl)
        put(lowm, 1, tl, "low")
    # --- ED
    ed_a = np.full(n, np.nan); ed_h = np.full(n, np.nan); ed_q = np.full(n, np.nan); ed_ma = np.full(n, np.nan)
    if comp & {"ed", "edh", "edq"}:
        t = dec2.load_roots([ROOT / r for r in args.ed_roots.split("+")])
        r2 = dec2.predict2(t, keys, gate_h=args.gate_h, gate_h2=(args.gate_h2 if args.gate_h2 > 0 else None), h_site_median=args.h_site_median,
                           memory_zone=(de.strata_of_sites() if args.decode == "memory" else None))
        ed_a = np.where(cl | lowm, np.nan, r2["a"]); ed_h = np.where(cl, np.nan, r2["h"]); ed_q = r2["q_a"]; ed_ma = r2["mis_a"]
        print(f"   ED: {t.site.nunique()} sites; usable agb rows {int(np.isfinite(ed_a).sum())}, height rows {int(np.isfinite(ed_h).sum())}")
    # --- ED WITH FIRE at fire-possible sites (11-10 19:30 UTC: with the window fire at fp1 1e-7 the old-row effect of exact-input ED follows ORNL's with
    #     corr 0.83, slope 0.74; heights rise where the agb collapses): roots given by --edf-roots, chronosequence decode with wide effect bounds
    edf_a = np.full(n, np.nan); edf_h = np.full(n, np.nan)
    if comp & {"edf", "edfh"} and args.edf_roots:
        tf = dec2.load_roots([ROOT / r for r in args.edf_roots.split("+")])
        rf = dec2.predict2(tf, keys, qmax_a=3.0, qmax_h=0.9)
        edf_a = np.where(pos & ~lowm, rf["a"], np.nan); edf_h = np.where(pos, rf["h"], np.nan)
        print(f"   ED with fire: {tf.site.nunique()} sites; usable fire-possible agb rows {int(np.isfinite(edf_a).sum())}, height rows {int(np.isfinite(edf_h).sum())}")
    # --- ORNL
    orn = np.full(n, np.nan); kind = np.zeros(n, dtype=np.int8); qs = np.zeros(n, dtype=np.int8); qv = np.full(n, np.nan)
    if comp & {"ornl", "hgt", "ed", "edh"}:
        z = np.load(ROOT / f"work/ornl2522/rows_{args.ornl_tag}.npz"); kind = z["kind"]; qs = z["qsrc"]; qv = z["q"]
        orn = z["pred"] * de.A / 1.055 / de.DIV[1] - FL_A
        qm = z["qmatch"]; adj = np.where(np.isfinite(qm) & cl & (kind == 3), np.exp(0.5 * (qm - qv)), 1.0)     # cluster: mean with the matched-pixel estimate
        orn = (orn + FL_A) * adj - FL_A
    burnable = cl | (mxf >= 30)
    m0 = np.isfinite(orn) & ~(burnable & (qv > args.ornl_qmax)) & ~lowm
    ref = np.maximum(S[:, 1], 0.15)
    rlow = np.where(np.isfinite(qv) & (qv < -0.3), args.ornl_ratio_neg, args.ornl_ratio)      # a clear loss in ORNL: the target may fall further below the stack
    orn_t = orn.copy()
    amp_rows = np.zeros(n, bool)

    def _use_ed(mk: np.ndarray) -> np.ndarray:
        """Rows whose ORNL effect takes exact-input ED's own effect at weight --amp-ed-weight (optional filters: ORNL effect >= --amp-ed-qmin, ED effect
        >= --amp-ed-edmin = both quiet; --amp-ed-src nonstrict = only rows without a strict ORNL value)."""
        u = mk & np.isfinite(ed_q) & np.isfinite(ed_a) & (age > 1) & (args.amp_ed_weight > 0)
        u &= (np.nan_to_num(qv, nan=-9.0) >= args.amp_ed_qmin) & (np.nan_to_num(ed_q, nan=-9.0) >= args.amp_ed_edmin)
        if args.amp_ed_src == "nonstrict":
            u &= ~np.isin(qs, (1, 3))
        return u
    if args.amp:     # amplified effect per fire class: target = lookup x exp(a (q - q0))  (joint ridge 11-10: at fire-possible sites the truth's site effect is ~2x ORNL's around q0 = 0.05)
        for item in args.amp.split(","):
            k, v = item.split("="); vv = [float(x) for x in v.split(":")]; a_, q0_ = vv[0], vv[1]
            qmax_ = vv[2] if len(vv) > 2 else args.amp_qmax      # optional third field = the class's own qmax (pos=1.4:0.05:-0.10: loss rows only, the quiet rows keep plain ORNL)
            qmin_ = vv[3] if len(vv) > 3 else -9.0               # optional fourth field = qmin (pos=1.4:0.05:9:-0.10: quiet rows only, the loss rows keep plain ORNL)
            mk = {"pos": pos, "b": cb, "cl": cl, "nev": nev}[k] & np.isfinite(orn) & (qs != 2) & (np.nan_to_num(qv, nan=0.0) < qmax_) & (np.nan_to_num(qv, nan=0.0) >= qmin_)
            # --amp-ed-weight w: where the weak-fire ED row is usable (its lookup matches y0) its own effect is averaged into q before the amplification
            # (11-10: on strict rows ED's effect is ORNL's with slope 1.0-1.1, corr 0.75, rms difference 0.07 = two reconstructions of the same response)
            use_ed = _use_ed(mk); amp_rows |= mk
            qmix = np.where(use_ed, (1.0 - args.amp_ed_weight) * qv + args.amp_ed_weight * np.nan_to_num(ed_q), qv)
            if k == "pos" and args.q_nb_weight > 0:
                qnb = np.load(ROOT / args.q_nb_file)
                use_nb = mk & np.isfinite(qnb) & (kind == 1)
                if args.q_nb_src != "all":
                    use_nb &= np.isin(qs, (4, 5, 6, 7)) if args.q_nb_src == "nonstrict" else np.isin(qs, (4, 5, 6)) if args.q_nb_src == "proxy" else (qs == 1)
                qmix = np.where(use_nb, (1.0 - args.q_nb_weight) * qmix + args.q_nb_weight * np.nan_to_num(qnb), qmix)
                print(f"   amplified model, {k}: regional q ({Path(args.q_nb_file).stem}, rows: {args.q_nb_src}) mixed in with weight {args.q_nb_weight} on {int(use_nb.sum())} old rows; rms (q_nb - q) {np.sqrt(np.mean((qnb - qv)[use_nb] ** 2)):.3f}")
            qc = np.clip(qmix, -1.5, 0.6)
            orn_t = np.where(mk, (orn + FL_A) * np.exp(-qv) * np.exp(a_ * (qc - q0_)) - FL_A, orn_t)
            if use_ed.any(): print(f"   amplified model, {k}: ED effect averaged in (weight {args.amp_ed_weight}) on {int(use_ed.sum())} rows; rms (ED q - ORNL q) {np.sqrt(np.mean((ed_q - qv)[use_ed] ** 2)):.3f}")
            rlow = np.where(mk, 6.0, rlow)
    if args.amp_ed_weight > 0 and args.amp_ed_plain:      # the same averaging on fire-possible rows that keep the PLAIN ORNL target (outside the amplified range)
        up = _use_ed(pos & np.isfinite(orn) & (qs != 2) & ~amp_rows)
        qm_ = (1.0 - args.amp_ed_weight) * qv + args.amp_ed_weight * np.nan_to_num(ed_q)
        orn_t = np.where(up, (orn + FL_A) * np.exp(-qv) * np.exp(np.clip(qm_, -1.5, 0.6)) - FL_A, orn_t)
        print(f"   plain ORNL rows at fire-possible sites: ED effect averaged in (weight {args.amp_ed_weight}) on {int(up.sum())} rows")
    tgt_o = np.clip(orn_t, ref / rlow - FL_A * (S[:, 1] < 0.15), ref * args.ornl_ratio)
    # --- ED / ORNL agreement at burnable non-cluster sites, per site x scenario (old rows, measured q_old)
    state = np.full(n, "unchk", dtype=object)
    chk = (cb | pos) & (age >= 200) & np.isfinite(ed_a) & np.isfinite(orn) & (qs == 1) & (S[:, 1] > 0.3)
    if chk.any():
        g = pd.DataFrame(dict(site=site[chk], scen=scen[chk], d=np.log((ed_a[chk] + FL_A) / (orn[chk] + FL_A)))).groupby(["site", "scen"]).d.agg(["median", "size"])
        g = g[g["size"] >= 3]
        verdict = {k: ("agree" if abs(r["median"] + 0.02) <= args.tol else "dis") for k, r in g.iterrows()}
        state = np.array([verdict.get((int(s_), sc_), "unchk") for s_, sc_ in zip(site, scen)], dtype=object)
        for nm, mk in (("class b", cb), ("fire-possible", pos)):
            ks = {(int(s_), sc_) for s_, sc_ in zip(site[mk & chk], scen[mk & chk])} & set(verdict)
            print(f"   ED / ORNL check, {nm}: {len(ks)} site x scenario, agree {sum(1 for k in ks if verdict[k] == 'agree')}")
    # fire-possible sites: exact-input ED without the truth's fire does not follow the truth there (slope of its site effect on ORNL's 0.16, corr 0.3;
    # the stack 0.42, the board ~2), so with --amp it is not used for agb at all; without --amp it counts where it agrees with ORNL (the 11-10 morning rule)
    ed_ok_a = np.isfinite(ed_a) & (nev | (cb & (state != "dis")) | (pos & (state == "agree") & ("pos=" not in args.amp)))
    w1, w2, w3 = (float(x) for x in args.ed_weights.split(","))
    wed = np.where(age == 1, w1, np.where(age < 200, w2, w3)); wed = np.where(cb, args.ed_weight_b, wed)
    # --- forced-fire ED (scripts/ed/forced_fire.py -> scripts/ed/ff_rows.py): per site the lookup fire is tuned on the y0 profile and the window fire on
    #     file K's old-row agb; the RAW year-40 state (memory-corrected) then gives the heights of every seed age (ffh) and the agb of the younger rows
    #     (ffa; the old rows are the tuning target and keep their values). Training world: heights rms ~0.10 against 0.37 for the chronosequence.
    ff_h = np.full(n, np.nan); ff_a = np.full(n, np.nan)
    if comp & {"ffh", "ffa"} and args.ff_rows:
        zf = np.load(ROOT / args.ff_rows); ff_h = np.where(pos, zf["h"], np.nan); ff_a = np.where(pos & ~lowm, zf["a"], np.nan)
        print(f"   forced-fire ED: {int(np.isfinite(ff_h).sum())} height rows, {int(np.isfinite(ff_a).sum())} agb rows at {len(np.unique(site[np.isfinite(ff_h)]))} fire-possible sites")
    if "ffa" in comp:
        oka = np.isfinite(ff_a) & (age < 150)
        # age-1 rows only on request (step ffa_age1 > 0): ED's age-1 agb is its noisiest band (training validation 0.22 rms; at fire-never sites the ORNL
        # new-forest effect beat it), and a label with step 0 would close the cell to the ORNL parts
        if steps.get("ffa_age1", 0.0) > 0:
            put(oka & (age == 1), 1, ff_a, "ffa_age1")
        put(oka & (age > 1), 1, ff_a, "ffa_ym")
    if "ed" in comp and args.pos_ed_quiet > 0:
        # fire-possible rows that are QUIET in both sources (ORNL q >= -0.1, ED's own effect >= -0.15; seed gates of the decode): there weak-fire ED is
        # as valid as at fire-never sites (11-10: where its lookup matches y0 its window effect is ORNL's, corr 0.87, rms 0.07), so the two are blended
        # like the fire-never rows instead of ORNL alone. The loss rows stay with ORNL (weak-fire ED shows only a fraction of the losses).
        posq = pos & np.isfinite(ed_a) & np.isfinite(ed_q) & (ed_q >= -0.15) & np.isfinite(qv) & (qv >= -0.1) & (age > 1) & m0
        put(posq, 1, args.pos_ed_quiet * ed_a + (1 - args.pos_ed_quiet) * tgt_o, "both_posq")
    if "ed" in comp:
        both = ed_ok_a & m0 if "ornl" in comp else np.zeros(n, bool)
        mix = wed * ed_a + (1 - wed) * tgt_o
        # robust combination: where ED and ORNL disagree by more than --robust (2.9 % of the rows carry 36 % of their squared difference; the outlier is
        # ORNL's in two cases of three) the middle value of ED, ORNL and the stack counts
        far = np.abs(ed_a - tgt_o) > args.robust
        mix = np.where(far, np.median(np.stack([ed_a, tgt_o, S[:, 1]]), axis=0), mix)
        for nm, mk in (("nev", nev), ("b", cb), ("pos", pos)):
            put(both & mk, 1, mix, f"both_{nm}")
            put(ed_ok_a & ~both & mk, 1, ed_a, f"ed_{nm}")
    if "edf" in comp:        # ED with fire replaces the ORNL parts at fire-possible sites wherever its row is usable; loss rows (own effect below -0.15) labelled _m
        fq = np.log(np.maximum(edf_a + FL_A, 1e-3) / np.maximum(S[:, 1] + FL_A, 1e-3))
        put(np.isfinite(edf_a) & (age == 1), 1, edf_a, "edf_age1"); put(np.isfinite(edf_a) & (fq < -0.15), 1, edf_a, "edf_m"); put(np.isfinite(edf_a), 1, edf_a, "edf")
    if "edq" in comp:
        # fire-possible rows WITHOUT a strict ORNL effect (proxy q from relaxed / top-quartile / all pixels, failed state check, regression, or no ORNL row):
        # where the weak-fire ED lookup matches y0 (seed gate), ED's own effect stands in for q in the amplified model. 11-10 21:40 UTC: on strict rows
        # ED's effect IS ORNL's (corr 0.74-0.79, slope 1.0-1.1) while the proxies sit 0.10 lower and follow ED with slope 0.3-0.45.
        a_, q0_ = 1.0, 0.0
        for item in (args.amp.split(",") if args.amp else []):
            if item.startswith("pos="): a_, q0_ = [float(x) for x in item.split("=")[1].split(":")][:2]
        nostrict = ~np.isin(qs, (1, 3)) | ~np.isfinite(orn)
        okq = pos & nostrict & np.isfinite(ed_a) & np.isfinite(ed_q) & (age > 1) & ~lowm
        tgt_e = np.maximum((ed_a + FL_A) * np.exp(-ed_q) * np.exp(a_ * (np.clip(ed_q, -1.5, 0.6) - q0_)), 0.0) - FL_A
        put(okq & (age >= 200), 1, tgt_e, "pe_pos_old"); put(okq, 1, tgt_e, "pe_pos_ym")
    if "ornl" in comp:
        meas = np.isin(qs, (1, 3)); prox = np.isin(qs, (4, 5, 6)); fst = np.isin(qs, (7, 10)); regq = qs == 2
        _pl = ROOT / "work/ff/prior_lookup_ratio.npy"
        lookup_ok = (np.abs(np.nan_to_num(np.load(_pl)[:, 0], nan=9.0)) <= 0.12)[site] if _pl.exists() else np.zeros(n, bool)
        regs = np.isin(qs, (8, 9))        # rows v5: REGIONAL strict old-growth effect (8 / 24 cells around the site, state-checked) - labels pr_*
        qn = qv < -0.10           # rows where ORNL shows a loss against REF: the board wants at least the full step there (joint ridge 11-10)
        labs = [("cl_old_m", cl & (kind == 1) & meas & qn), ("cl_old", cl & (kind == 1) & meas), ("cl_mid_m", cl & (kind == 3) & meas & qn), ("cl_mid", cl & (kind == 3) & meas),
                ("cl_age1", cl & (kind == 2) & meas), ("px_cl", cl & prox), ("ps_cl", cl & fst), ("reg_cl", cl & regq),
                ("pos_old_m", pos & (kind == 1) & meas & qn), ("pos_old", pos & (kind == 1) & meas), ("pos_mid_m", pos & (kind == 3) & meas & qn), ("pos_mid", pos & (kind == 3) & meas),
                ("pos_age1", pos & (kind == 2) & meas),
                ("pr_pos_old", pos & regs & (kind == 1)), ("pr_pos_ym", pos & regs & (kind != 1)), ("pr_b", cb & regs), ("pr_cl", cl & regs),
                ("px_pos_old_m", pos & prox & (kind == 1) & qn),      # proxy LOSS rows (q < -0.1): own label since 11-11; its step follows px_pos_old unless given
                # failed-state old rows at sites whose LOOKUP is unburned (main-pass lookup within 12 % of y0): there the failed biomass check is mostly
                # canopy cover (the product's pixel value = cover x table value), not a wrong state; step follows ps_pos_old unless given
                ("ps_pos_old_u", pos & fst & (kind == 1) & lookup_ok),
                # proxy GAIN rows (q > +0.10): exact-input ED confirms the gain at matching-lookup sites (11-11 10:10 UTC: ORNL +0.165 / ED +0.121 at
                # q 0.1-0.3, +0.44 / +0.22 above 0.3, the stack -0.02 / 0.00) - unlike the proxy rows around q = 0, whose pivoted target is in doubt
                ("px_pos_old_g", pos & prox & (kind == 1) & (np.nan_to_num(qv, nan=0.0) > 0.10)),
                # young / mid LOSS rows of proxy / failed-state sites (11-11: a loss shows in every pixel state of a node, a CO2 gain of young pixels does not
                # carry to the row it stands in for) - own labels, steps follow px_pos_ym / ps_pos_ym unless given
                ("px_pos_ym_m", pos & prox & (kind != 1) & qn), ("ps_pos_ym_m", pos & fst & (kind != 1) & qn),
                ("px_pos_old", pos & prox & (kind == 1)), ("px_pos_ym", pos & prox & (kind != 1)), ("ps_pos_old", pos & fst & (kind == 1)), ("ps_pos_ym", pos & fst & (kind != 1)),
                ("reg_pos_neg", pos & regq & (qv < -0.15)), ("reg_pos_pos", pos & regq & (qv >= -0.15)),
                ("px_b", cb & prox), ("ps_b", cb & fst), ("o_b_m", cb & meas & qn), ("o_b", cb & meas),
                ("px_nev_cold", nev & cold & prox), ("px_nev_warm", nev & ~cold & prox),
                ("nev_cold_old", nev & cold & (kind == 1) & meas), ("nev_cold_ym", nev & cold & (kind != 1) & meas),
                ("nev_warm_old", nev & ~cold & (kind == 1) & meas), ("nev_warm_ym", nev & ~cold & (kind != 1) & meas), ("reg_nev", nev & regq)]
        for nm, mk in labs:
            put(m0 & mk, 1, tgt_o, nm)
    if "midlow" in comp:
        mm = (S[:, 1] > args.low_thr) & (S[:, 1] < 0.4)
        dlt = np.where(S[:, 1] <= 0.3, args.midlow, args.midlow * (0.4 - S[:, 1]) / 0.1)
        put(mm, 1, S[:, 1] - dlt, "midlow")
    # --- heights
    veg_h = S[:, 0] > 0.02
    if "ffh" in comp:
        okf_ = np.isfinite(ff_h) & veg_h
        put(okf_ & (age == 1), 0, ff_h, "ffh_age1"); put(okf_ & (age < 150), 0, ff_h, "ffh_ym"); put(okf_, 0, ff_h, "ffh_old")
    if "edh" in comp:
        hlo, hhi = (float(x) for x in args.h_ages.split(":"))
        okh0 = np.isfinite(ed_h) & veg_h & (age >= hlo) & (age <= hhi)         # --h-ages: ED heights only for these seed ages
        okh = okh0.copy()
        if args.h_strata:
            okh &= np.isin(strat, args.h_strata.split(","))                     # --h-strata: and only in these climate strata
        # --h-strata-pos: the fire-possible sites have their own strata list (they sit mostly in the dry / warm / moist zones)
        okh_p = okh if args.h_strata_pos == "same" else okh0 if args.h_strata_pos == "all" else np.zeros(n, bool) if args.h_strata_pos == "none" else okh0 & np.isin(strat, args.h_strata_pos.split(","))
        # age-1 rows: our seed is one growth year older than the organisers' and ED's age-1 agb is its noisiest band (implied error 0.018 vs 0.002 of ORNL) -> own label
        # fire-possible sites: only rows whose AGB seed matches too (11-10 21:30 UTC: 36 % of the usable height rows there come from lookups that are burned
        # in the truth - the height gate alone lets them pass - and they carry 2/3 of the energy with a weak-fire height rise that is not the truth's situation)
        okp = okh_p & pos & (np.nan_to_num(ed_ma, nan=9.0) < args.pos_h_agb_gate)
        put(okh & nev & (strat == "boreal") & (age == 1), 0, ed_h, "edh_age1_bor")
        put(okh & (nev | (cb & (state != "dis"))) & (age == 1), 0, ed_h, "edh_age1"); put(okp & (age == 1), 0, ed_h, "edh_pos_age1")
        put(okh & nev & (strat == "boreal"), 0, ed_h, "edh_nev_bor"); put(okh & nev, 0, ed_h, "edh_nev")
        put(okh & cb & (state == "agree"), 0, ed_h, "edh_b"); put(okh & cb & (state == "unchk"), 0, ed_h, "edh_b_unchk")
        # fire-possible sites: heights do not move with the fire loss (board 11-10) and the board's level of old heights there (-2.6 +- 2 % against the
        # stack) agrees with no-fire ED (-4.2 %): ED heights at every fire-possible site, loss site x scenario (ORNL q_old < -0.1) under their own label
        qold_site = np.load(ROOT / "work/ornl2522/qold_row_v4.npy") if (ROOT / "work/ornl2522/qold_row_v4.npy").exists() else np.full(n, np.nan)
        put(okp & np.isfinite(qold_site) & (qold_site < -0.10), 0, ed_h, "edh_pos_m"); put(okp, 0, ed_h, "edh_pos")
    if "edfh" in comp:
        okf = np.isfinite(edf_h) & veg_h
        fqa = np.log(np.maximum(edf_a + FL_A, 1e-3) / np.maximum(S[:, 1] + FL_A, 1e-3))
        put(okf & (age == 1), 0, edf_h, "edfh_age1"); put(okf & np.isfinite(fqa) & (fqa < -0.15), 0, edf_h, "edfh_m"); put(okf, 0, edf_h, "edfh")
    if "hgt" in comp:
        mh = veg_h & np.isfinite(qv) & ~cl & (mxf < 1e3) & (part[:, 0] == "")
        Yh = {sc: np.maximum((np.asarray(io.open_test(sc)[1][:, :, 0], dtype=np.float64) - shift.FLOORS[0]) * shift.M_HEIGHT, 0.3) for sc in ("ssp126", "ssp585")}
        la = np.log(AGES); tg = np.full(n, np.nan)
        for i in np.where(mh)[0]:
            a = int(age[i]); st_ = strat[i]
            hc = float(np.exp(np.interp(np.log(min(a + 40, 500)), la, np.log(Yh[scen[i]][site[i]]))))
            grp = "age1" if a == 1 else ("young" if a < 50 else ("mid" if a < 150 else "old"))
            c_, k_ = CK[a] if st_ in ("moist", "warm", "dry") else (CK_COOL[grp] if st_ == "cool" else CK_BOR[grp])
            tg[i] = float(np.clip(hc * np.exp(c_ + k_ * float(np.clip(qv[i], -1.0, 1.0))) * de.A / de.DIV[0], S[i, 0] / 1.6, S[i, 0] * 1.6))
        put(mh & (strat == "boreal"), 0, tg, "hgt_boreal"); put(mh & (strat == "cool"), 0, tg, "hgt_cool"); put(mh & ~cold, 0, tg, "hgt_warm")
    if "edhfill" in comp and "edh" in comp:
        # fire-never rows of the strata that take ED heights but have no usable ED row of their own (seed gates; mostly low boreal stands): the y0
        # chronosequence at age + 40 x the MEDIAN ED height response of the usable rows of the same stratum x age band x scenario. The level is what the
        # board confirmed (CY / CM, 11-11); the site pattern of these rows stays the chronosequence's.
        chh = np.load(ROOT / "work/lookup_h40_m.npy").astype(np.float64) / 1.14114 / 10.90645
        qh = r2["q_h"]; lab_h = part[:, 0].astype(str)
        src = np.isin(lab_h, ("edh_nev", "edh_nev_bor")) & np.isfinite(qh) & (age > 1)
        bandf = np.where(age < 50, 1, np.where(age < 150, 2, 3))
        key = pd.Series(strat.astype(str)) + "|" + pd.Series(bandf.astype(str)) + "|" + pd.Series(scen.astype(str))
        med = pd.Series(qh)[src].groupby(key[src]).median()
        lvl = key.map(med).to_numpy(dtype=float)
        tfill = chh * np.exp(np.nan_to_num(lvl))
        okf = nev & veg_h & (age > 1) & (chh > 0.1) & np.isfinite(lvl) & (S[:, 0] > 0.1)
        if args.h_strata:
            okf &= np.isin(strat, args.h_strata.split(","))
        put(okf, 0, tfill, "edh_fill")
    if "chp" in comp:
        # fire-possible sites (outside the cluster): the stack's heights deviate from the y0 chronosequence at age + 40 by 5-14 % rms even after the block
        # level is matched (798 squared units, 11-11 00:50 UTC; fire-never sites: the stack IS the chronosequence). Target = chronosequence x the stack's own
        # block level (median of log(stack / chronosequence) per stratum x age band x scenario), i.e. only the row pattern changes; the levels of the
        # stack (board-calibrated) stay. In the cluster the old probes reject this pattern (kappa -0.06 +- 0.16), at fire-possible sites they are blind.
        chh = np.load(ROOT / "work/lookup_h40_m.npy").astype(np.float64) / 1.14114 / 10.90645
        okc = pos & (S[:, 0] > 0.3) & (chh > 0.3) & (age > 1)
        bandc = np.where(age < 50, 1, np.where(age < 150, 2, 3))
        key = pd.Series(strat.astype(str)) + "|" + pd.Series(bandc.astype(str)) + "|" + pd.Series(scen.astype(str))
        if args.chp_type_levels:
            # block level per ROW TYPE too (lookup at age + 40: burned state = >= 30 m with <= 6 kgC, forest = >= 12 m, else low). 11-11 04:45 UTC: with one level
            # per stratum x band the burned-state rows are lifted by 8-10 % (the stack holds them at the chronosequence, forest rows 7 % above it; 203 of the
            # 831 full-step units), although the old probes put the truth's mean burned-state height at +1 +- 4 % against the stack. The part should change
            # the row pattern inside a type only.
            cha = np.load(ROOT / "work/lookup_agb40_kgC.npy").astype(np.float64); chm = chh * 1.14114 * 10.90645
            typ = np.where((chm >= 30.0) & (cha <= 6.0), "bs", np.where(chm >= 12.0, "fo", "lo"))
            key = key + "|" + pd.Series(typ)
        rl = pd.Series(np.log(np.where(okc, S[:, 0] / np.maximum(chh, 1e-9), 1.0)))
        lvl = rl[okc].groupby(key[okc]).transform("median").reindex(rl.index).to_numpy()
        tch = chh * np.exp(np.nan_to_num(lvl))
        put(okc & (age >= 150), 0, tch, "chp_pos_old"); put(okc, 0, tch, "chp_pos_ym")
    if "bareh" in comp:
        put(S[:, 0] <= 1e-9, 0, np.full(n, -0.900947 / 10.90645), "bareh")
    # --- heights: keep only a share of the block-level change of the ED / predictor parts (the stack's block levels are board-calibrated)
    if args.h_level != 1.0:
        band = np.where(age == 1, 0, np.where(age < 50, 1, np.where(age < 150, 2, 3)))
        blk = pd.factorize(pd.Series(strat.astype(str)) + "|" + pd.Series(band.astype(str)) + "|" + pd.Series(scen.astype(str)))[0]
        for lab in sorted(set(part[:, 0]) - {"", "bareh", "edfh", "edfh_m", "edfh_age1", "hcoup", "chp_pos_old", "chp_pos_ym", "edh_fill"}):
            m = part[:, 0] == lab
            dlt = pd.Series(T[m, 0] - S[m, 0]); mean = dlt.groupby(blk[m]).transform("mean").to_numpy()
            T[m, 0] = S[m, 0] + (dlt.to_numpy() - mean) + args.h_level * mean
    # --- apply the steps
    stp = np.zeros(S.shape)
    for lab in np.unique(part):
        if lab:
            stp[part == lab] = steps[lab]
    P = S + stp * (T - S)
    # steps above 1 on a loss must not run below a grass state (0.4 kgC, the lookup's burned level; floor-coded -0.02) unless the target itself is lower
    GRASS = 0.4 / 1.204 / 3.39567 - FL_A
    over = (stp[:, 1] > 1.0) & (T[:, 1] < S[:, 1])
    P[over, 1] = np.maximum(P[over, 1], np.minimum(T[over, 1], GRASS))
    if "hcoup" in comp:
        # heights follow an agb COLLAPSE (training truth, old rows, scripts/diag/train_height_coupling.py: agb log change -0.25..-0.5 -> height +12 %,
        # -0.5..-1 -> +29 %, below -1 -> +44 %; linear fit +0.05 - 0.195 x change; the calibrated cluster heights of the stack show the same -0.20).
        # Applied where this file puts the agb of a fire-possible / class-b row BELOW the stack: height x exp(-c x log(agb file / agb stack)).
        dq = np.log(np.maximum(P[:, 1] + FL_A, 0.02) / np.maximum(S[:, 1] + FL_A, 0.02))
        mh = (pos | cb) & (dq < -0.10) & (part[:, 0] == "") & (S[:, 0] > 0.05) & np.isin(part[:, 1], ("pos_old_m", "pos_mid_m", "px_pos_old", "px_pos_ym", "ps_pos_old", "o_b_m", "reg_pos_neg", "pos_old", "pos_mid"))
        th = S[:, 0] * np.exp(-args.hcoup * np.clip(dq, -1.5, 0.0))
        T[mh, 0] = th[mh]; part[mh, 0] = "hcoup"
        P[mh, 0] = S[mh, 0] + steps["hcoup"] * (T[mh, 0] - S[mh, 0])
    if "hcol" in comp:
        # heights of the rows this file takes down HARD (agb below exp(--hcol-da) of the y0 chronosequence at age + 40, chronosequence agb > 3 kgC): the
        # training-world conditional height given the year-40 agb (scripts/heights/cond_height.py; --hcol-file, solution units). Training truth: collapsing
        # forest rows end at 31-35 m (fire patches keep 30 % of every cohort and the channel is the tallest cohort per patch) while burned-state rows that
        # burn on lose their survivors; the model averages both. Labels by cluster / elsewhere and by the chronosequence height (below / above 27 m).
        Hc = np.load(ROOT / args.hcol_file).astype(np.float64); Lk2 = np.load(ROOT / "work/lookup_agb40_kgC.npy").astype(np.float64); hc2 = np.load(ROOT / "work/lookup_h40_m.npy").astype(np.float64)
        daf = np.log(((P[:, 1] + FL_A) * 3.39567 * 1.204 + 0.05) / (Lk2 + 0.05))
        mh = (daf < -args.hcol_da) & (Lk2 > 3.0) & (age > 1) & (S[:, 0] > 0.05) & (part[:, 0] == "") & np.isfinite(Hc)
        for lab, mk in (("hcol_cl", cl & (hc2 < 27.0)), ("hcol_cl_tall", cl & (hc2 >= 27.0)), ("hcol_pos", ~cl & (hc2 < 27.0)), ("hcol_pos_tall", ~cl & (hc2 >= 27.0))):
            mm = mh & mk
            T[mm, 0] = Hc[mm]; part[mm, 0] = lab; P[mm, 0] = S[mm, 0] + steps[lab] * (T[mm, 0] - S[mm, 0])
    if args.block_levels:
        blocks = {"pos_h": (pos & (S[:, 0] > 0.05), 0), "b_h": (cb & (S[:, 0] > 0.05), 0), "cl_h": (cl & (S[:, 0] > 0.05), 0),
                  "pos_a": (pos & (S[:, 1] > 0.05), 1), "pos_moist_old_a": (pos & (strat == "moist") & (age >= 150) & (S[:, 1] > 0.05), 1),
                  "b_a": (cb & (S[:, 1] > 0.05), 1), "cl_old_a": (cl & (age >= 150) & (S[:, 1] > 0.05), 1), "nev_ym_a": (nev & (age < 150) & (S[:, 1] > 0.05), 1)}
        for item in args.block_levels.split(","):
            k, v = item.split("="); mk, col = blocks[k]; v = float(v)
            off = FL_A if col == 1 else 0.0
            before = P[mk, col].copy(); P[mk, col] = (P[mk, col] + off) * np.exp(v) - off
            print(f"   block level {k}: {int(mk.sum())} cells x exp({v:+.3f}); energy of the shift {np.sum((P[mk, col] - before) ** 2):.1f}")
    P[:, 0] = np.maximum(P[:, 0], -0.0827 if "bareh" in comp else 0.0); P[:, 1] = np.maximum(P[:, 1], -FL_A)
    # the height channel is capped at 35.000 m in the organisers' model (training maximum): nothing this file changes may end above the cap
    HCAP = 35.0 / 1.14114 / 10.90645
    over_cap = (P[:, 0] > HCAP) & (P[:, 0] > S[:, 0])
    if over_cap.any():
        print(f"   height cap: {int(over_cap.sum())} changed cells above 35 m set to max(35 m, stack); energy removed {np.sum((P[over_cap, 0] - np.maximum(HCAP, S[over_cap, 0])) ** 2):.2f}")
        P[over_cap, 0] = np.maximum(HCAP, S[over_cap, 0])
    N = S.size
    print(f"{args.out}: components {sorted(comp)}")
    for c, cn in ((0, "height"), (1, "agb")):
        for lab in sorted(set(part[:, c]) - {""}):
            m = part[:, c] == lab
            print(f"   {cn:6s} {lab:13s} step {steps[lab]:.2f}: {int(m.sum()):6d} cells; mean full change {np.mean(T[m, c] - S[m, c]):+.4f}; energy at full step {np.sum((T[m, c] - S[m, c]) ** 2):7.1f}, applied {np.sum((P[m, c] - S[m, c]) ** 2):7.1f}")
        veg = S[:, c] > 0.02; un = veg & (part[:, c] == "")
        print(f"   {cn:6s} left at the stack: {int(un.sum())} vegetated cells, S^2 weight {np.sum(S[un, c] ** 2) / np.sum(S[veg, c] ** 2):.3f}")
    print(f"   total energy vs the stack {np.sum((P - S) ** 2):.0f} squared units ({np.sum((P - S) ** 2) / N:.5f}); vs C {np.sum((P - C) ** 2) / N:.5f}; column means {P[:, 0].mean():.4f} {P[:, 1].mean():.4f}")
    np.save(ROOT / f"work/dir_{args.out}.npy", (P - C).astype(np.float32)); np.save(ROOT / f"work/part_{args.out}.npy", part.astype("U13"))
    if args.dump_parts:
        np.save(ROOT / f"work/dirx_{args.dump_parts}__base.npy", (S - C).astype(np.float32))
        for c in (0, 1):
            for lab in sorted(set(part[:, c]) - {""}):
                d = np.zeros_like(S); m = part[:, c] == lab; d[m, c] = T[m, c] - S[m, c]
                np.save(ROOT / f"work/dirx_{args.dump_parts}_{lab}.npy", d.astype(np.float32))
    if not args.no_write:
        shift.write_predictions(P, ROOT / "results" / f"{args.out}.csv")
        print(f"   wrote results/{args.out}.csv")


if __name__ == "__main__":
    main()
