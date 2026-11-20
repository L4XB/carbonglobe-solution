#!/usr/bin/env python
"""Write the next day's design spec from the scores of the ladder of 11-12 (cand49_E / Y / LY / PLY / PLYH).

The ladder measures four deviations from file E, each from one pair of neighbouring files (rounded scores, +-0.5 quantum each):
    YM         = E -> Y        young / mid rows of the non-strict fire-possible sites at 0.75 towards their plain ORNL rows
    plain_loss = Y -> LY       loss rows of the fire-possible sites plain
    pivot_off  = LY -> PLY     quiet rows plain, no class-b pivot
    no_H       = PLY -> PLYH   no ED heights in the warm / moist / dry fire-never zones and at class b
With kappa = optimal step along a deviation (gain at the full step = (2 kappa - 1) x energy) the posterior of the four kappas is sampled (Gaussian
priors, exact rounding intervals) and the new BASE is the file at the posterior-mean steps:
    anchor recipe (plain if w_P + w_L >= 1, else E's amplified recipe) + mix entries for the four deviations (scripts/design_run.py, prediction space).
New factors are added as mix entries evaluated on the anchor (one pair of builds each) or as post steps. Layout "loo": the ALL-IN file (base + every
factor) is the first file and each of the others leaves one factor out - every file is a full candidate and the all-in composite is on the board;
layout "singles": base and base + one factor each.

    python scripts/make_spec_next.py --scores 25,24,24,22,23 --factors F,cl_full,orn_full,nev_orn [--day 50] [--out work/logs/day50_spec.json]
        [--weights P=1,L=1,Y=1,H=0] [--layout loo] [--prefix cand50] [--ff-rows work/ff/ff_rows_day50.npz]
Scores are in units of 0.001 in the order E, Y, LY, PLY, PLYH. Nothing is submitted here; `DRY=1 CHECK=1 bash scripts/submit_batch_day50.sh` tests the spec.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
Q = 370.26
HH = {"edh_nev": 1.0, "edh_nev_bor": 0.85, "edh_age1": 1.0, "edh_age1_bor": 1.0, "edh_b": 0.9, "edh_b_unchk": 0.9}
RA = {"ps_pos_old": 0.6, "ps_pos_old_u": 0.8, "px_pos_old_m": 0.85, "px_pos_old": 0.2, "px_pos_old_g": 0.6, "pos_old": 1.0, "pos_mid": 1.0}
L11 = {"pos_old_m": 1.1, "pos_mid_m": 1.1}
L10 = {"pos_old_m": 1.0, "pos_mid_m": 1.0}
YS = {k: 0.75 for k in ("ps_pos_ym", "ps_pos_ym_m", "px_pos_ym", "px_pos_ym_m")}
AMP_E = {"pos": {"a": 1.4, "q0": 0.05}, "b": {"a": 1.0, "q0": 0.025}}
PLAIN_ON = {"amp": {"pos": {"a": 1.0, "q0": 0.0, "qmax": 9, "qmin": -9}, "b": None}, "steps": L10}
NO_H = {"h_strata": "boreal,cool", "steps": {"edh_b": 0.0, "edh_b_unchk": 0.0}}
HS = "user@server:carbonglobe/work/edv3"
PYT = "python"
# priors of the four kappas (mean, sd): P = mixture of "plain" (1) and "pivot right" (~0, old probes -0.12 +- 0.46); L: plain 1 / uniform offset 0.76 /
# amplified 0.2-0.5; Y (on the 0.75-step deviation): plain 1.4, own-fuel 0.8-0.95, amplified 0.5-0.6; H: ED heights expected right (training 0.02-0.04)
PRIOR = {"P": (0.35, 0.50), "L": (0.60, 0.45), "Y": (0.90, 0.40), "H": (0.15, 0.30)}
# steps may run past E on the amplified side (kappa < 0 = the truth sits beyond E: stronger pivot / steeper loss slope), not past the plain target
CLIP = {"P": (-1.0, 1.0), "L": (-1.0, 1.0), "Y": (0.0, 1.33), "H": (0.0, 1.0)}

# factor library: name -> (kind, payload, text). kind "mix" = recipe delta evaluated on the anchor and added in prediction space; "post" = post step;
# "recipe" = plain recipe delta (changes every build of the file).
LIB = {
    "F": ("post", {"id": "ffpos", "tool": "apply_ff", "ff_rows": "@FF", "where": "pos", "h_step": 0.85, "h_old_max": -0.15, "h_forest_strata": "moist,warm"},
          "forced-fire ED heights (low / young seeds everywhere, burning forest seeds moist / warm, step 0.85)"),
    "Fa": ("post", {"id": "ffagb", "tool": "apply_ff", "ff_rows": "@FF", "where": "pos", "a_step": 0.85, "a_old_max": -0.3, "a_ages": "1:100", "a_nonstrict": True},
           "forced-fire ED agb of ages 1-100 at burning site x scenarios without a strict ORNL effect (step 0.85)"),
    "cl_full": ("mix", {"steps": {"cl_old": 0.8, "cl_old_m": 1.0, "cl_mid_m": 1.0, "cl_mid": 1.0, "cl_age1": 1.0, "px_cl": 0.6, "ps_cl": 0.8}},
                "cluster ORNL parts at (nearly) full steps"),
    "cl_meas": ("mix", {"steps": {"cl_old": 0.8, "cl_old_m": 1.0, "cl_mid_m": 1.0, "cl_mid": 1.0, "cl_age1": 1.0}},
                "measured cluster ORNL parts at (nearly) full steps"),
    "orn_full": ("mix", {"steps": {"ps_pos_old": 1.0, "ps_pos_old_u": 1.0, "px_pos_old_m": 1.0, "pos_age1": 1.0, "o_b": 1.0}},
                 "failed-state / proxy-loss old rows, age-1 rows and class-b rows at full steps"),
    "pxg_full": ("mix", {"steps": {"px_pos_old_g": 1.0, "px_pos_old": 0.6}}, "proxy gain rows at the full step, proxy quiet rows at 0.6"),
    # 11-12: the only direct weighing of proxy / failed-state rows so far (YM) came out at 0.81 of the plain rows where the old ridge had 0.0-0.2, so
    # every other ridge-set proxy step is suspect in the same direction: one bundle for all non-strict old rows outside the cluster plus the class-b proxies
    "op_full": ("mix", {"steps": {"ps_pos_old": 1.0, "ps_pos_old_u": 1.0, "px_pos_old_m": 1.0, "pos_age1": 1.0, "o_b": 1.0, "px_pos_old_g": 1.0, "px_pos_old": 0.6,
                                  "px_b": 0.7, "ps_b": 0.7}},
                "failed-state / proxy old rows of the fire-possible sites at (nearly) full steps, age-1 rows, class-b measured rows 1.0 and class-b proxies 0.7"),
    "Y_full": ("mix", {"steps": {k: 1.0 for k in YS}}, "young / mid rows of the non-strict sites at the full step"),
    "nev_orn": ("mix", {"extra": {"edw": ["--ed-weights", "0,0,0"]}}, "fire-never agb from ORNL alone where both sources exist"),
    "nev_ed": ("mix", {"extra": {"edw": ["--ed-weights", "1,1,1"]}}, "fire-never agb from exact-input ED alone where both sources exist"),
    "B_full": ("recipe", {"b": 1.0}, "rows v6 + memory decode at the full step"),
    "no_B": ("recipe", {"b": 0.0}, "without rows v6 + memory decode"),
    "amp_up": ("mix", {"amp": {"pos": {"a": 1.8}}}, "amplified model with slope 1.8 instead of 1.4"),
    "piv_up": ("mix", {"amp": {"pos": {"q0": 0.09}}}, "pivot 0.09 instead of 0.05"),
    "lvl": ("post", {"id": "lvl", "tool": "apply_level", "factor": 0.976, "strata": "warm,moist,dry,cool,cluster"}, "non-boreal vegetated heights x 0.976"),
    # exaggerated LEVEL PROBES (measurement files, never candidates): x 0.94 on S^2 337k = 1214 squared units -> the level of the block to +- 0.4 %
    "lvl_x": ("post", {"id": "lvlx", "tool": "apply_level", "factor": 0.94, "strata": "warm,moist,dry,cool,cluster"},
              "LEVEL PROBE: non-boreal vegetated heights x 0.94 (kappa' x 6 % = the level error of the block)"),
    "lvl_x_nev": ("post", {"id": "lvlxn", "tool": "apply_level", "factor": 0.92, "classes": "never,b", "strata": "warm,moist,dry,cool"},
                  "LEVEL PROBE: fire-never / class-b non-boreal vegetated heights x 0.92"),
    "lvl_x_pos": ("post", {"id": "lvlxp", "tool": "apply_level", "factor": 0.92, "classes": "pos,cluster", "strata": "warm,moist,dry,cool,cluster"},
                  "LEVEL PROBE: fire-possible / cluster vegetated heights x 0.92"),
    "lvl_x_bor": ("post", {"id": "lvlxb", "tool": "apply_level", "factor": 0.80, "strata": "boreal"}, "LEVEL PROBE: boreal vegetated heights x 0.80"),
    # exaggerated CLUSTER probes (11-12 02:00 UTC): every cluster ORNL part at 1.5 x its deviation from the stack. With a common kappa of the
    # part group, kappa = [sum e_i (2.25 - s_i^2) - (S_probe - S_A)] / (2 sum e_i (1.5 - s_i)), e_i = full-step energies, s_i = steps in A
    # (measured rows: sensitivity 1125 units per unit kappa -> +- 0.13; proxy / failed-state rows: 953 -> +- 0.16)
    "cl_x_meas": ("mix", {"steps": {"cl_old": 1.5, "cl_old_m": 1.5, "cl_mid": 1.5, "cl_mid_m": 1.5, "cl_age1": 1.5}}, "CLUSTER PROBE: measured ORNL rows of the cluster at 1.5 x their deviation from the stack"),
    "cl_x_prox": ("mix", {"steps": {"px_cl": 1.5, "ps_cl": 1.5}}, "CLUSTER PROBE: proxy / failed-state ORNL rows of the cluster at 1.5 x their deviation from the stack"),
    # the same for the non-strict OLD rows of the fire-possible sites (proxy quiet / gain / loss, failed state): all at 1.5 x their deviation
    # (px_pos_old 1.2 here because the base entry xchk already adds 0.3). Sensitivity 2 sum e_i (1.5 - s_i) ~ 2440 units per unit kappa -> +- 0.06
    "op_x": ("mix", {"steps": {"px_pos_old": 1.2, "px_pos_old_g": 1.5, "px_pos_old_m": 1.5, "ps_pos_old": 1.5, "ps_pos_old_u": 1.5}},
             "PROBE: proxy / failed-state old rows of the fire-possible sites at 1.5 x their deviation from the stack"),
    # probes prepared for 11-14 / 11-15 (levers = sum e_i (t - s_i); kappa +- 151 / (2 lever))
    "bor_x": ("mix", {"steps": {"edh_nev_bor": 2.0}}, "PROBE: ED December heights of the boreal fire-never zone at step 2.0 (base 0.85; lever 612 -> kappa +- 0.12)"),
    "pos_x": ("mix", {"steps": {"pos_old": 1.6, "pos_old_m": 1.6, "pos_mid": 1.6, "pos_mid_m": 1.6}}, "PROBE: measured (strict) ORNL rows of the fire-possible sites at 1.6 x their deviation (base 1.0; lever 676 -> +- 0.11)"),
    "nev_x": ("mix", {"steps": {"both_nev": 1.6, "ed_nev": 1.6}}, "PROBE: fire-never agb (ED / ORNL blend and ED-only rows) at 1.6 x the deviation from the stack (base 0.95 / 1.0; lever 1077 -> +- 0.07)"),
    "F_x": ("post", {"id": "ffpos", "tool": "apply_ff", "ff_rows": "@FF", "where": "pos", "h_step": 1.7, "h_old_max": -0.15, "h_forest_strata": "moist,warm"},
            "PROBE: forced-fire ED heights at twice the base step (1.7 instead of 0.85; kappa_F = (3 - dS / E) / 2, +- 0.17)"),
    "b_x": ("mix", {"steps": {"both_b": 1.6, "ed_b": 1.45, "o_b": 1.3, "o_b_m": 1.9}}, "PROBE: class-b agb parts at 0.6 beyond their base steps (1.0 / 0.85 / 0.7 / 1.3)"),
    "alvl_x_pos": ("post", {"id": "alvlxp", "tool": "apply_level", "col": "a", "factor": 0.93, "min": 0.15, "classes": "pos", "strata": "warm,moist,dry,cool,boreal"},
                   "LEVEL PROBE: fire-possible agb x 0.93"),
    "alvl_x_nev": ("post", {"id": "alvlxn", "tool": "apply_level", "col": "a", "factor": 0.95, "min": 0.15, "classes": "never", "strata": "warm,moist,dry,cool,boreal"},
                   "LEVEL PROBE: fire-never agb x 0.95"),
    "alvl_x_b": ("post", {"id": "alvlxb", "tool": "apply_level", "col": "a", "factor": 0.90, "min": 0.15, "classes": "b", "strata": "warm,moist,dry,cool,boreal"}, "LEVEL PROBE: class-b agb x 0.90"),
    "alvl_x_cl": ("post", {"id": "alvlxc", "tool": "apply_level", "col": "a", "factor": 0.88, "min": 0.15, "classes": "cluster", "strata": "cluster"}, "LEVEL PROBE: cluster agb x 0.88"),
    # 11-12: scenario-contrast probes (scripts/apply_contrast.py; x (1 + s) on ssp585 rows, x (1 - s) on ssp126 rows; readout like a level probe: delta = s (1 - dS / E) / 2
    # = half the relative error of the file's ssp585 - ssp126 difference). Old probes leave these open: fire-possible heights +1.4 +- 2.3 %, mature fire-never heights +- 2.3 %
    "hscen_x_pos": ("post", {"id": "hscxp", "tool": "apply_contrast", "col": "h", "s": 0.08, "classes": "pos"}, "CONTRAST PROBE: fire-possible heights x (1 +- 0.08) by scenario (787 units)"),
    # 11-12 07:30 UTC: the joint posterior of the scored files leaves the part probes little to add (op_x 9, pos_x 13, nev_x 15 units; scripts/diag/probe_value.py),
    # block structure that no file has weighed is worth more: scenario contrast of MATURE heights (stack 0 %, main-pass ED +3..+7.5 %), class-b agb level, cold-zone height scale
    "lvl_x_pos2": ("post", {"id": "lvlxp2", "tool": "apply_level", "factor": 0.85, "classes": "pos", "strata": "boreal,cool,warm,moist,dry"},
                   "LEVEL PROBE: fire-possible vegetated heights x 0.85 (S2 129615, public-energy sampling 3.2 %: reads the scale to +- 0.46 %)"),
    "alvl_x_b2": ("post", {"id": "alvlxb", "tool": "apply_level", "col": "a", "factor": 0.80, "min": 0.02, "classes": "b", "strata": "warm,moist,dry,cool,boreal"},
                  "LEVEL PROBE: class-b agb x 0.80 (reads the level to +- 0.5 %)"),
    "lvl_x_cold": ("post", {"id": "lvlxc", "tool": "apply_level", "factor": 0.85, "classes": "never,b", "strata": "boreal,cool"},
                   "LEVEL PROBE: fire-never / class-b vegetated heights of the boreal and cool zones x 0.85 (reads the scale to +- 0.75 %)"),
    "lvl_x_nbw": ("post", {"id": "lvlxw", "tool": "apply_level", "factor": 0.85, "classes": "never,b", "strata": "warm,moist,dry"},
                  "LEVEL PROBE: fire-never / class-b vegetated heights of the warm, moist and dry zones x 0.85 (reads the scale to +- 0.5 %)"),
    "hscen_x_pcm": ("post", {"id": "hscxm", "tool": "apply_contrast", "col": "h", "s": 0.08, "classes": "pos,cluster", "ages": "50:500"},
                    "CONTRAST PROBE: mature (seed ages 50-500) heights of the fire-possible sites and the cluster x (1 +- 0.08) by scenario (731 units; old probes: -2.3 +- 4.1 %)"),
    "hscen_x_nev": ("post", {"id": "hscxn", "tool": "apply_contrast", "col": "h", "s": 0.08, "classes": "never,b", "ages": "50:500"},
                    "CONTRAST PROBE: mature fire-never / class-b heights x (1 +- 0.08) by scenario (964 units)"),
    "ascen_x_nev": ("post", {"id": "ascxn", "tool": "apply_contrast", "col": "a", "s": 0.06, "classes": "never,b", "ages": "150:500"},
                    "CONTRAST PROBE: old fire-never / class-b agb x (1 +- 0.06) by scenario (1369 units)"),
    "ascen_x_pos": ("post", {"id": "ascxp", "tool": "apply_contrast", "col": "a", "s": 0.08, "classes": "pos", "ages": "150:500"}, "CONTRAST PROBE: old fire-possible agb x (1 +- 0.08) by scenario (501 units)"),
    "lvl15": ("post", {"id": "lvl15", "tool": "apply_level", "factor": 0.985, "strata": "warm,moist,dry,cool,cluster"},
              "non-boreal vegetated heights x 0.985 (a candidate bet on the two old hints of a 2.5 % level excess)"),
    # 11-12 01:30 UTC, scripts-free cross-check (scratch ed_crosscheck.py): exact-input ED from the SEED's own state confirms ORNL's deviation from
    # the stack at slope 0.41 (class-b proxies), 0.67 (class-b failed state), 0.53 (quiet fire-possible proxies) - the ridge had 0.0 / 0.1 / 0.2 there
    "xchk": ("mix", {"steps": {"px_b": 0.4, "ps_b": 0.65, "px_pos_old": 0.5}}, "class-b proxy / failed-state rows and quiet fire-possible proxies at the steps exact-input ED supports"),
}


def resolve(nm: str) -> tuple[str, str, dict, str]:
    """Library entry, or a GENERIC block probe written as  lvl:<h|a>:<factor>:<classes>:<zones>[:<agelo-agehi>]  (scripts/apply_level.py) or
    con:<h|a>:<s>:<classes>:<zones>[:<agelo-agehi>]  (scripts/apply_contrast.py); '+' separates classes / zones, '*' = all.
    Returns (safe factor name, kind, payload, text); e.g. lvl:a:0.85:never:warm -> lvl_a_085_never_warm."""
    if nm in LIB:
        return (nm,) + tuple(LIB[nm])
    f = nm.split(":")
    assert f[0] in ("lvl", "con") and len(f) in (5, 6) and f[1] in ("h", "a"), f"unknown factor {nm}"
    val = float(f[2])
    cls = "never,b,pos,cluster" if f[3] == "*" else f[3].replace("+", ",")
    zones = "boreal,cool,warm,moist,dry,cluster" if f[4] == "*" else f[4].replace("+", ",")
    safe = nm.replace(":", "_").replace("+", "-").replace("*", "all").replace(".", "")
    what = "agb" if f[1] == "a" else "vegetated heights"
    if f[0] == "lvl":
        pay = {"id": "g_" + safe, "tool": "apply_level", "factor": val, "classes": cls, "strata": zones}
        if f[1] == "a":
            pay.update({"col": "a", "min": 0.02})
        txt = f"LEVEL PROBE: {what} x {val:g} [{cls} | {zones}"
    else:
        pay = {"id": "g_" + safe, "tool": "apply_contrast", "col": f[1], "s": val, "classes": cls, "strata": zones}
        if f[1] == "h":                                    # only pairs that stay below the 35 m cap (as scripts/direct_batch.py)
            pay["pair_max"] = round(35.0 / 1.14114 / 10.90645 / (1.0 + val) - 1e-6, 6)
        txt = f"CONTRAST PROBE: {what} x (1 +- {val:g}) on ssp585 / ssp126 rows [{cls} | {zones}"
    if len(f) == 6:
        pay["ages"] = f[5].replace("-", ":"); txt += f" | ages {f[5]}"
    return safe, "post", pay, txt + "]"


def posterior(scores: list[float], en: dict[str, float], n: int = 400000, seed: int = 0) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Samples of the four kappas and of E's true score given the five rounded scores (x 0.001)."""
    rng = np.random.default_rng(seed)
    k = {f: rng.normal(PRIOR[f][0], PRIOR[f][1], n) for f in "PLYH"}
    sE = (scores[0] + rng.uniform(-0.5, 0.5, n)) * Q
    g = {f: (2 * k[f] - 1) * en[f] for f in "PLYH"}
    sY = sE - g["Y"]; sLY = sY - g["L"]; sPLY = sLY - g["P"]; sPLYH = sPLY - g["H"]
    ok = np.ones(n, bool)
    for s, obs in ((sY, scores[1]), (sLY, scores[2]), (sPLY, scores[3]), (sPLYH, scores[4])):
        ok &= np.abs(s / Q - obs) <= 0.5
    if ok.sum() < 200:      # the priors hardly reach the data: widen them (the data decide)
        k = {f: rng.normal(PRIOR[f][0], 3 * PRIOR[f][1], n) for f in "PLYH"}
        g = {f: (2 * k[f] - 1) * en[f] for f in "PLYH"}
        sY = sE - g["Y"]; sLY = sY - g["L"]; sPLY = sLY - g["P"]; sPLYH = sPLY - g["H"]
        ok = np.ones(n, bool)
        for s, obs in ((sY, scores[1]), (sLY, scores[2]), (sPLY, scores[3]), (sPLYH, scores[4])):
            ok &= np.abs(s / Q - obs) <= 0.5
        print(f"   (priors widened x 3: {int(ok.sum())} samples accepted)")
    return {f: k[f][ok] for f in "PLYH"}, sE[ok]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="E,Y,LY,PLY,PLYH in units of 0.001")
    ap.add_argument("--factors", default="F,cl_full,orn_full,nev_orn", help="comma list from the library: " + ", ".join(LIB))
    ap.add_argument("--weights", default="", help="override the posterior-mean steps, e.g. P=1,L=1,Y=1.2,H=0")
    ap.add_argument("--layout", default="loo", choices=("loo", "singles"))
    ap.add_argument("--day", type=int, default=50)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--ff-rows", default="")
    ap.add_argument("--design", default="work/logs/day49_design.json")
    ap.add_argument("--anchor", default="auto", choices=("auto", "plain", "amp"))
    ap.add_argument("--extra", default="", help="further files as inclusion patterns over the factors, e.g. 100 = base + first factor only (named after the layout's last file)")
    ap.add_argument("--rows", default="", help="full control of the layout: inclusion patterns over the factors, first = the best-guess file (overrides --layout / --extra)")
    ap.add_argument("--base-mix", default="", help="library entries of kind mix that go into the BASE at weight 1 (not weighed), comma list")
    ap.add_argument("--base-steps", default="", help="builder steps set in the BASE as one mix entry at weight 1, e.g. cl_old=0.4,px_cl=0.35 (use instead of a base-mix entry on the same parts)")
    ap.add_argument("--base-post", default="", help="JSON list (or @file) of further post steps of the BASE, applied after the forced-fire heights and before --base-level, "
                    "e.g. [{\"id\": \"hlvl\", \"tool\": \"apply_hlevel\", \"remove\": \"cold=0.75\"}]")
    ap.add_argument("--factor-json", default="", help="@file with a JSON list of extra factors [{\"name\": ..., \"delta\": recipe delta, \"text\": ...}], appended to --factors "
                    "(e.g. a probe that replaces the base post step 'cal' by id with other values)")
    ap.add_argument("--f-step", type=float, default=0.0, help="h_step of the forced-fire heights in the BASE (default: the library value 0.85)")
    ap.add_argument("--base-level", default="", help="height level factors in the BASE: pos=<f> (fire-possible / cluster), nev=<f> (fire-never / class b non-boreal), bor=<f> (boreal), all=<f> (non-boreal)")
    args = ap.parse_args()
    scores = [float(x) for x in args.scores.split(",")]; assert len(scores) == 5
    pre = args.prefix or f"cand{args.day}"; out = ROOT / (args.out or f"work/logs/day{args.day}_spec.json"); ffr = args.ff_rows or f"work/ff/ff_rows_day{args.day}.npz"
    d49 = json.load(open(ROOT / args.design)); e0 = d49["energies"]
    en = {"P": float(e0["pivot_off"]), "L": float(e0["plain_loss"]), "H": float(e0["no_H"]), "Y": float(e0["YM"])}
    kap, sE = posterior(scores, en)
    print(f"ladder scores (x 0.001): E {scores[0]:.0f}, Y {scores[1]:.0f}, LY {scores[2]:.0f}, PLY {scores[3]:.0f}, PLYH {scores[4]:.0f}; accepted samples {len(sE)}")
    w = {}
    for f, nm in (("Y", "YM"), ("L", "plain_loss"), ("P", "pivot_off"), ("H", "no_H")):
        g = (2 * kap[f] - 1) * en[f]
        w[f] = float(np.clip(kap[f].mean(), *CLIP[f]))
        if abs(w[f]) < 0.12:                              # a step this small is not worth its pair of builds
            w[f] = 0.0
        elif f != "Y" and w[f] > 0.88:
            w[f] = 1.0
        print(f"   {nm:10s} energy {en[f]:4.0f}: gain {g.mean():+5.0f} +- {g.std():3.0f} units, kappa {kap[f].mean():+.2f} +- {kap[f].std():.2f} (prior {PRIOR[f][0]} +- {PRIOR[f][1]}), P(gain > 0) {np.mean(g > 0):.2f} -> step {w[f]:.2f}")
    for item in [x for x in args.weights.split(",") if x]:
        f, v = item.split("="); w[f] = float(v); print(f"   override: step {f} = {w[f]}")
    # predicted score of the base (true scale, quanta)
    sb = sE - sum((2 * kap[f] * w[f] - w[f] ** 2) * en[f] for f in "PLYH")
    best = {"E": sE, "Y": sE - (2 * kap["Y"] - 1) * en["Y"]}
    best["LY"] = best["Y"] - (2 * kap["L"] - 1) * en["L"]; best["PLY"] = best["LY"] - (2 * kap["P"] - 1) * en["P"]; best["PLYH"] = best["PLY"] - (2 * kap["H"] - 1) * en["H"]
    print("   true scores (x 0.001): " + ", ".join(f"{k_} {v.mean() / Q:.2f} +- {v.std() / Q:.2f}" for k_, v in best.items()) + f" | new base {sb.mean() / Q:.2f} +- {sb.std() / Q:.2f}")
    anchor = args.anchor if args.anchor != "auto" else ("plain" if w["P"] + w["L"] >= 1.0 else "amp")
    mix = []
    if anchor == "plain":
        base = {"amp": {"pos": {"a": 1.0, "q0": 0.0}}, "steps": {**HH, **L10, **RA}, "b": 0.5}
        if w["P"] < 1.0:
            mix.append({"id": "P", "w": round(1.0 - w["P"], 4), "delta": {"amp": {"pos": {"a": 1.4, "q0": 0.05, "qmin": -0.10}, "b": {"a": 1.0, "q0": 0.025}}}})
        if w["L"] < 1.0:
            mix.append({"id": "L", "w": round(1.0 - w["L"], 4), "delta": {"amp": {"pos": {"a": 1.4, "q0": 0.05, "qmax": -0.10}}, "steps": L11}})
        if w["Y"] > 0:
            mix.append({"id": "Y", "w": round(w["Y"], 4), "delta": {"steps": YS}})
    else:
        base = {"amp": AMP_E, "steps": {**HH, **L11, **RA}, "b": 0.5}
        if w["P"] != 0:
            mix.append({"id": "P", "w": round(w["P"], 4), "delta": {"amp": {"pos": {"qmax": -0.10}, "b": None}}})
        if w["L"] != 0:
            mix.append({"id": "L", "w": round(w["L"], 4), "delta": {"amp": {"pos": {"qmin": -0.10}}, "steps": L10}})
        if w["Y"] > 0:
            mix.append({"id": "Y", "w": round(w["Y"], 4), "on": PLAIN_ON, "delta": {"steps": YS}})
    if w["H"] > 0:
        mix.append({"id": "H", "w": round(w["H"], 4), "delta": NO_H})
    for nm in [x for x in args.base_mix.split(",") if x]:
        kind, pay, _ = LIB[nm]; assert kind == "mix", nm
        mix.append({"id": nm, "w": 1.0, "delta": pay})
    if args.base_steps:
        mix.append({"id": "base_steps", "w": 1.0, "delta": {"steps": {k_: float(v_) for k_, v_ in (it.split("=") for it in args.base_steps.split(",") if it)}}})
    base["mix"] = mix
    LEVEL_BLOCKS = {"pos": {"classes": "pos,cluster", "strata": "warm,moist,dry,cool,cluster"}, "nev": {"classes": "never,b", "strata": "warm,moist,dry,cool"},
                    "bor": {"strata": "boreal"}, "all": {"strata": "warm,moist,dry,cool,cluster"}}
    base_level = [{"id": "lvl_" + k_, "tool": "apply_level", "factor": float(v_), **LEVEL_BLOCKS[k_]} for k_, v_ in (it.split("=") for it in args.base_level.split(",") if it)]
    names_f = []
    factors = []
    for nm_raw in [x for x in args.factors.split(",") if x]:
        nm, kind, pay, txt = resolve(nm_raw); names_f.append(nm)
        pay = json.loads(json.dumps(pay).replace("@FF", ffr))
        if kind == "mix":
            delta = {"mix": [{"id": nm, "w": 1.0, "delta": pay}]}
        elif kind == "post":
            delta = {"post": [pay]}
        else:
            delta = pay
        factors.append({"name": nm, "delta": delta, "text": txt})
    if args.factor_json:
        extra_f = json.loads((ROOT / args.factor_json.lstrip("@")).read_text())
        for f_ in extra_f:
            factors.append({"name": f_["name"], "delta": f_["delta"], "text": f_.get("text", f_["name"])}); names_f.append(f_["name"])
    k = len(factors)
    if args.layout == "loo":
        include = [[1] * k] + [[0 if j == i else 1 for j in range(k)] for i in range(k)]; names = ["A"] + [f"A{i + 1}" for i in range(k)]
    else:
        include = [[0] * k] + [[1 if j == i else 0 for j in range(k)] for i in range(k)]; names = ["N"] + [f"N{i + 1}" for i in range(k)]
    for pat in [x for x in args.extra.split(",") if x]:
        assert len(pat) == k and set(pat) <= {"0", "1"}, f"--extra pattern {pat} needs {k} digits"
        include.append([int(ch) for ch in pat]); names.append(f"{names[0]}{len(names)}")
    if args.rows:                                          # full control: inclusion patterns over the factors, first row = the best-guess file
        pats = [x for x in args.rows.split(",") if x]
        assert all(len(p_) == k and set(p_) <= {"0", "1"} for p_ in pats), f"--rows patterns need {k} digits"
        include = [[int(ch) for ch in p_] for p_ in pats]; names = ["A"] + [f"A{i}" for i in range(1, len(pats))]
    # forced-fire heights stay in every file unless F is one of the weighed factors
    if "F" not in names_f:
        p = json.loads(json.dumps(LIB["F"][1]).replace("@FF", ffr)); base["post"] = [p]
        if args.f_step > 0:
            p["h_step"] = args.f_step
    if args.base_post:                                     # e.g. the height-level step: after the forced-fire heights, before the level steps and every probe's post step
        txt_ = (ROOT / args.base_post[1:]).read_text() if args.base_post.startswith("@") else args.base_post
        base["post"] = base.get("post", []) + json.loads(txt_)
    if base_level:                                         # level steps come after the forced-fire heights
        base["post"] = base.get("post", []) + base_level
    roots = "work/edv3/ff_pos_srv work/edv3/ff_pos2_srv work/edv3/ff_pos_mac work/edv3/ff_t1 work/edv3/ff_pos_mainpass"
    pre_steps = []
    for r in ("ff_pos_srv", "ff_pos2_srv"):
        for sub, pat in (("csv", "site*.csv"), ("tune", "site*.json")):
            pre_steps.append(f"mkdir -p work/edv3/{r}/{sub}; timeout 120 rsync -a --timeout=60 --include '{pat}' --exclude '*' {HS}/{r}/{sub}/ work/edv3/{r}/{sub}/ 2>/dev/null; true")
    pre_steps.append(f"{PYT} scripts/ed/ff_rows.py {roots} --out {ffr} > work/logs/ff_rows_day{args.day}.log 2>&1 || cp work/ff/ff_rows_day49.npz {ffr}")
    wt = ", ".join(f"{nm} {w[f]:.2f}" for f, nm in (("P", "plain quiet rows"), ("L", "plain loss rows"), ("Y", "young / mid rows"), ("H", "no ED heights warm / moist / dry / class b")))
    spec = {
        "prefix": pre, "names": names, "base": base, "factors": factors, "include": include,
        "base_text": f"base = ladder of 11-12 at its posterior-mean steps ({wt}; anchor {anchor}; scores E {scores[0]:.0f} Y {scores[1]:.0f} LY {scores[2]:.0f} PLY {scores[3]:.0f} PLYH {scores[4]:.0f}), ED December heights in the fire-never zones and at class b, rows v6 + memory decode at half step" + ("" if "F" in names_f else ", forced-fire heights"),
        "min_energy": 20.0, "pre": pre_steps,
        "note": f"written {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by scripts/make_spec_next.py --scores {args.scores} --factors {args.factors} --layout {args.layout}" + (f" --weights {args.weights}" if args.weights else "") + (f" --extra {args.extra}" if args.extra else "") + (f" --rows {args.rows}" if args.rows else "") + (f" --base-mix {args.base_mix}" if args.base_mix else "") + (f" --base-steps {args.base_steps}" if args.base_steps else "") + (f" --base-level {args.base_level}" if args.base_level else ""),
    }
    if out.exists():
        shutil.copy(out, out.with_name(out.name + f".bak_{time.strftime('%H%M', time.gmtime())}"))
    out.write_text(json.dumps(spec, indent=1))
    print(f"wrote {out.relative_to(ROOT)}: prefix {pre}, files {names}, factors {names_f}, layout {args.layout}, anchor {anchor}, mix " + ", ".join(f"{m['id']} {m['w']}" for m in mix))


if __name__ == "__main__":
    main()
