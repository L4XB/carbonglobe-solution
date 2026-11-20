#!/usr/bin/env python
"""Move a candidate file towards the forced-fire ED rows (scripts/ed/ff_rows.py) on the cells where those rows exist.

    heights  P <- P + h_step x (ff.h - P)     low / young seeds and burning forest rows of the fire-possible sites (ff_rows 'h')
             P <- P + hq_step x (ff.hq - P)   quiet forest rows (ff_rows 'hq'; default step 0)
    agb      P <- P + a_step x (ff.a - P)     seed ages within --a-ages (default 10:100; the old rows are the tuning target of the forced runs), only at
                                              site x scenarios with a forced old-row effect below --a-old-max, not on low cells (stack agb <= 0.02)

A post-process on purpose: the builder's parts replace a cell's target and step from the stack, here the step runs from the file's own value, so the
factor "forced-fire heights / agb" has the same meaning on top of every file of a weighing design. Heights are capped at 35 m, agb at the floor.

    python scripts/apply_ff.py --inp results/cand49_K.csv --out results/x.csv --h-step 0.7 --a-step 0.5 [--ff-rows work/ff/ff_rows.npz]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

FL_A = 0.401412 / 3.39567
HCAP = 35.0 / 1.14114 / 10.90645


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ff-rows", default="work/ff/ff_rows.npz")
    ap.add_argument("--h-step", type=float, default=0.0)
    ap.add_argument("--hq-step", type=float, default=0.0)
    ap.add_argument("--a-step", type=float, default=0.0)
    ap.add_argument("--a-ages", default="10:100")
    ap.add_argument("--h-forest-strata", default="", help="forest-seed rows of 'h' only in these climate zones (comma list; default all). Training "
                    "validation by zone (11-11): burning forest seeds moist 0.354 -> 0.065, warm 0.184 -> 0.131, DRY 0.150 -> 0.193 (no skill: the "
                    "truth's dry-zone heights do not respond to the agb loss, the forced runs raise them by 0.3-0.7)")
    ap.add_argument("--a-strata", default="", help="agb rows only in these climate zones (default all)")
    ap.add_argument("--a-nonstrict", action="store_true", help="agb rows only at site x scenarios WITHOUT a strict ORNL old-growth effect (rows v6 qsrc of "
                    "the age-300 row not 1 / 3): at strict sites ORNL's state-matched row effects are kept for the young / mid rows")
    ap.add_argument("--h-old-max", type=float, default=9.0, help="forest-seed rows of 'h' only at site x scenarios whose forced old-row effect is below this "
                    "(a forced fire that only trims a large no-fire GAIN to a positive target is not the validated case)")
    ap.add_argument("--a-old-max", type=float, default=9.0, help="agb rows only at site x scenarios whose forced old-row effect is below this (training: the "
                    "young-row agb of the forced runs beats 'chronosequence x old-row effect' only at burning cells: < -0.6: 0.74 -> 0.21, -0.6..-0.3: 0.23 -> "
                    "0.15; at quieter cells 0.05-0.10 -> 0.06-0.24, no gain)")
    ap.add_argument("--max-h-change", type=float, default=0.6, help="largest log change of a height that is applied in full (larger ones are clipped)")
    ap.add_argument("--max-a-change", type=float, default=1.5)
    ap.add_argument("--o-step", type=float, default=0.0, help="old rows (age >= 150) of BURNING site x scenarios: step towards the forced run's within-site row "
                    "pattern at the file's own mean old-row effect")
    ap.add_argument("--o-burn", type=float, default=-0.3, help="burning = the forced run's mean old-row effect against the y0 chronosequence below this")
    ap.add_argument("--oq-step", type=float, default=0.0, help="the same pattern step for the quiet site x scenarios (mean effect above --oq-min)")
    ap.add_argument("--oq-min", type=float, default=-0.15)
    ap.add_argument("--max-o-change", type=float, default=1.0)
    ap.add_argument("--where", default="pos", choices=("pos", "cluster", "both"), help="fire-possible sites outside the collapse cluster (default), the cluster, or both")
    ap.add_argument("--sites", default=None, help="only these sites (file with site numbers)")
    ap.add_argument("--not-sites", default=None, help="not these sites (file with site numbers)")
    args = ap.parse_args()
    P = shift.read_predictions(ROOT / args.inp).astype(float); S = shift.read_predictions(ROOT / "results/spatcorr_stack_20261110_s08.csv").astype(float)
    keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy()
    mx = np.load(ROOT / "work/fire/max_factor_rule833.npy")[site]; cl = (np.load(ROOT / "work/collapse_cluster_labels.npy") == 0)[site]
    pos = {"pos": (mx >= 1e3) & ~cl, "cluster": cl, "both": (mx >= 1e3) | cl}[args.where]
    if args.sites:
        pos &= np.isin(site, np.array([int(x) for x in (ROOT / args.sites).read_text().split()]))
    if args.not_sites:
        pos &= ~np.isin(site, np.array([int(x) for x in (ROOT / args.not_sites).read_text().split()]))
    z = np.load(ROOT / args.ff_rows); Q = P.copy(); msg = []
    strat = None
    if args.h_forest_strata or args.a_strata:
        sys.path.insert(0, str(ROOT / "scripts" / "ed"))
        import dec_eval as de
        strat = de.strata_of_sites()[site]
    for nm, col, arr, step in (("h", 0, z["h"], args.h_step), ("hq", 0, z["hq"], args.hq_step)):
        if step <= 0:
            continue
        m = pos & np.isfinite(arr) & (S[:, 0] > 0.02)
        if nm == "h":
            m &= (z["typ"] != 2) | (np.nan_to_num(z["qa_old"], nan=9.0) < args.h_old_max)
            if args.h_forest_strata:
                m &= (z["typ"] != 2) | np.isin(strat, args.h_forest_strata.split(","))
        t = np.minimum(arr, HCAP)
        lr = np.clip(np.log(np.maximum(t, 1e-3) / np.maximum(P[:, 0], 1e-3)), -args.max_h_change, args.max_h_change)
        t = np.where(m, np.minimum(P[:, 0] * np.exp(lr), HCAP), P[:, 0])
        Q[m, 0] = P[m, 0] + step * (t[m] - P[m, 0])
        msg.append(f"{nm}: {int(m.sum())} cells at {len(np.unique(site[m]))} sites, step {step}, energy {np.sum((Q[m, 0] - P[m, 0]) ** 2):.0f} (full step {np.sum((t[m] - P[m, 0]) ** 2):.0f}), mean log change {np.mean(lr[m]):+.3f}")
    if args.a_step > 0:
        lo, hi = (int(x) for x in args.a_ages.split(":"))
        m = pos & np.isfinite(z["a"]) & (age >= lo) & (age <= hi) & (S[:, 1] > 0.02) & (np.nan_to_num(z["qa_old"], nan=9.0) < args.a_old_max)
        if args.a_strata:
            m &= np.isin(strat, args.a_strata.split(","))
        if args.a_nonstrict:
            qs = np.load(ROOT / "work/ornl2522/rows_v6.npz")["qsrc"].reshape(-1, 15)
            m &= ~np.repeat(np.isin(qs[:, 10], (1, 3)), 15)
        lr = np.clip(np.log(np.maximum(z["a"] + FL_A, 1e-3) / np.maximum(P[:, 1] + FL_A, 1e-3)), -args.max_a_change, args.max_a_change)
        t = np.where(m, (P[:, 1] + FL_A) * np.exp(lr) - FL_A, P[:, 1])
        Q[m, 1] = np.maximum(P[m, 1] + args.a_step * (t[m] - P[m, 1]), -FL_A)
        msg.append(f"a: {int(m.sum())} cells at {len(np.unique(site[m]))} sites, step {args.a_step}, energy {np.sum((Q[m, 1] - P[m, 1]) ** 2):.0f} (full step {np.sum((t[m] - P[m, 1]) ** 2):.0f}), mean log change {np.mean(lr[m]):+.3f}")
    if (args.o_step > 0 or args.oq_step > 0) and "em_a" in z.files:
        # old-row PATTERN: effect of every old row against the y0 chronosequence (kgC, offset 0.05); the forced run's rows keep their deviations from
        # the site x scenario mean, the mean itself stays the file's. Steps apply from the file's own value.
        KG = 3.39567 * 1.204
        key = site.astype(np.int64) * 2 + (keys.scenario.to_numpy() == "ssp585")
        chr_a = np.where(np.isfinite(z["chr_a"]), z["chr_a"], 0.0)
        m0 = pos & np.isfinite(z["em_a"]) & (age >= 150) & (S[:, 1] > 0.02)
        nk = np.bincount(key[m0], minlength=key.max() + 1).astype(float)
        me = np.bincount(key[m0], weights=z["em_a"][m0], minlength=key.max() + 1) / np.maximum(nk, 1)        # forced run: mean old-row effect
        eff = np.log((np.maximum((P[:, 1] + FL_A) * KG, 0) + 0.05) / (chr_a + 0.05))
        mk = np.bincount(key[m0], weights=eff[m0], minlength=key.max() + 1) / np.maximum(nk, 1)              # the file: mean old-row effect
        enough = nk[key] >= 4
        for nm, step, mm in (("o (burning)", args.o_step, me[key] < args.o_burn), ("oq (quiet)", args.oq_step, me[key] > args.oq_min)):
            if step <= 0:
                continue
            m = m0 & enough & mm
            lr = np.clip((mk[key] + z["em_a"] - me[key]) - eff, -args.max_o_change, args.max_o_change)
            t = np.where(m, ((np.maximum((P[:, 1] + FL_A) * KG, 0) + 0.05) * np.exp(np.where(m, lr, 0.0)) - 0.05) / KG - FL_A, P[:, 1])
            Q[m, 1] = np.maximum(P[m, 1] + step * (t[m] - P[m, 1]), -FL_A)
            msg.append(f"{nm}: {int(m.sum())} cells at {len(np.unique(site[m]))} sites, step {step}, energy {np.sum((Q[m, 1] - P[m, 1]) ** 2):.0f} (full step {np.sum((t[m] - P[m, 1]) ** 2):.0f}), rms log change {np.sqrt(np.mean(lr[m] ** 2)) if m.any() else 0:.3f}")
    shift.write_predictions(Q, ROOT / args.out)
    print(f"{args.inp} -> {args.out}: " + " | ".join(msg))


if __name__ == "__main__":
    main()
