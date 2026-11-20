#!/usr/bin/env python
"""Make the young / mid rows and the heights of BURNING site x scenarios consistent with the file's own old-row agb loss, where no forced-fire row
exists, using the TRAINING truth of the test sites' twins (work/twin/resp_train.npz).

  agb      seed ages 10-100 (optionally age 1) of fire-possible site x scenarios whose OLD rows carry a mean effect below --burn (-0.3) against the
           y0 chronosequence: the row's effect may not sit above  old-row mean + training profile(age, loss)  (medians of the twins' truth: young
           stands lose less than old ones, +0.57 / +0.26 / +0.11 at ages 10 / 50 / 100 for losses beyond -1, +0.26 / +0.13 / +0.09 at -1..-0.6,
           +0.13 / +0.07 / +0.05 at -0.6..-0.3). One-sided: rows already below that value keep theirs (strict ORNL rows are flat; the board's
           cluster steps back flat), rows above it are the stack's no-loss values at sites whose old rows took a proxy loss.
  heights  forest-type rows (lookup at age + 40 >= 12 m, not tall + light) in the moist / warm zones whose own agb effect is below --burn: the height
           may not sit below  chronosequence x exp(training relation(zone, agb effect))  (moist +0.20 / +0.32 / +0.35 at losses of 0.3-0.6 / 0.6-1 /
           beyond 1, warm +0.26 / +0.28 / +0.59; the stack answers +0.09..+0.16). One-sided as well.
Cells that carry a forced-fire row (--ff-rows: 'a' for the agb part, 'h' for the heights) are left alone.

    python scripts/apply_profile.py --inp results/x.csv --out results/y.csv --a-step 0.7 --h-step 0.7 [--ff-rows work/ff/ff_rows.npz]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "ed"))
from cgb import shift  # noqa: E402
import dec_eval as de  # noqa: E402

FL_A = 0.401412 / 3.39567; KG = 3.39567 * 1.204; HC = 1.14114 * 10.90645
AGES = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500])
QB = np.array([-1.4, -0.8, -0.45, -0.2])        # centres of the loss bins (old-row mean effect); the offset fades to 0 at -0.2
EDGES = ((-9.0, -1.0), (-1.0, -0.6), (-0.6, -0.3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ff-rows", default="")
    ap.add_argument("--a-step", type=float, default=0.0); ap.add_argument("--h-step", type=float, default=0.0)
    ap.add_argument("--burn", type=float, default=-0.3)
    ap.add_argument("--age1", action="store_true", help="include the age-1 rows in the agb part")
    ap.add_argument("--h-strata", default="moist,warm")
    ap.add_argument("--margin", type=float, default=0.05, help="a row is changed only if it is on the wrong side of the consistent value by more than this (log)")
    args = ap.parse_args()
    P = shift.read_predictions(ROOT / args.inp).astype(float); S = shift.read_predictions(ROOT / "results/spatcorr_stack_20261110_s08.csv").astype(float)
    keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy()
    assert (age.reshape(-1, 15) == AGES[None, :]).all()
    mx = np.load(ROOT / "work/fire/max_factor_rule833.npy")[site]; cl = (np.load(ROOT / "work/collapse_cluster_labels.npy") == 0)[site]
    pos = (mx >= 1e3) & ~cl; strat = de.strata_of_sites()[site]
    c = np.load(ROOT / "work/test_chrono_rows.npz"); ch_a = np.maximum(c["ch_a"], 0.0); ch_h = np.minimum(c["ch_h"], 35.0)
    tr = np.load(ROOT / "work/twin/resp_train.npz"); ra = tr["r_a"].reshape(-1, 15); rh = tr["r_h"].reshape(-1, 15)
    ff = np.load(ROOT / args.ff_rows) if args.ff_rows else None
    has_a = np.isfinite(ff["a"]) if ff is not None else np.zeros(len(P), bool); has_h = np.isfinite(ff["h"]) if ff is not None else np.zeros(len(P), bool)
    old = AGES >= 150; pos2 = pos.reshape(-1, 15)[:, 0]; Q = P.copy(); msg = []
    kg = np.maximum((P[:, 1] + FL_A) * KG, 0.0); eff = np.log((kg + 0.05) / (ch_a + 0.05)); eff2 = eff.reshape(-1, 15); om = eff2[:, old].mean(axis=1)
    # --- training tables
    tm = np.nanmean(ra[:, old], axis=1); okt = pos2 & np.isfinite(ra).all(axis=1)
    prof = np.zeros((4, 15))
    for i, (lo, hi) in enumerate(EDGES):
        m = okt & (tm >= lo) & (tm < hi); prof[i] = np.median(ra[m] - tm[m, None], axis=0)
    if args.a_step > 0:
        off = np.array([[np.interp(q, QB, prof[:, j]) for j in range(15)] for q in np.clip(om, QB[0], QB[-1])])       # (site x scenario, 15)
        tgt_eff = om[:, None] + off
        rows = (pos2 & (om < args.burn))[:, None] & ((AGES >= (1 if args.age1 else 10)) & (AGES <= 100))[None, :]
        m = rows.reshape(-1) & ~has_a & (S[:, 1] > 0.02) & (eff > tgt_eff.reshape(-1) + args.margin)
        t = ((ch_a + 0.05) * np.exp(tgt_eff.reshape(-1)) - 0.05) / KG - FL_A
        Q[m, 1] = np.maximum(P[m, 1] + args.a_step * (t[m] - P[m, 1]), -FL_A)
        msg.append(f"agb profile: {int(m.sum())} cells at {len(np.unique(site[m]))} sites, step {args.a_step}, energy {np.sum((Q[m, 1] - P[m, 1]) ** 2):.0f} (full step {np.sum((t[m] - P[m, 1]) ** 2):.0f}), mean log change at the full step {np.mean((tgt_eff.reshape(-1) - eff)[m]) if m.any() else 0:+.3f}")
    if args.h_step > 0:
        typ = np.where((ch_h >= 30) & (ch_a <= 6) & (ch_a >= 0.25), 3, np.where(ch_h >= 12, 2, 1)); forest = (typ == 2) & (age >= 30)
        lh = np.log(np.maximum(P[:, 0] * HC, 0.3) / np.maximum(ch_h, 0.3)); tgt_h = np.full(len(P), np.nan)
        ra1 = ra.reshape(-1); rh1 = rh.reshape(-1)
        for z in args.h_strata.split(","):
            zm = pos & (strat == z) & forest
            rel = [float(np.median(rh1[zm & np.isfinite(rh1) & (ra1 >= lo) & (ra1 < hi)])) for lo, hi in EDGES] + [0.0]
            tgt_h = np.where(zm, np.interp(np.clip(eff, QB[0], QB[-1]), QB, np.array(rel)), tgt_h)
            msg.append(f"training height relation {z}: " + " / ".join(f"{v:+.3f}" for v in rel[:3]))
        m = np.isfinite(tgt_h) & (eff < args.burn) & ~has_h & (S[:, 0] > 0.02) & (lh < tgt_h - args.margin)
        t = np.minimum(np.maximum(ch_h, 0.3) * np.exp(np.nan_to_num(tgt_h)), 35.0) / HC
        Q[m, 0] = P[m, 0] + args.h_step * (t[m] - P[m, 0])
        msg.append(f"heights: {int(m.sum())} cells at {len(np.unique(site[m]))} sites, step {args.h_step}, energy {np.sum((Q[m, 0] - P[m, 0]) ** 2):.0f} (full step {np.sum((t[m] - P[m, 0]) ** 2):.0f}), mean log change at the full step {np.mean((tgt_h - lh)[m]) if m.any() else 0:+.3f}")
    shift.write_predictions(Q, ROOT / args.out)
    print(f"{args.inp} -> {args.out}: " + " | ".join(msg))


if __name__ == "__main__":
    main()
