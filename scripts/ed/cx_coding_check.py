#!/usr/bin/env python
"""Coding check of the truth's agb column with ED exact-input runs at NON-BURNING sites.

A site counts as non-burning when ED's own old-stand agb (ages >= 150) does not fall below 0.9 x its year-0 value in
either scenario. For those sites: median log ratio of ED (hybrid for agb: x40 * y0 / e0; raw for height) to the stack C,
per band, C taken as (a) physical (agb "as is") and (b) coded like height (C x 1.14114 = physical). Also the scenario
contrast log(585 / 126) of ED and of C, and the same ratios for the exact twin's training outcome.
    python scripts/ed/cx_coding_check.py work/edv3/cl_cx/results
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

DIV = np.array([10.90645, 3.39567]); M = 1.14114
d = Path(sys.argv[1])
keys = shift.template_keys(); site = keys["site"].to_numpy(); age = keys["age"].to_numpy(); scen = keys["scenario"].to_numpy()
C = shift.read_predictions(ROOT / "results/spatcorr_stack_20261106_s08.csv").astype(float) * DIV      # solution scale x y_mean
R = shift.read_predictions(ROOT / "results/latent_ref_y0_m114114.csv").astype(float) * DIV          # physical
tw = np.load(ROOT / "work/train_twin_states.npz"); tr0 = tw["tr0"][:, :, 0, :].astype(float); tr40 = tw["tr40"].mean(2).astype(float)
idx = {int(s): i for i, s in enumerate(tw["sites"])}
ex = np.load(ROOT / "work/test_cell_exact.npz")
AG = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500])
bands = {"young": np.isin(AG, [10, 20, 30]), "mid": np.isin(AG, [50, 70, 100]), "old": AG >= 150}
done = sorted(set(int(f.name.split("_")[0]) for f in d.glob("*_ssp585.csv")) & set(int(f.name.split("_")[0]) for f in d.glob("*_ssp126.csv")))
rows = []
for s in done:
    if s not in idx:
        continue
    rec = {}
    burn = False
    for sc in ("ssp126", "ssp585"):
        e40 = pd.read_csv(d / f"{s}_{sc}.csv").set_index("age").loc[AG]; e0 = pd.read_csv(d / f"{s}_{sc}_y0.csv").set_index("age").loc[AG]
        m = (site == s) & (scen == sc)
        a0, a40 = e0.agb_kgC.to_numpy(), e40.agb_kgC.to_numpy()
        if a40[AG >= 150].mean() < 0.9 * a0[AG >= 150].mean():
            burn = True
        hyb = a40 * np.clip(R[m, 1] / np.maximum(a0, 1e-6), 0.5, 2.0)
        rec[sc] = dict(hyb=hyb, h40=e40.height_m.to_numpy(), Ca=C[m, 1], Ch=C[m, 0], Ra=R[m, 1], Rh=R[m, 0], seed=np.abs(np.log((a0 + 0.01) / (R[m, 1] + 0.01))))
    rows.append((s, burn, rec))
nb = [(s, r) for s, b, r in rows if not b]
print(f"{len(rows)} sites with twins, non-burning {len(nb)}: {[s for s, _ in nb]}")
print("band   | agb: median log(ED / C)  [as is]   [C x 1.141]   IQR  | height: median log(ED / (C x 1.141))  IQR | twin: log(twin40 / C) agb [as is] [x1.141]  height [x1.141]")
for b, bm in bands.items():
    la, lh, ta, th = [], [], [], []
    for s, r in nb:
        for sc in ("ssp126", "ssp585"):
            x = r[sc]; g = bm & (x["Ca"] > 0.3) & (x["hyb"] > 0.3) & (x["seed"] < 0.25)
            la += list(np.log(x["hyb"][g] / x["Ca"][g])); lh += list(np.log(x["h40"][g] / (x["Ch"][g] * M)))
            t40 = tr40[idx[s]]
            ta += list(np.log(np.maximum(t40[g, 1], 1e-3) / x["Ca"][g])); th += list(np.log(np.maximum(t40[g, 0], 1e-3) / (x["Ch"][g] * M)))
    la, lh, ta, th = map(np.array, (la, lh, ta, th))
    q = lambda v: f"[{np.percentile(v, 25):+.3f} {np.percentile(v, 75):+.3f}]"
    print(f"{b:6s} | n {len(la):4d}  {np.median(la):+.3f}   {np.median(la) - np.log(M):+.3f}   {q(la)} | {np.median(lh):+.3f} {q(lh)} | {np.median(ta):+.3f}  {np.median(ta) - np.log(M):+.3f}   {np.median(th):+.3f}")
print("scenario contrast log(585 / 126), non-burning sites, median [IQR]:")
for b, bm in bands.items():
    ce, cc = [], []
    for s, r in nb:
        g = bm & (r["ssp126"]["Ca"] > 0.3)
        ce += list(np.log(r["ssp585"]["hyb"][g] / r["ssp126"]["hyb"][g])); cc += list(np.log(r["ssp585"]["Ca"][g] / r["ssp126"]["Ca"][g]))
    ce, cc = np.array(ce), np.array(cc)
    print(f"  {b:6s} agb: ED {np.median(ce):+.3f} [{np.percentile(ce, 25):+.3f} {np.percentile(ce, 75):+.3f}]  C {np.median(cc):+.3f} [{np.percentile(cc, 25):+.3f} {np.percentile(cc, 75):+.3f}]  corr {np.corrcoef(ce, cc)[0, 1]:+.2f}")
