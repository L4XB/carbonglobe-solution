#!/usr/bin/env python
"""Per-site view of an ED results root against the stack C, the decoded y0 (R) and the exact twin's training outcome.
Mean over old ages (>= 150) and over young ages (10-30) of agb (kgC) and height (m): ED year-0 (e0), ED year-40 (raw and
hybrid = x40 * y0 / e0), C, R, twin y0 -> y40.
    python scripts/ed/cx_site_table.py work/edv3/cl_cx/results [site list file]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

DIV = np.array([10.90645, 3.39567]); HS = 1.14114
d = Path(sys.argv[1])
only = set(int(l) for l in open(sys.argv[2]) if l.strip()) if len(sys.argv) > 2 else None
keys = shift.template_keys(); site = keys["site"].to_numpy(); age = keys["age"].to_numpy(); scen = keys["scenario"].to_numpy()
C = shift.read_predictions(ROOT / "results/spatcorr_stack_20261106_s08.csv").astype(float) * DIV * np.array([HS, 1.0])   # physical
R = shift.read_predictions(ROOT / "results/latent_ref_y0_m114114.csv").astype(float) * DIV
tw = np.load(ROOT / "work/train_twin_states.npz"); tr0 = tw["tr0"][:, :, 0, :].astype(float); tr40 = tw["tr40"].mean(2).astype(float)
idx = {int(s): i for i, s in enumerate(tw["sites"])}
ex = np.load(ROOT / "work/test_cell_exact.npz")
AG = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500])
bands = {"old": AG >= 150, "yng": np.isin(AG, [10, 20, 30])}
done = sorted(set(int(f.name.split("_")[0]) for f in d.glob("*_ssp585.csv")) & set(int(f.name.split("_")[0]) for f in d.glob("*_ssp126.csv")))
print("site   lat  scen band |  agb: R    e0   ED40  EDhyb    C   | twin0 -> twin40 |  h: R    e0   ED40    C   twin40")
rows = []
for s in done:
    if only is not None and s not in only:
        continue
    for sc in ("ssp126", "ssp585"):
        e40 = pd.read_csv(d / f"{s}_{sc}.csv").set_index("age").loc[AG]; e0 = pd.read_csv(d / f"{s}_{sc}_y0.csv").set_index("age").loc[AG]
        m = (site == s) & (scen == sc)
        for b, bm in bands.items():
            ra, ca = R[m, 1][bm].mean(), C[m, 1][bm].mean(); rh, ch = R[m, 0][bm].mean(), C[m, 0][bm].mean()
            a0, a40 = e0.agb_kgC.to_numpy()[bm].mean(), e40.agb_kgC.to_numpy()[bm].mean(); h0, h40 = e0.height_m.to_numpy()[bm].mean(), e40.height_m.to_numpy()[bm].mean()
            hyb = (e40.agb_kgC.to_numpy()[bm] * np.clip(R[m, 1][bm] / np.maximum(e0.agb_kgC.to_numpy()[bm], 1e-6), 0.5, 2.0)).mean()
            t0 = tr0[idx[s], bm, 1].mean() if s in idx else np.nan; t40 = tr40[idx[s], bm, 1].mean() if s in idx else np.nan; th = tr40[idx[s], bm, 0].mean() if s in idx else np.nan
            rows.append((s, sc, b, ra, a0, a40, hyb, ca, t0, t40, rh, h0, h40, ch, th))
            print(f"{s:5d} {ex['lat'][s]:6.1f} {sc[3:]} {b} | {ra:6.2f} {a0:6.2f} {a40:6.2f} {hyb:6.2f} {ca:6.2f} | {t0:6.2f} -> {t40:6.2f} | {rh:5.1f} {h0:5.1f} {h40:5.1f} {ch:5.1f} {th:5.1f}")
df = pd.DataFrame(rows, columns=["site", "scen", "band", "R", "e0", "ed40", "hyb", "C", "t0", "t40", "Rh", "h0", "h40", "Ch", "th"])
for b in ("old", "yng"):
    x = df[df.band == b]
    print(f"{b}: n {len(x)}  mean agb R {x.R.mean():.2f} EDhyb {x.hyb.mean():.2f} C {x.C.mean():.2f} twin40 {x.t40.mean():.2f} | rms(EDhyb - C) {np.sqrt(((x.hyb - x.C) ** 2).mean()):.2f}  rms(twin40 - C) {np.sqrt(((x.t40 - x.C) ** 2).mean()):.2f}  rms(R - C) {np.sqrt(((x.R - x.C) ** 2).mean()):.2f} "
          f"| corr(EDhyb - R, C - R) {np.corrcoef(x.hyb - x.R, x.C - x.R)[0, 1]:+.2f}  corr(twin40 - t0, C - R) {np.corrcoef(x.t40 - x.t0, x.C - x.R)[0, 1]:+.2f} | h: rms(ED40 - C) {np.sqrt(((x.h40 - x.Ch) ** 2).mean()):.2f} rms(twin - C) {np.sqrt(((x.th - x.Ch) ** 2).mean()):.2f} rms(R - C) {np.sqrt(((x.Rh - x.Ch) ** 2).mean()):.2f}")
