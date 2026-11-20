#!/usr/bin/env python
"""Proxies of the old-growth effect q_old for cells WITHOUT old-growth pixels (rows v3: regression on q_new), from the full-resolution cells:
  q_all   median log(B2060_ssp / B2060_REF) over all existing-forest pixels (REF 2021 > 0.3)
  q_top   the same over the top 25 % of the existing pixels by REF 2060 biomass
  q_slow  over the pixels in the lowest third of REF growth (>= 8 pixels)
  q_rel   relaxed old-growth definition (REF 2021 > 0.5, REF growth < 0.2, >= 6 pixels)
Evaluated against the measured q_old where it exists, by fire class. Writes work/ornl2522/q_proxies.csv."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "ed"))
import dec_eval as de  # noqa: E402
z = np.load(ROOT / "work/ornl2522/cells_full.npz"); sites = z["sites"]
f = lambda a: np.where(a == -32768, np.nan, a.astype(np.float32) * np.float32(0.01))
cl = np.load(ROOT / "work/collapse_cluster_labels.npy") == 0; mx = np.load(ROOT / "work/fire/max_factor_rule833.npy"); strat = de.strata_of_sites()
rows = []
for i0, s in enumerate(sites):
    s = int(s); r0 = f(z["r0"][i0]); r1 = f(z["r1"][i0]); nr = f(z["nr"][i0])
    okx = np.isfinite(r0) & np.isfinite(r1) & (r0 > 0.3) & (r1 > 0)
    g = np.where(okx, np.log(np.maximum(r1, 1e-6) / np.maximum(r0, 1e-6)), np.nan)
    old = okx & (r0 > 1.0) & (g < 0.12) & (g > -0.2)
    for sc in ("126", "585"):
        s1 = f(z["s" + sc][i0]); n1 = f(z["n" + sc][i0])
        m = okx & np.isfinite(s1) & (s1 > 0); lq = np.where(m, np.log(np.maximum(s1, 1e-6) / np.maximum(r1, 1e-6)), np.nan)
        rec = dict(site=s, scen="ssp" + sc, n_exist=int(m.sum()), n_old=int((old & m).sum()))
        rec["q_old"] = float(np.median(lq[old & m])) if (old & m).sum() >= 12 else np.nan
        rec["q_all"] = float(np.median(lq[m])) if m.sum() >= 12 else np.nan
        if m.sum() >= 24:
            thr = np.quantile(r1[m], 0.75); mt = m & (r1 >= thr); rec["q_top"] = float(np.median(lq[mt]))
            gthr = np.quantile(g[m], 1 / 3); ms = m & (g <= gthr); rec["q_slow"] = float(np.median(lq[ms])); rec["g_slow"] = float(np.median(g[ms])); rec["b_top"] = float(np.median(r1[mt]))
        mr = m & (r0 > 0.5) & (g < 0.2) & (g > -0.3)
        rec["q_rel"] = float(np.median(lq[mr])) if mr.sum() >= 6 else np.nan
        mn = np.isfinite(nr) & np.isfinite(n1) & (nr > 0.3) & (n1 > 0)
        rec["q_new"] = float(np.median(np.log(n1[mn] / nr[mn]))) if mn.sum() >= 12 else np.nan
        rows.append(rec)
d = pd.DataFrame(rows); d["cls"] = np.where(cl[d.site], "cluster", np.where(mx[d.site] >= 1e3, "pos", np.where(mx[d.site] >= 30, "b", "never"))); d["strat"] = strat[d.site.to_numpy()]
d.to_csv(ROOT / "work/ornl2522/q_proxies.csv", index=False)
print(f"{len(d)} site x scenario; measured q_old {int(d.q_old.notna().sum())}")
print("class: n with q_old | per proxy: n both, corr, slope of q_old on the proxy, rms residual | coverage of the proxy where q_old is missing")
for k in ("pos", "cluster", "b", "never"):
    g = d[d.cls == k]; line = f"   {k:7s}: {int(g.q_old.notna().sum()):5d} |"
    for p in ("q_all", "q_top", "q_slow", "q_rel", "q_new"):
        m = g.q_old.notna() & g[p].notna()
        if m.sum() < 20: line += f" {p}: -"; continue
        b = np.polyfit(g[p][m], g.q_old[m], 1); res = g.q_old[m] - np.polyval(b, g[p][m])
        line += f" {p}: {int(m.sum())} r {np.corrcoef(g[p][m], g.q_old[m])[0, 1]:+.2f} slope {b[0]:.2f} icpt {b[1]:+.3f} rms {res.std():.3f} (miss-cover {int((g.q_old.isna() & g[p].notna()).sum())}) |"
    print(line)
