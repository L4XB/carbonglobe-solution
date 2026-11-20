#!/usr/bin/env python
"""Per site x scenario, over the strict old-growth pixels (and the relaxed ones as a fallback): q (CLIM_CO2 2060), q_clim (CLIM, no CO2 change, 2060),
q_co2 = q - q_clim, q_2030 (CLIM_CO2 effect already present in 2030). Writes work/ornl2522/q_components.csv."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
z = np.load(ROOT / "work/ornl2522/cells_full.npz"); x = np.load(ROOT / "work/ornl2522/cells_full_extra.npz"); sites = z["sites"]
assert (sites == x["sites"]).all()
f = lambda a: np.where(a == -32768, np.nan, a.astype(np.float32) * np.float32(0.01))
cl = np.load(ROOT / "work/collapse_cluster_labels.npy") == 0; mx = np.load(ROOT / "work/fire/max_factor_rule833.npy")
rows = []
for i0, s in enumerate(sites):
    s = int(s); r0 = f(z["r0"][i0]); r1 = f(z["r1"][i0]); r30 = f(x["r_2030"][i0])
    okx = np.isfinite(r0) & np.isfinite(r1) & (r0 > 0.3) & (r1 > 0)
    g = np.where(okx, np.log(np.maximum(r1, 1e-6) / np.maximum(r0, 1e-6)), np.nan)
    strict = okx & (r0 > 1.0) & (g < 0.12) & (g > -0.2); relax = okx & (r0 > 0.5) & (g < 0.2) & (g > -0.3)
    for sc in ("126", "585"):
        s1 = f(z["s" + sc][i0]); c1 = f(x["c" + sc][i0]); s30 = f(x[f"s{sc}_2030"][i0])
        for nm, sel, nmin in (("strict", strict, 12), ("relax", relax, 6)):
            m = sel & np.isfinite(s1) & (s1 > 0) & np.isfinite(c1) & (c1 > 0) & np.isfinite(s30) & (s30 > 0) & np.isfinite(r30) & (r30 > 0)
            if m.sum() < nmin: continue
            rows.append(dict(site=s, scen="ssp" + sc, src=nm, n=int(m.sum()), q=float(np.median(np.log(s1[m] / r1[m]))), q_clim=float(np.median(np.log(c1[m] / r1[m]))),
                             q_2030=float(np.median(np.log(s30[m] / r30[m]))), b_ref=float(np.median(r1[m]))))
            break
d = pd.DataFrame(rows); d["q_co2"] = d.q - d.q_clim
d["cls"] = np.where(cl[d.site], "cluster", np.where(mx[d.site] >= 1e3, "pos", np.where(mx[d.site] >= 30, "b", "never")))
d.to_csv(ROOT / "work/ornl2522/q_components.csv", index=False)
print(f"{len(d)} site x scenario ({int((d.src == 'strict').sum())} strict)")
print("class / q bin: n | median q, q_clim, q_co2, q_2030 | corr(q, q_clim), sd q_clim, sd q_co2 | share of the 2060 effect present in 2030 (median q_2030 / q)")
for k in ("never", "b", "pos", "cluster"):
    g = d[d.cls == k]
    print(f"   {k:7s} all        : {len(g):5d} | {g.q.median():+.3f} {g.q_clim.median():+.3f} {g.q_co2.median():+.3f} {g.q_2030.median():+.3f} | {np.corrcoef(g.q, g.q_clim)[0, 1]:+.2f} {g.q_clim.std():.3f} {g.q_co2.std():.3f}")
    for nm, lo, hi in (("q < -0.3", -9, -0.3), ("-0.3..-0.1", -0.3, -0.1), ("-0.1..0.1", -0.1, 0.1), ("> 0.1", 0.1, 9)):
        h = g[(g.q >= lo) & (g.q < hi)]
        if len(h) >= 8: print(f"           {nm:11s}: {len(h):5d} | {h.q.median():+.3f} {h.q_clim.median():+.3f} {h.q_co2.median():+.3f} {h.q_2030.median():+.3f} | {np.median(h.q_2030 / h.q):+.2f}")
