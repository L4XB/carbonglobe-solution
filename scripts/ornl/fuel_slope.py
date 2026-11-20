#!/usr/bin/env python
"""Does the ORNL SSP effect of a pixel depend on its biomass (fuel)? Within each cell x scenario: slope of log(B2060_ssp / B2060_REF) on the pixel's REF biomass
across slow-growing existing-forest pixels (REF growth < 0.2), by fire class; plus the fuel ratio (old pixels' biomass / the site's old y0 agb) of the measured rows."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "ed")); sys.path.insert(0, str(ROOT / "scripts" / "ornl"))
from cgb import io, shift  # noqa: E402
import dec_eval as de, ornl_rows as orr  # noqa: E402
z = np.load(ROOT / "work/ornl2522/cells_full.npz"); sites = z["sites"]
f = lambda a: np.where(a == -32768, np.nan, a.astype(np.float32) * np.float32(0.01))
cl = np.load(ROOT / "work/collapse_cluster_labels.npy") == 0; mx = np.load(ROOT / "work/fire/max_factor_rule833.npy")
_, y1 = io.open_test("ssp126"); FL = shift.FLOORS; M = shift.M_HEIGHT
Yold = ((y1[:, :, 1] - FL[1]) * M * orr.F_AGB)[:, 7:].mean(axis=1)
rows = []
for i0, s in enumerate(sites):
    s = int(s); r0 = f(z["r0"][i0]); r1 = f(z["r1"][i0])
    okx = np.isfinite(r0) & np.isfinite(r1) & (r0 > 0.5) & (r1 > 0)
    g = np.where(okx, np.log(np.maximum(r1, 1e-6) / np.maximum(r0, 1e-6)), np.nan)
    slow = okx & (g < 0.2) & (g > -0.3)
    for sc in ("126", "585"):
        s1 = f(z["s" + sc][i0]); m = slow & np.isfinite(s1) & (s1 > 0)
        if m.sum() < 40 or np.std(r1[m]) < 0.15 * np.mean(r1[m]): continue
        lq = np.log(s1[m] / r1[m]); x = r1[m]
        lo, hi = np.quantile(x, [0.2, 0.8])
        b = np.polyfit(x, lq, 1)
        rows.append(dict(site=s, scen=sc, n=int(m.sum()), fuel=float(np.median(x)), q=float(np.median(lq)), q_lo=float(np.median(lq[x <= lo])), q_hi=float(np.median(lq[x >= hi])),
                         f_lo=float(np.median(x[x <= lo])), f_hi=float(np.median(x[x >= hi])), slope=float(b[0]), yold=float(Yold[s])))
d = pd.DataFrame(rows); d["cls"] = np.where(cl[d.site], "cluster", np.where(mx[d.site] >= 1e3, "pos", np.where(mx[d.site] >= 30, "b", "never")))
d["dq"] = d.q_hi - d.q_lo; d["rel"] = d.dq / np.log(d.f_hi / d.f_lo)           # change of the effect per e-fold of fuel
d.to_csv(ROOT / "work/ornl2522/fuel_slope.csv", index=False)
print("class: cells x scenario | median fuel (kgC), median y0 old agb | median effect q | q of the low-fuel fifth -> high-fuel fifth (medians) | median change per e-fold of fuel | share with a more negative effect at high fuel")
for k in ("never", "b", "pos", "cluster"):
    g = d[d.cls == k]
    if len(g) < 10: continue
    print(f"   {k:7s}: {len(g):5d} | {g.fuel.median():5.2f} {g.yold.median():5.2f} | {g.q.median():+.3f} | {g.q_lo.median():+.3f} -> {g.q_hi.median():+.3f} | {g.rel.median():+.3f} | {np.mean(g.dq < 0):.2f}")
    for nm, m in (("q < -0.2", g.q < -0.2), ("-0.2..0", (g.q >= -0.2) & (g.q < 0)), ("0..0.1", (g.q >= 0) & (g.q < 0.1)), ("> 0.1", g.q >= 0.1)):
        h = g[m]
        if len(h) >= 10: print(f"        cells with q {nm:8s}: {len(h):4d} | low-fuel fifth {h.q_lo.median():+.3f} (fuel {h.f_lo.median():.1f}) -> high-fuel fifth {h.q_hi.median():+.3f} (fuel {h.f_hi.median():.1f}); change per e-fold {h.rel.median():+.3f}; fuel / y0 old {np.median(h.fuel / h.yold):.2f}")
