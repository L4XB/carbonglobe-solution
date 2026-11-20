#!/usr/bin/env python
"""REF growth 2021 -> 2060 of the OLDEST forest states near the node of every exact site: is the product's reference run in equilibrium with the lookup?

If the pixel value is cover x table(state), then for a pixel in the oldest state REF60 / REF21 = lookup(old + 40) / lookup(old) (~1 for a flat old-growth
profile) PROVIDED the reference run lives in the lookup's world. A negative value at fire-possible sites would mean the reference run burns where the
lookup does not - the truth relative to the lookup's chronosequence is then exp(q + delta) with delta = that REF change, i.e. the "pivot" of the
amplified model, site by site.
Old states are taken as the pixels in the upper part of the REF 2021 biomass distribution near the node (cover ~1, tall canopy): per site the median
of g = log(REF60 / REF21) over the pixels within 0.1 deg of the node whose REF 2021 biomass is above the 70th / 90th percentile of the window's
forest pixels (>= 12 pixels), together with the y0 profile's own growth log(y0(500) / y0(300)) scaled to 40 years.
Writes work/ornl2522/node_ref_growth.npz.   python scripts/ornl/node_ref_growth.py
"""
import sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src')); sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/ed'))
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]; D = ROOT / 'work/ornl2522'
sites = np.load(D / 'node_win_sites.npy'); n = len(sites)
L = {k: np.load(D / f'node_win_{k}.npy', mmap_mode='r') for k in ('r0', 'r1', 's126', 's585')}
f = lambda a: np.where(a == -32768, np.nan, a.astype(np.float32) * np.float32(0.01))
d = (np.arange(100) + 0.5 - 50) / 50.0; dc = np.maximum(np.abs(d)[:, None], np.abs(d)[None, :]); cen = dc <= 0.2; cell = dc <= 0.5
out = {k: np.full(n, np.nan) for k in ('g70', 'g90', 'g70_cell', 'b70', 'n70', 'gall', 'q585_70', 'q126_70')}
for i in range(n):
    r0 = f(L['r0'][i]); r1 = f(L['r1'][i]); ok = np.isfinite(r0) & np.isfinite(r1) & (r0 > 0.5) & (r1 > 0)
    if ok.sum() < 40: continue
    g = np.where(ok, np.log(np.maximum(r1, 1e-6) / np.maximum(r0, 1e-6)), np.nan)
    thr70, thr90 = np.quantile(r0[ok & cell], [0.7, 0.9]) if (ok & cell).sum() >= 20 else (np.nan, np.nan)
    m70 = ok & cen & (r0 >= thr70); m90 = ok & cen & (r0 >= thr90); mc = ok & cell & (r0 >= thr70)
    out['gall'][i] = np.nanmedian(g[ok & cen]) if (ok & cen).sum() >= 12 else np.nan
    if m70.sum() >= 12:
        out['g70'][i] = np.median(g[m70]); out['b70'][i] = np.median(r0[m70]); out['n70'][i] = m70.sum()
        for k, sc in (('q126_70', 's126'), ('q585_70', 's585')):
            s1 = f(L[sc][i]); mm = m70 & np.isfinite(s1) & (s1 > 0)
            if mm.sum() >= 12: out[k][i] = np.median(np.log(s1[mm] / r1[mm]))
    if m90.sum() >= 8: out['g90'][i] = np.median(g[m90])
    if mc.sum() >= 20: out['g70_cell'][i] = np.median(g[mc])
np.savez(D / 'node_ref_growth.npz', sites=sites, **out)
from cgb import shift, io
import dec_eval as de
mx = np.load(ROOT / 'work/fire/max_factor_rule833.npy'); cl = np.load(ROOT / 'work/collapse_cluster_labels.npy') == 0; strat = de.strata_of_sites()
pr = np.load(ROOT / 'work/ff/prior_lookup_ratio.npy')[:, 0]
cls = np.where(cl[sites], 'cluster', np.where(mx[sites] >= 1e3, 'pos', np.where(mx[sites] >= 30, 'b', 'never')))
print('REF growth 2021 -> 2060 (log) of the high-biomass pixels near the node, per site: median [p25, p75] by class (pixels above the 70th / 90th percentile of REF 2021 biomass; all pixels)')
for c in ('never', 'b', 'pos', 'cluster'):
    for zz in ('boreal', 'cool', 'warm', 'moist', 'dry'):
        m = (cls == c) & (strat[sites] == zz) & np.isfinite(out['g70'])
        if m.sum() < 15: continue
        a, b, e = out['g70'][m], out['g90'][m], out['gall'][m]
        print(f'  {c:8s} {zz:6s} n {int(m.sum()):4d} | top 30 %: {np.median(a):+.3f} [{np.percentile(a, 25):+.3f}, {np.percentile(a, 75):+.3f}] | top 10 %: {np.nanmedian(b):+.3f} | all: {np.nanmedian(e):+.3f} | REF 2021 biomass of the top 30 %: {np.median(out["b70"][m]):.1f} kgC')
m = (cls == 'pos') & np.isfinite(out['g70'])
for nm, mm in (('fire-possible, unburned lookup', m & (pr[sites] <= 0.12)), ('fire-possible, burned lookup', m & (pr[sites] > 0.12))):
    a = out['g70'][mm]; q = out['q585_70'][mm]
    print(f'  {nm}: n {int(mm.sum())} | top-30 % REF growth median {np.median(a):+.3f} [{np.percentile(a, 25):+.3f}, {np.percentile(a, 75):+.3f}]; corr with their ssp585 effect {np.corrcoef(a[np.isfinite(q)], q[np.isfinite(q)])[0, 1]:+.2f}')
    for lo, hi in ((-9, -0.3), (-0.3, -0.1), (-0.1, 0.1), (0.1, 9)):
        k = np.isfinite(q) & (q >= lo) & (q < hi)
        if k.sum() > 10: print(f'      ssp585 effect [{lo:+.1f}, {hi:+.1f}): n {int(k.sum()):4d} REF growth {np.median(a[k]):+.3f}')
