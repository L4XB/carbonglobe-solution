#!/usr/bin/env python
"""Memory of the seed miss in the forced-fire runs (training validation rows of ff_train_eval.py): slope of log(truth / ED raw) on log(y0 / ED seed)
by seed-age band and fire intensity, for heights and agb; and the error of the raw state, of the fitted memory decode and of fixed exponents."""
import sys
import numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1] + '/ff_train_eval_rows.csv')
lg = lambda a, b, e: np.log((np.maximum(a, 0) + e) / (np.maximum(b, 0) + e))
d['sh'] = lg(d.y0_h, d.s_h, 0.3); d['sa'] = lg(d.y0_a, d.s_a, 0.05); d['eh'] = lg(d.t_h, d.e_h, 0.1); d['ea'] = lg(d.t_a, d.e_a, 0.01)
d['fire'] = lg(d.t_a, d.c_a, 0.01)       # the truth's agb change against the chronosequence
rms = lambda v: float(np.sqrt(np.mean(np.asarray(v) ** 2)))
bands = ((1, 1), (10, 30), (50, 100), (150, 500))
for nm, col, sc in (('HEIGHT', 'eh', 'sh'), ('AGB', 'ea', 'sa')):
    print(f'{nm}: log(truth / raw) on the seed miss log(y0 / seed)')
    for lo, hi in bands:
        for flo, fhi, fn in ((-9, -0.3, 'burning (truth agb < -0.3)'), (-0.3, 9, 'quiet')):
            g = d[(d.age >= lo) & (d.age <= hi) & (d.fire >= flo) & (d.fire < fhi) & (d[sc].abs() < 1.0)]
            if len(g) < 12 or g[sc].std() < 1e-3: continue
            sl, ic = np.polyfit(g[sc], g[col], 1); res = g[col] - sl * g[sc] - ic
            print(f'   ages {lo:3d}-{hi:3d} {fn:27s} n {len(g):4d} | sd of the seed miss {g[sc].std():.3f} | slope {sl:+.2f} intercept {ic:+.3f} | rms raw {rms(g[col]):.3f} -> with the slope {rms(res + ic):.3f} (centred {rms(res):.3f}) | m = 0.3: {rms(g[col] - 0.3 * g[sc]):.3f}, m = 1 (chrono-like): {rms(g[col] - g[sc]):.3f}')
print('raw error by the size of the seed miss (heights; all ages >= 10):')
for lo, hi in ((0, 0.05), (0.05, 0.15), (0.15, 0.35), (0.35, 9)):
    g = d[(d.age >= 10) & (d.sh.abs() >= lo) & (d.sh.abs() < hi)]
    if len(g) > 10: print(f'   |seed miss h| {lo}-{hi}: n {len(g):4d} rms raw h {rms(g.eh):.3f} agb {rms(g.ea):.3f}')
print('raw error by how well the old-row target was met:')
for lo, hi in ((0, 0.06), (0.06, 0.15), (0.15, 9)):
    g = d[(d.miss >= lo) & (d.miss < hi)]
    if len(g) > 10: print(f'   miss {lo}-{hi}: n {len(g):4d} cells {g.cell.nunique():3d} rms raw h {rms(g.eh):.3f} agb {rms(g.ea):.3f} (old rows agb {rms(g.ea[g.age >= 200]):.3f})')
print('by cell (old rows): truth height change vs ED raw; worst cells first')
c = d[d.age >= 150].groupby('cell').apply(lambda g: pd.Series(dict(fire=g.fire.median(), t_h=g.t_h.median(), e_h=g.e_h.median(), y0_h=g.y0_h.median(), y0_a=g.y0_a.median(), t_a=g.t_a.median(), e_a=g.e_a.median(), err_h=rms(g.eh), L=g.L.iloc[0], cw=g.cw.iloc[0], miss=g.miss.iloc[0])), include_groups=False)
pd.set_option('display.width', 250); print(c.sort_values('err_h', ascending=False).round(3).head(14).to_string())
