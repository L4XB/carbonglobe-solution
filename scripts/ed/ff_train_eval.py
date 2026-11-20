#!/usr/bin/env python
"""Forced-fire ED on TRAINING cells against the truth (scripts/ed/forced_fire.py --train): year-40 December heights and agb of every seed age.

Per row (cell x seed age; the two "scenarios" are two forced runs of identical inputs): truth, the y0 chronosequence at age + 40 (what the stack's
no-change baseline is), ED raw, and the chronosequence decode (y0 profile at age + 40 x ED's effect against its own first-record profile). Printed by
seed-age band and by the truth's agb change: rms of log(prediction / truth) for heights and agb, the share of rows at the 35 m cap, and the
correlation of the responses.
    python scripts/ed/ff_train_eval.py work/edv3/ff_train1
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
AGES = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]); LA = np.log(AGES)
root = Path(sys.argv[1]); rows = []
for f in sorted((root / 'csv').glob('site*.csv')):
    c = int(f.stem[4:]); tf = root / 'truth' / f'{c}.csv'; jf = root / 'tune' / f'site{c:05d}.json'
    if not tf.exists() or not jf.exists(): continue
    y = pd.read_csv(tf).set_index('age').reindex(AGES); t = pd.read_csv(f); j = json.load(open(jf))
    x = np.log(np.minimum(AGES + 40, 500))
    ch_h = np.exp(np.interp(x, LA, np.log(y.y0_height.to_numpy() + 0.1))) - 0.1; ch_a = np.exp(np.interp(x, LA, np.log(y.y0_agb.to_numpy() + 0.01))) - 0.01
    for sc, g in t.groupby('scenario'):
        g = g.drop_duplicates('age').set_index('age').reindex(AGES)
        if g.end_h_m12.isna().any(): continue
        e_h = np.exp(np.interp(x, LA, np.log(np.maximum(g.pre_h_m12.to_numpy(), 0) + 0.1))); e_a = np.exp(np.interp(x, LA, np.log(np.maximum(g.pre_agb_m12.to_numpy(), 0) + 0.01)))
        qh = np.log((np.maximum(g.end_h_m12.to_numpy(), 0) + 0.1) / e_h); qa = np.log((np.maximum(g.end_agb_m12.to_numpy(), 0) + 0.01) / e_a)
        for i, a in enumerate(AGES):
            rows.append(dict(cell=c, sc=sc, age=a, t_h=y.y40_height.iloc[i], t_a=y.y40_agb.iloc[i], y0_h=y.y0_height.iloc[i], y0_a=y.y0_agb.iloc[i], c_h=ch_h[i], c_a=ch_a[i],
                             e_h=g.end_h_m12.iloc[i], e_a=g.end_agb_m12.iloc[i], s_h=g.pre_h_m12.iloc[i], s_a=g.pre_agb_m12.iloc[i], d_h=(ch_h[i] + 0.1) * np.exp(np.clip(qh[i], -0.9, 0.9)) - 0.1,
                             d_a=(ch_a[i] + 0.01) * np.exp(np.clip(qa[i], -3, 3)) - 0.01, L=j['L'], cw=j['c'][sc], miss=j['miss'][sc], lookup_fa=j['lookup_fa'], target=j['target'][sc]))
d = pd.DataFrame(rows)
if d.empty: print('no finished cells'); sys.exit()
d['d_h'] = d.d_h.clip(upper=35.0)
lg = lambda a, b, e: np.log((np.maximum(a, 0) + e) / (np.maximum(b, 0) + e))
for k, e in (('h', 0.1), ('a', 0.01)):
    d[f'r_ch_{k}'] = lg(d[f'c_{k}'], d[f't_{k}'], e); d[f'r_raw_{k}'] = lg(d[f'e_{k}'], d[f't_{k}'], e); d[f'r_dec_{k}'] = lg(d[f'd_{k}'], d[f't_{k}'], e)
    d[f'seed_{k}'] = np.abs(lg(d[f's_{k}'], d[f'y0_{k}'], 0.3 if k == 'h' else 0.05)); d[f'rt_{k}'] = -d[f'r_ch_{k}']; d[f're_{k}'] = lg(d[f'd_{k}'], d[f'c_{k}'], e)
rms = lambda v: float(np.sqrt(np.mean(np.asarray(v) ** 2)))
print(f'{root}: {d.cell.nunique()} cells, {len(d)} rows (two runs per cell); lookup strength L: ' + ', '.join(f'{v:.0e}: {n}' for v, n in d.drop_duplicates("cell").L.round(9).value_counts().sort_index().items())[:300])
print(f'old-row target met within (median |miss|): {d.drop_duplicates(["cell", "sc"]).miss.median():.3f}; share of cell x run with miss > 0.15: {(d.drop_duplicates(["cell", "sc"]).miss > 0.15).mean():.2f}')
def rep(g, nm):
    if len(g) < 10: return
    cc = np.corrcoef(g.rt_h, g.re_h)[0, 1] if g.rt_h.std() > 0 and g.re_h.std() > 0 else float('nan')
    print(f'{nm:34s} n {len(g):4d} | HEIGHT rms log err: chrono {rms(g.r_ch_h):.3f} ED raw {rms(g.r_raw_h):.3f} decode {rms(g.r_dec_h):.3f} (bias {g.r_dec_h.mean():+.3f}) corr of responses {cc:+.2f} | at the cap (>= 34.9 m): truth {np.mean(g.t_h >= 34.9):.2f} ED {np.mean(g.e_h >= 34.9):.2f} | '
          f'AGB rms: chrono {rms(g.r_ch_a):.3f} raw {rms(g.r_raw_a):.3f} decode {rms(g.r_dec_a):.3f} (bias {g.r_dec_a.mean():+.3f})')
ok = (d.seed_h < 0.15) & (d.seed_a < 0.25)
rep(d, 'all rows'); rep(d[ok], 'seed matches (h < 0.15, agb < 0.25)'); rep(d[~ok], 'seed does not match')
for lo, hi, nm in ((1, 1, 'age 1'), (10, 30, 'ages 10-30'), (50, 100, 'ages 50-100'), (150, 500, 'ages 150-500')):
    rep(d[ok & (d.age >= lo) & (d.age <= hi)], f'   seed ok, {nm}')
for lo, hi, nm in ((-9, -1.0, 'truth agb change < -1'), (-1.0, -0.3, '-1..-0.3'), (-0.3, -0.05, '-0.3..-0.05'), (-0.05, 9, '> -0.05')):
    rep(d[ok & (d.rt_a >= lo) & (d.rt_a < hi)], f'   seed ok, {nm}')
rep(d[ok & (d.miss < 0.15)], 'seed ok, old-row target met')
rep(d[ok & (d.y0_h >= 30) & (d.y0_a <= 6)], 'seed ok, burned-state seeds')
d.to_csv(root / 'ff_train_eval_rows.csv', index=False)
