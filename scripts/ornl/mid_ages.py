"""Scenario effect q(age) for the intermediate seed ages: ED's own q (against the lookup chronosequence at age + 40)
compared with (a) ORNL q_old / q_new interpolation and (b) ORNL pixels matched by their REF growth."""
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts/ed'); sys.path.insert(0, 'scripts/ornl')
import numpy as np, pandas as pd
from pathlib import Path
from cgb import io, shift
import dec_eval as de
import ornl_rows as orr
ROOT = Path('.')
ex = np.load('work/test_cell_exact.npz'); lat, lon = ex['lat'], ex['lon']
R0 = orr.p2('Biomass_REF_existing_forest_2021_2030', 0); R1 = orr.p2('Biomass_REF_existing_forest_2051_2060', 9)
S1 = {sc: orr.p2(f'Biomass_GFDL-ESM4_SSP{sc}_CLIM_CO2_existing_forest_2051_2060', 9) for sc in ('126', '585')}
qo = pd.read_csv('work/ornl2522/exact_q_old.csv').set_index('site'); qn = pd.read_csv('work/ornl2522/exact_q_new.csv').set_index('site')
_, y1 = io.open_test('ssp126'); _, y5 = io.open_test('ssp585'); FL = shift.FLOORS; M = shift.M_HEIGHT
Y = {'126': (y1[:, :, 1] - FL[1]) * M * 1.055, '585': (y5[:, :, 1] - FL[1]) * M * 1.055}
AGES = orr.AGES
mx = np.load('work/fire/max_factor_rule833.npy'); cl = np.load('work/collapse_cluster_labels.npy') == 0; strat = de.strata_of_sites()
parts = [de.load_root(ROOT / r, 'dec') for r in ('work/edv3/full_cx16', 'work/edv3/fire_cx17b')]
seen = set(); keep = []
for q in parts:
    q = q[~q.site.isin(seen)]; seen |= set(q.site.unique()); keep.append(q)
t = pd.concat(keep, ignore_index=True)
t = t[(mx[t.site.to_numpy()] < 30) & ~cl[t.site.to_numpy()]]
rows = []
for (s, sc), g in t.groupby(['site', 'scenario']):
    k = sc[3:]
    if s not in qo.index or s not in qn.index or not np.isfinite(qo.loc[s, 'q' + k]) or not np.isfinite(qn.loc[s, 'q' + k]): continue
    y = Y[k][s]
    r0 = orr.cell(R0, lat[s], lon[s]); r1 = orr.cell(R1, lat[s], lon[s]); s1 = orr.cell(S1[k], lat[s], lon[s])
    ok = np.isfinite(r0) & np.isfinite(r1) & np.isfinite(s1) & (r0 > 0.3) & (r1 > 0) & (s1 > 0)
    gR = np.log(r1[ok] / r0[ok]); qpx = np.log(s1[ok] / r1[ok])
    g = g.set_index('age')
    for a in AGES:
        if a not in g.index: continue
        row = g.loc[a]
        mis = abs(np.log((row.y0_agb * 1.055 + 0.01) / (max(row.e0_a, 0) + 0.01)))
        if a > 1 and mis > 0.07: continue
        ed = row.e40_a * (row.y0_agb * 1.055 + 0.01) / (max(row.e0_a, 0) + 0.01) if a > 1 else row.e40_a - (row.e0_a - row.y0_agb * 1.055)
        base = orr.profile(y, a + 40)
        gstar = np.log(orr.profile(y, a + 40) / orr.profile(y, a + 1))
        w = np.abs(gR - gstar) <= max(0.25 * abs(gstar), 0.04)
        rows.append(dict(site=s, scen=sc, age=a, st=strat[s], qed=float(np.log(max(ed, 0.01) / base)), qold=qo.loc[s, 'q' + k], qnew=qn.loc[s, 'q' + k], gstar=gstar, nmatch=int(w.sum()), qmatch=float(np.median(qpx[w])) if w.sum() >= 5 else np.nan))
r = pd.DataFrame(rows)
print('rows', len(r), 'site-scenarios', r.groupby(['site', 'scen']).ngroups)
print('age | n | median qED, q_old, q_new | w = (qED - q_old) / (q_new - q_old) by regression | corr(qED, interpolation) | rows with >= 5 matched pixels: share, corr(qED, q_match), median diff, sd diff | sd(qED - interp)')
W = {}
for a in AGES:
    g = r[(r.age == a) & r.st.isin(['moist', 'warm'])]
    if len(g) < 30: continue
    x = (g.qnew - g.qold).to_numpy(); yv = (g.qed - g.qold).to_numpy()
    w = float(np.sum(x * yv) / np.sum(x * x)); W[int(a)] = w
    interp = g.qold + w * (g.qnew - g.qold)
    m = g.qmatch.notna()
    print(f'{a:4d} | {len(g):4d} | {g.qed.median():+.3f} {g.qold.median():+.3f} {g.qnew.median():+.3f} | w {w:+.2f} | {np.corrcoef(g.qed, interp)[0, 1]:+.2f} | {m.mean():.2f}, ' + (f'{np.corrcoef(g.qed[m], g.qmatch[m])[0, 1]:+.2f}, {np.median(g.qmatch[m] - g.qed[m]):+.3f}, {np.std(g.qmatch[m] - g.qed[m]):.3f}' if m.sum() > 10 else 'n/a') + f' | {np.std(g.qed - interp):.3f}')
print('w(age):', {k: round(v, 3) for k, v in W.items()})
