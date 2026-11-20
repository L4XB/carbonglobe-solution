"""Fit the ORNL / chronosequence height predictor on ED heights: log(ED_h / chrono_h(age + 40)) = c + k q (+ m log(curve / chrono))."""
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts/ed')
import numpy as np, pandas as pd
from pathlib import Path
from cgb import io, shift
import dec_eval as de
ROOT = Path('.')
AGES = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500])
keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy(); scen = keys.scenario.to_numpy()
S = shift.read_predictions('results/spatcorr_stack_20261110_s08.csv').astype(float); C = shift.read_predictions('results/spatcorr_stack_20261106_s08.csv').astype(float)
_, y1 = io.open_test('ssp126'); _, y5 = io.open_test('ssp585'); FL = shift.FLOORS; M = shift.M_HEIGHT
Y = {'ssp126': (y1 - FL) * M, 'ssp585': (y5 - FL) * M}
mx = np.load('work/fire/max_factor_rule833.npy'); cl = np.load('work/collapse_cluster_labels.npy') == 0; strat = de.strata_of_sites()
z = np.load('work/ornl2522/rows_v3.npz'); orn = z['pred']; oq = z['q']
parts = [de.load_root(ROOT / r, 'dec') for r in ('work/edv3/full_cx16', 'work/edv3/fire_cx17b')]
seen = set(); keep = []
for q in parts:
    q = q[~q.site.isin(seen)]; seen |= set(q.site.unique()); keep.append(q)
t = pd.concat(keep, ignore_index=True); t = t[(mx[t.site.to_numpy()] < 1e3) & ~cl[t.site.to_numpy()]]
Pe, sim = de.predict(t, keys, C, 'raw', 0.07, 0.0, 1.055)
rows = np.where(sim[:, 0] & np.isfinite(oq))[0]
ed_h = Pe[rows, 0] * de.DIV[0] / de.A
ch = np.zeros(len(rows)); cv = np.zeros(len(rows))
for j, i in enumerate(rows):
    y = Y[scen[i]][site[i]]; yh, ya = y[:, 0], y[:, 1]
    ch[j] = float(np.exp(np.interp(np.log(min(age[i] + 40, 500)), np.log(AGES), np.log(np.maximum(yh, 0.3)))))
    o = np.argsort(ya); cv[j] = float(np.interp(orn[i] / 1.055, np.maximum.accumulate(ya[o]), yh[o]))
q = oq[rows]; yv = np.log(ed_h / ch); xc = np.log(np.maximum(cv, 0.3) / ch)
sth = np.log(S[rows, 0] * de.DIV[0] / de.A / ed_h)
a_ = age[rows]; st = strat[site[rows]]
print('age: n | fit log(ED / chrono) = c + k q: c, k, rms | with the curve term m: c, k, m, rms | stack rms | chrono rms')
coef = {}
for a in AGES:
    m = (a_ == a) & np.isin(st, ['moist', 'warm'])
    if m.sum() < 40: continue
    A1 = np.column_stack([np.ones(m.sum()), q[m]]); c1, *_ = np.linalg.lstsq(A1, yv[m], rcond=None)
    A2 = np.column_stack([np.ones(m.sum()), q[m], xc[m]]); c2, *_ = np.linalg.lstsq(A2, yv[m], rcond=None)
    coef[int(a)] = (float(c1[0]), float(c1[1]))
    print(f'{a:4d}: {int(m.sum()):5d} | {c1[0]:+.3f} {c1[1]:+.3f} rms {np.sqrt(np.mean((yv[m] - A1 @ c1) ** 2)):.3f} | {c2[0]:+.3f} {c2[1]:+.3f} {c2[2]:+.3f} rms {np.sqrt(np.mean((yv[m] - A2 @ c2) ** 2)):.3f} | {np.sqrt(np.mean(sth[m] ** 2)):.3f} | {np.sqrt(np.mean(yv[m] ** 2)):.3f}')
print('coef (moist + warm):', {k: (round(v[0], 3), round(v[1], 3)) for k, v in coef.items()})
for stn, ms in (('cool', st == 'cool'), ('boreal', st == 'boreal')):
    for grp, ma in (('age1', a_ == 1), ('young', (a_ > 1) & (a_ < 50)), ('mid', (a_ >= 50) & (a_ < 150)), ('old', a_ >= 150)):
        m = ms & ma
        if m.sum() < 30: continue
        A1 = np.column_stack([np.ones(m.sum()), q[m]]); c1, *_ = np.linalg.lstsq(A1, yv[m], rcond=None)
        print(f'  {stn:6s} {grp:5s}: n {int(m.sum()):4d} | c {c1[0]:+.3f} k {c1[1]:+.3f} rms {np.sqrt(np.mean((yv[m] - A1 @ c1) ** 2)):.3f} | stack rms {np.sqrt(np.mean(sth[m] ** 2)):.3f} | chrono rms {np.sqrt(np.mean(yv[m] ** 2)):.3f}')
