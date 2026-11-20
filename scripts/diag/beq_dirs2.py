"""Burned-state directions split by seed-age group and by the strength of the ORNL loss, for proxy (x) and measured (m) fire-possible sites.
Level 1.2 kgC. work/dirx_bq_<src>_<qbin>_<agegroup>.npy; qbin: a = q_old in [-0.3, -0.15), b = [-0.5, -0.3), c = < -0.5."""
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts/ed')
import numpy as np, pandas as pd
from cgb import shift
FL_A = 0.401412 / 3.39567
keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy(); scen = keys.scenario.to_numpy()
S = shift.read_predictions('results/spatcorr_stack_20261110_s08.csv').astype(float)
cl = (np.load('work/collapse_cluster_labels.npy') == 0)[site]; mxf = np.load('work/fire/max_factor_rule833.npy')[site]; pos = ~cl & (mxf >= 1e3)
z = np.load('work/ornl2522/rows_v4.npz'); q = z['q']; qs = z['qsrc']
t = pd.DataFrame(dict(site=site, scen=scen, q=np.where(age >= 200, q, np.nan), qs=np.where(age >= 200, qs, 0)))
g = t.groupby(['site', 'scen']).agg(q=('q', 'median'), qs=('qs', 'max')).reset_index()
qo = dict(zip(zip(g.site, g.scen), g.q)); so = dict(zip(zip(g.site, g.scen), g.qs))
qold = np.array([qo.get((s_, c_), np.nan) for s_, c_ in zip(site, scen)]); sold = np.array([so.get((s_, c_), 0) for s_, c_ in zip(site, scen)])
np.save('work/ornl2522/qold_row_v4.npy', qold); np.save('work/ornl2522/qsrc_site_v4.npy', sold)
lv = 1.2 / 1.204 / 3.39567 - FL_A
for src, ms in (('m', sold == 1), ('x', np.isin(sold, (4, 5, 6, 7)))):
    for qb, lo, hi in (('a', -0.3, -0.15), ('b', -0.5, -0.3), ('c', -9, -0.5)):
        for an, ma in (('age1', age == 1), ('young', (age > 1) & (age < 50)), ('mid', (age >= 50) & (age < 150)), ('old', age >= 150)):
            m = pos & ms & np.isfinite(qold) & (qold >= lo) & (qold < hi) & ma & (S[:, 1] > lv)
            d = np.zeros_like(S); d[m, 1] = lv - S[m, 1]
            np.save(f'work/dirx_bq_{src}_{qb}_{an}.npy', d.astype(np.float32))
            print(f'{src} {qb} {an:5s}: rows {int(m.sum()):5d}, sites x scen {len(set(zip(site[m], scen[m]))):4d}, mean stack (floor-free) {(S[m, 1].mean() + FL_A) if m.any() else 0:.2f}, energy {np.sum(d ** 2):7.1f}')
