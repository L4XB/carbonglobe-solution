"""Evaluate floor-coded ORNL row predictions (rows_<tag>.npz) against the stacks and floor-coded ED, by class, kind and stratum;
writes screen directions work/dir_ornl<tag>_<part>.npy (file - C)."""
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts/ed')
import numpy as np, pandas as pd
from pathlib import Path
from cgb import shift
import dec_eval as de
ROOT = Path('.')
tag = sys.argv[1] if len(sys.argv) > 1 else 'v2'
FL = 0.401412 / 3.39567
z = np.load(f'work/ornl2522/rows_{tag}.npz'); pred, kind = z['pred'], z['kind']
keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy(); scen = keys.scenario.to_numpy()
C = shift.read_predictions('results/spatcorr_stack_20261106_s08.csv').astype(float); S = shift.read_predictions('results/spatcorr_stack_20261110_s08.csv').astype(float)
P = pred * de.A / 1.055 / de.DIV[1] - FL
strat = de.strata_of_sites()[site]; cl = (np.load('work/collapse_cluster_labels.npy') == 0)[site]; mx = np.load('work/fire/max_factor_rule833.npy')[site]
cls = np.where(cl, 'cluster', np.where(mx < 30, 'never', np.where(mx < 1e3, 'class b', 'possible')))
n_all = C.size; have = np.isfinite(P)
parts = [de.load_root(ROOT / r, 'dec') for r in ('work/edv3/full_cx16', 'work/edv3/fire_cx17b')]
seen = set(); keep = []
for q in parts:
    q = q[~q.site.isin(seen)]; seen |= set(q.site.unique()); keep.append(q)
t = pd.concat(keep, ignore_index=True)
Pe, sim = de.predict(t, keys, C, 'raw', 0.07, 0.0, 1.055)
E = np.where(sim[:, 1], Pe[:, 1] - FL, np.nan)
print(f'rows with an ORNL prediction {int(have.sum())}; E_all of (ORNL - C) {np.nansum((P - C[:, 1]) ** 2) / n_all:.5f}, of (ORNL - stack) {np.nansum((P - S[:, 1]) ** 2) / n_all:.5f}')
print('class     kind   | rows | median ORNL - C, ORNL - stack (solution units) | energy vs stack | common rows with ED: n, corr of (x - C), sd(ORNL - ED), median(ORNL - ED), sd(ED - C), sd(ORNL - C)')
kn = {1: 'old', 2: 'age1', 3: 'mid'}
for c in ('never', 'class b', 'possible', 'cluster'):
    for k in (2, 3, 1):
        m = have & (kind == k) & (cls == c)
        if m.sum() < 20: continue
        mc = m & np.isfinite(E)
        s = f'{c:9s} {kn[k]:5s} | {int(m.sum()):6d} | {np.median(P[m] - C[m, 1]):+.3f} {np.median(P[m] - S[m, 1]):+.3f} | {np.sum((P[m] - S[m, 1]) ** 2) / n_all:.5f} | '
        if mc.sum() > 30:
            a = P[mc] - C[mc, 1]; b = E[mc] - C[mc, 1]
            s += f'{int(mc.sum()):5d}, {np.corrcoef(a, b)[0, 1]:+.2f}, {np.std(a - b):.3f}, {np.median(a - b):+.3f}, {np.std(b):.3f}, {np.std(a):.3f}'
        print(s)
for st in ('boreal', 'cool', 'warm', 'moist'):
    for k in (2, 3, 1):
        m = have & (kind == k) & (cls == 'never') & (strat == st)
        if m.sum() < 20: continue
        mc = m & np.isfinite(E)
        s = f'   never {st:6s} {kn[k]:5s} | {int(m.sum()):6d} | {np.median(P[m] - C[m, 1]):+.3f} {np.median(P[m] - S[m, 1]):+.3f} | {np.sum((P[m] - S[m, 1]) ** 2) / n_all:.5f} | '
        if mc.sum() > 30:
            a = P[mc] - C[mc, 1]; b = E[mc] - C[mc, 1]
            s += f'{int(mc.sum()):5d}, {np.corrcoef(a, b)[0, 1]:+.2f}, {np.std(a - b):.3f}, {np.median(a - b):+.3f}, {np.std(b):.3f}, {np.std(a):.3f}'
        print(s)
def save(name, m):
    d = np.zeros((len(P), 2), np.float32); d[m, 1] = (P - C[:, 1])[m]; np.save(f'work/dir_ornl{tag}_{name}.npy', d)
save('mid_never', have & (kind == 3) & (cls == 'never')); save('mid_possible', have & (kind == 3) & np.isin(cls, ['class b', 'possible'])); save('mid_cluster', have & (kind == 3) & (cls == 'cluster')); save('all', have)
