"""Evaluate ORNL row predictions (work/ornl2522/rows_<tag>.npz) against the stack and ED; energies of the implied direction."""
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts/ed')
import numpy as np, pandas as pd
from pathlib import Path
from cgb import shift
import dec_eval as de
ROOT = Path('.')
tag = sys.argv[1] if len(sys.argv) > 1 else 'v1'
z = np.load(f'work/ornl2522/rows_{tag}.npz'); pred, kind = z['pred'], z['kind']
keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy(); scen = keys.scenario.to_numpy()
C = shift.read_predictions('results/spatcorr_stack_20261106_s08.csv').astype(float); S = shift.read_predictions('results/spatcorr_stack_20261110_s08.csv').astype(float)
U = de.A / 1.055 / de.DIV[1]                       # physical agb (x 1.055 decode) -> solution units
P = pred * U
strat = de.strata_of_sites()[site]; cl = (np.load('work/collapse_cluster_labels.npy') == 0)[site]; mx = np.load('work/fire/max_factor_rule833.npy')[site]
cls = np.where(cl, 'cluster', np.where(mx < 30, 'never', np.where(mx < 1e3, 'class b', 'possible')))
n_all = C.size
have = np.isfinite(P)
Dc = np.where(have, P - C[:, 1], 0.0); Ds = np.where(have, P - S[:, 1], 0.0)
print(f'rows with an ORNL prediction: {int(have.sum())} ({int((kind == 1).sum())} old, {int((kind == 2).sum())} age 1); E_all of (ORNL - C) {np.sum(Dc ** 2) / n_all:.5f}, of (ORNL - stack 11-10) {np.sum(Ds ** 2) / n_all:.5f}')
print('class      kind   rows | median log(ORNL / C)  scatter | vs stack 11-10 | energy vs C | energy vs stack')
for c in ('never', 'class b', 'possible', 'cluster'):
    for k, kn in ((1, 'old'), (2, 'age1')):
        m = have & (kind == k) & (cls == c) & (C[:, 1] > 0.02)
        if m.sum() < 20: continue
        lr = np.log(P[m] / C[m, 1]); ls = np.log(P[m] / np.maximum(S[m, 1], 1e-4))
        print(f'{c:9s} {kn:5s} {int(m.sum()):6d} | {np.median(lr):+.3f}  {0.5 * (np.percentile(lr, 84) - np.percentile(lr, 16)):.3f} | {np.median(ls):+.3f}  {0.5 * (np.percentile(ls, 84) - np.percentile(ls, 16)):.3f} | {np.sum(Dc[m] ** 2) / n_all:.5f} | {np.sum(Ds[m] ** 2) / n_all:.5f}')
for st in ('boreal', 'cool', 'warm', 'moist', 'dry'):
    for k, kn in ((1, 'old'), (2, 'age1')):
        m = have & (kind == k) & (strat == st) & (cls == 'never') & (C[:, 1] > 0.02)
        if m.sum() < 20: continue
        lr = np.log(P[m] / C[m, 1])
        print(f'   fire-never {st:6s} {kn:5s} {int(m.sum()):6d} | {np.median(lr):+.3f}  {0.5 * (np.percentile(lr, 84) - np.percentile(lr, 16)):.3f} | energy vs C {np.sum(Dc[m] ** 2) / n_all:.5f}')
# against ED on common rows (well seeded)
idx = {(int(s), str(sc), int(a)): i for i, (s, sc, a) in enumerate(zip(keys.site, keys.scenario, keys.age))}
parts = [de.load_root(ROOT / r, 'dec') for r in ('work/edv3/full_cx16', 'work/edv3/fire_cx17b')]
seen = set(); keep = []
for q in parts:
    q = q[~q.site.isin(seen)]; seen |= set(q.site.unique()); keep.append(q)
t = pd.concat(keep, ignore_index=True)
Pe, sim = de.predict(t, keys, C, 'raw', 0.07, 0.0, 1.055)
m0 = have & sim[:, 1] & (C[:, 1] > 0.3 / de.DIV[1])
print('common rows with a well-seeded ED prediction:', int(m0.sum()))
for c in ('never', 'class b', 'possible', 'cluster'):
    for k, kn in ((1, 'old'), (2, 'age1')):
        m = m0 & (kind == k) & (cls == c)
        if m.sum() < 30: continue
        lo = np.log(P[m] / C[m, 1]); le = np.log(Pe[m, 1] / C[m, 1])
        print(f'{c:9s} {kn:5s} rows {int(m.sum()):5d}: log(ORNL / C) vs log(ED / C): corr {np.corrcoef(lo, le)[0, 1]:+.2f}; medians {np.median(lo):+.3f} / {np.median(le):+.3f}; sd {lo.std():.3f} / {le.std():.3f}; sd(ORNL - ED) {np.std(lo - le):.3f}; slope ORNL on ED {np.polyfit(le, lo, 1)[0]:+.2f}')
np.save(f'work/dir_ornl_{tag}_c.npy', np.column_stack([np.zeros(len(Dc)), Dc]).astype(np.float32))
