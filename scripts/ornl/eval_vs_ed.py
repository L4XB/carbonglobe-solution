"""ORNL row predictions against ED on common well-seeded rows, by stratum and seed age (validates the lookup-growth factor G)."""
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts/ed')
import numpy as np, pandas as pd
from pathlib import Path
from cgb import shift
import dec_eval as de
ROOT = Path('.')
tag = sys.argv[1] if len(sys.argv) > 1 else 'v1'
z = np.load(f'work/ornl2522/rows_{tag}.npz'); pred, kind = z['pred'], z['kind']
keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy(); scen = keys.scenario.to_numpy()
C = shift.read_predictions('results/spatcorr_stack_20261106_s08.csv').astype(float)
U = de.A / 1.055 / de.DIV[1]; P = pred * U
strat = de.strata_of_sites()[site]; cl = (np.load('work/collapse_cluster_labels.npy') == 0)[site]; mx = np.load('work/fire/max_factor_rule833.npy')[site]
parts = [de.load_root(ROOT / r, 'dec') for r in ('work/edv3/full_cx16', 'work/edv3/fire_cx17b')]
seen = set(); keep = []
for q in parts:
    q = q[~q.site.isin(seen)]; seen |= set(q.site.unique()); keep.append(q)
t = pd.concat(keep, ignore_index=True)
Pe, sim = de.predict(t, keys, C, 'raw', float(sys.argv[2]) if len(sys.argv) > 2 else 0.07, 0.0, 1.055)
m0 = np.isfinite(P) & sim[:, 1] & (C[:, 1] > 0.3 / de.DIV[1]) & ~cl & (mx < 1e3)
print('stratum scen  age | rows | median log(ORNL / ED) | sd(ORNL - ED) | corr of log(x / C) | medians log(ORNL / C), log(ED / C)')
for st in ('boreal', 'cool', 'warm', 'moist'):
    for sc in ('ssp126', 'ssp585'):
        for a in sorted(np.unique(age[m0])):
            m = m0 & (strat == st) & (scen == sc) & (age == a)
            if m.sum() < 8: continue
            lo = np.log(P[m] / C[m, 1]); le = np.log(Pe[m, 1] / C[m, 1])
            print(f'{st:6s} {sc[3:]} {a:4d} | {int(m.sum()):4d} | {np.median(lo - le):+.3f} | {np.std(lo - le):.3f} | {np.corrcoef(lo, le)[0, 1]:+.2f} | {np.median(lo):+.3f} {np.median(le):+.3f}')
