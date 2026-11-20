"""ED vs C: scenario contrast (585 - 126) and growth (ssp126 vs R) per group x band x column, on one ED results dir."""
import sys; sys.path.insert(0, './scripts'); sys.path.insert(0, './src')
from pathlib import Path
import numpy as np, pandas as pd
from cgb import shift
import ed_transfer as et
ROOT = Path('.')
ed_dir = Path(sys.argv[1]); alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
only = set(int(s) for s in sys.argv[3].split(',')) if len(sys.argv) > 3 and sys.argv[3] else None
keys = shift.template_keys(); site = keys['site'].to_numpy(); band = shift.band_of(keys['age'].to_numpy()); age = keys['age'].to_numpy(); scen = keys['scenario'].to_numpy()
rd = lambda p: shift.read_predictions(ROOT / p).astype(np.float64)
C = rd('results/spatcorr_stack_20261106_s08.csv'); R = rd('results/latent_ref_y0_m114114.csv')
ED, sim, rep = et.read_ed(ed_dir, keys, (alpha, alpha), form='hybrid', R=R, smooth='despike')
if only is not None:
    sim = sim & np.isin(site, list(only))
cl = (np.load(ROOT / 'work/collapse_cluster_labels.npy') == 0)[site]
df = pd.DataFrame({'site': site, 'age': age, 'scen': scen, 'band': band, 'cl': cl, 'Ch': C[:, 0], 'Ca': C[:, 1], 'Eh': ED[:, 0], 'Ea': ED[:, 1], 'Rh': R[:, 0], 'Ra': R[:, 1]})[sim]
w = df.pivot_table(index=['site', 'age', 'band', 'cl'], columns='scen', values=['Ch', 'Ca', 'Eh', 'Ea', 'Rh', 'Ra']).reset_index()
print(f'{ed_dir.name}: {w.site.nunique()} sites (alpha {alpha})')
for grp, gm in (('out', ~w.cl), ('cluster', w.cl)):
    for b in ('young', 'mid', 'old'):
        for c in ('h', 'a'):
            x = w[gm & (w.band == b)]; x = x[x[(f'R{c}', 'ssp126')] > 0.02]
            if len(x) < 10: continue
            cC = (x[(f'C{c}', 'ssp585')] - x[(f'C{c}', 'ssp126')]).mean(); cE = (x[(f'E{c}', 'ssp585')] - x[(f'E{c}', 'ssp126')]).mean()
            gC = (x[(f'C{c}', 'ssp126')] - x[(f'R{c}', 'ssp126')]).mean(); gE = (x[(f'E{c}', 'ssp126')] - x[(f'R{c}', 'ssp126')]).mean()
            e2 = np.mean([((x[(f'E{c}', s)] - x[(f'C{c}', s)]) ** 2).mean() for s in ('ssp126', 'ssp585')])
            print(f'  {grp:7s} {b:6s} {c}: contrast C {cC:+.3f} ED {cE:+.3f} | growth C {gC:+.3f} ED {gE:+.3f} | E[(ED-C)^2] {e2:.4f} | n {len(x)}')
