"""y0 fit of the ED spin-up (spin_* vs y0_* at the seed ages) over all finished sites of one or more roots, by zone."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

def load(root: Path) -> pd.DataFrame:
    fs = sorted((root / 'csv').glob('site*.csv'))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d[d.scenario == 'ssp126']
    return d

pd.set_option('display.width', 220)
roots = [Path(a) for a in sys.argv[1:]]
dfs = {r.name: load(r) for r in roots}
common = set.intersection(*[set(d.site) for d in dfs.values()])
print('sites per root:', {k: d.site.nunique() for k, d in dfs.items()}, 'common', len(common))
for name, d in dfs.items():
    for sub, dd in (('all', d), ('common', d[d.site.isin(common)])):
        dd = dd.copy()
        dd['zone'] = pd.cut(dd.lat_est.abs(), [-1, 23.5, 40, 55, 91], labels=['trop', 'warm', 'temp', 'boreal'])
        dd['eh'] = np.log((dd.spin_height_m + 0.5) / (dd.y0_height + 0.5))
        dd['ea'] = np.log((dd.spin_agb_kgC + 0.05) / (dd.y0_agb + 0.05))
        site_a = dd.groupby('site').ea.mean(); site_h = dd.groupby('site').eh.mean()
        g = dd.groupby('zone', observed=True)
        t = pd.DataFrame({'sites': g.site.nunique(), 'h_bias': g.eh.mean(), 'h_rms': g.eh.apply(lambda x: np.sqrt((x ** 2).mean())),
                          'a_bias': g.ea.mean(), 'a_rms': g.ea.apply(lambda x: np.sqrt((x ** 2).mean())),
                          'frac_rows_|ea|>ln2': g.ea.apply(lambda x: (x.abs() > np.log(2)).mean())})
        print(f'\n== {name} [{sub}]  sites with |mean ea|>ln2: {(site_a.abs() > np.log(2)).mean():.2f}, |mean eh|>ln2: {(site_h.abs() > np.log(2)).mean():.2f}')
        print(t.round(3).to_string())
