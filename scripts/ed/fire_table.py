"""Compare c400f (fire on) vs c400 (fire off) spin-up fit to y0 on the fire-test sites."""
import re, glob
from pathlib import Path
import numpy as np
import pandas as pd

root = Path('./work/edv3')
pat = re.compile(r'age\s+(\d+)\s+height\s+([\d.]+) / y0\s+([\d.]+) \| agb\s+([\d.]+) / y0\s+([\d.]+)')
rows = []
for f in sorted(root.glob('diag_fire_*.log')):
    site = int(f.stem.split('_')[-1])
    fire = {}
    for line in f.read_text().splitlines():
        m = pat.search(line)
        if m:
            a = int(m.group(1)); fire[a] = tuple(float(m.group(i)) for i in range(2, 6))
    src = None
    for cand in (root / 'csv' / f'site{site:05d}.csv', root / 'cal_c400' / 'csv' / f'site{site:05d}.csv'):
        if cand.exists():
            src = cand; break
    if src is None:
        print('no c400 ref for', site); continue
    d = pd.read_csv(src)
    d = d[d.scenario == 'ssp126'].set_index('age')
    lat = float(d.lat_est.iloc[0])
    for a, (hf, hy, af, ay) in fire.items():
        if a not in d.index: continue
        r = d.loc[a]
        rows.append(dict(site=site, lat=lat, age=a, h_y0=hy, h_f=hf, h_n=float(r.spin_height_m), a_y0=ay, a_f=af, a_n=float(r.spin_agb_kgC), src=src.parent.parent.name))
t = pd.DataFrame(rows)
t['zone'] = pd.cut(t.lat.abs(), [-1, 23.5, 40, 55, 91], labels=['trop', 'warm', 'temp', 'boreal'])
eps_h, eps_a = 0.5, 0.05
for col in ('f', 'n'):
    t[f'eh_{col}'] = np.log((t[f'h_{col}'] + eps_h) / (t.h_y0 + eps_h))
    t[f'ea_{col}'] = np.log((t[f'a_{col}'] + eps_a) / (t.a_y0 + eps_a))
print(t.groupby('src').site.nunique())
print(t.site.nunique(), 'sites')
g = t.groupby('zone', observed=True)
out = pd.DataFrame({'n_sites': g.site.nunique(),
                    'h_bias_nofire': g.eh_n.mean(), 'h_bias_fire': g.eh_f.mean(),
                    'h_rms_nofire': g.eh_n.apply(lambda x: np.sqrt((x**2).mean())), 'h_rms_fire': g.eh_f.apply(lambda x: np.sqrt((x**2).mean())),
                    'a_bias_nofire': g.ea_n.mean(), 'a_bias_fire': g.ea_f.mean(),
                    'a_rms_nofire': g.ea_n.apply(lambda x: np.sqrt((x**2).mean())), 'a_rms_fire': g.ea_f.apply(lambda x: np.sqrt((x**2).mean()))})
pd.set_option('display.width', 200)
print(out.round(3))
print('\nby age (all zones):')
g2 = t.groupby('age')
print(pd.DataFrame({'h_rms_n': g2.eh_n.apply(lambda x: np.sqrt((x**2).mean())), 'h_rms_f': g2.eh_f.apply(lambda x: np.sqrt((x**2).mean())),
                    'a_rms_n': g2.ea_n.apply(lambda x: np.sqrt((x**2).mean())), 'a_rms_f': g2.ea_f.apply(lambda x: np.sqrt((x**2).mean())),
                    'a_bias_n': g2.ea_n.mean(), 'a_bias_f': g2.ea_f.mean()}).round(3))
print('\nper site (mean over ages):')
ps = t.groupby(['site', 'zone'], observed=True)[['eh_n', 'eh_f', 'ea_n', 'ea_f']].mean().round(2)
print(ps.to_string())
