"""Which decode of an ED row is closest to the truth? Training-world rows of scripts/ed/train_height_eval.py (<root>/height_eval_rows.csv: truth y0 / y40,
ED first record e0 / year 40 e40, both chronosequences at age + 40).
    raw      e40
    hybrid   e40 x y0 / e0                      (same-age start-state ratio; the decode of the 0.032 file)
    chrono   y0-chrono(age + 40) x e40 / e0-chrono(age + 40)      (dec2.py; file A and later)
    persist  y0-chrono(age + 40)                (no ED at all)
rms of log(pred / truth) per zone x age band, for all rows and by the seed mismatch |log(y0 / e0)| of the column."""
import sys, json
import numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1] + '/height_eval_rows.csv')
zone = {}
for st, lst in json.load(open('work/edv3/train_hv_cells.json')).items():
    for _, c in lst: zone[int(c)] = st
d['zone'] = [zone.get(int(c), 'other') for c in d.cell]
AGES = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]); LA = np.log(AGES); x = np.log(np.minimum(AGES + 40, 500))
EPS = {'h': 0.1, 'a': 0.01}
rows = []
for cell, g in d.groupby('cell'):
    g = g.set_index('age').reindex(AGES)
    for c in ('h', 'a'):
        e = EPS[c]; y0 = g[f'y0_{c}'].to_numpy(); e0 = np.maximum(g[f'e0_{c}'].to_numpy(), 0); e40 = np.maximum(g[f'e40_{c}'].to_numpy(), 0); y40 = g[f'y40_{c}'].to_numpy()
        yc = np.exp(np.interp(x, LA, np.log(y0 + e))); ec = np.exp(np.interp(x, LA, np.log(e0 + e)))
        t = np.log(y40 + e)
        g[f'err_raw_{c}'] = np.log(e40 + e) - t
        g[f'err_hyb_{c}'] = np.log((e40 + e) * (y0 + e) / (e0 + e)) - t
        g[f'err_hybc_{c}'] = np.log((e40 + e) * np.clip((y0 + e) / (e0 + e), 0.67 if c == 'h' else 0.5, 1.5 if c == 'h' else 2.0)) - t
        g[f'err_chr_{c}'] = np.log(yc * (e40 + e) / ec) - t
        g[f'err_per_{c}'] = np.log(yc) - t
        g[f'err_mix_{c}'] = 0.5 * (g[f'err_hybc_{c}'] + g[f'err_chr_{c}'])
        g[f'mis_{c}'] = np.abs(np.log((y0 + e) / (e0 + e)))
    g['cell'] = cell; g['zone'] = zone.get(int(cell), 'other'); rows.append(g.reset_index())
D = pd.concat(rows, ignore_index=True)
rms = lambda v: float(np.sqrt(np.mean(np.square(v)))) if len(v) else float('nan')
for c, cn in (('h', 'HEIGHT'), ('a', 'AGB')):
    print(f'{cn}: rms of log(pred / truth), rows | raw | hybrid | hybrid (clipped ratio) | chrono | mean of hybrid and chrono | persistence')
    for z in ('boreal', 'cool', 'warm', 'moist'):
        for bn, lo, hi in (('age 1', 1, 1), ('10-30', 10, 30), ('50-100', 50, 100), ('150-500', 150, 500)):
            g = D[(D.zone == z) & (D.age >= lo) & (D.age <= hi)]
            for mn, mlo, mhi in (('all', 0, 9), ('mis < 0.05', 0, 0.05), ('0.05-0.12', 0.05, 0.12), ('0.12-0.25', 0.12, 0.25)):
                gg = g[(g[f'mis_{c}'] >= mlo) & (g[f'mis_{c}'] < mhi)]
                if len(gg) < 12: continue
                print(f'   {z:6s} {bn:8s} {mn:10s} n {len(gg):4d} | {rms(gg[f"err_raw_{c}"]):.3f} | {rms(gg[f"err_hyb_{c}"]):.3f} | {rms(gg[f"err_hybc_{c}"]):.3f} | {rms(gg[f"err_chr_{c}"]):.3f} | {rms(gg[f"err_mix_{c}"]):.3f} | {rms(gg[f"err_per_{c}"]):.3f}')
