"""Training-world rows of scripts/ed/train_height_eval.py (<root>/height_eval_rows.csv): is ED's response a SITE property?
Per zone and age band: slope / corr of the truth's response on ED's (a) row by row, (b) between cells (cell medians over the band), (c) within cells
(row minus cell median); and the rms error of four predictors of the truth's response: none (chronosequence), the row's ED response, the cell-median ED
response, the zone-median ED response."""
import sys
import numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1] + '/height_eval_rows.csv')
import json
zone = {}
for st, lst in json.load(open('work/edv3/train_hv_cells.json')).items():
    for _, c in lst: zone[int(c)] = st
d['zone'] = [zone.get(int(c), 'other') for c in d.cell]
gh, gh2 = 0.12, 0.15
for c, cn in (('h', 'HEIGHT'), ('a', 'AGB')):
    print(cn)
    for z in ('boreal', 'cool', 'warm', 'moist'):
        for bn, lo, hi in (('10-30', 10, 30), ('50-100', 50, 100), ('150-500', 150, 500)):
            g = d[(d.zone == z) & (d.age >= lo) & (d.age <= hi)]
            g = g[(g.mis_h < gh) & (g.mis2_h < gh2)] if c == 'h' else g[g.mis_a < np.where(g.age >= 200, 0.25, 0.12)]
            if len(g) < 20 or g.cell.nunique() < 6: continue
            rt, re = g[f'rt_{c}'], g[f're_{c}']
            mt, me = rt.groupby(g.cell).transform('median'), re.groupby(g.cell).transform('median')
            cm = g.groupby('cell')[[f'rt_{c}', f're_{c}']].median()
            sl = lambda x, y: (np.polyfit(x, y, 1)[0], np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else (np.nan, np.nan)
            s_row, s_bet, s_wit = sl(re, rt), sl(cm[f're_{c}'], cm[f'rt_{c}']), sl(re - me, rt - mt)
            zmed = re.median()
            rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
            print(f'   {z:6s} {bn:8s} rows {len(g):4d} cells {g.cell.nunique():3d} | slope (corr) row {s_row[0]:+.2f} ({s_row[1]:+.2f}), between cells {s_bet[0]:+.2f} ({s_bet[1]:+.2f}), within cells {s_wit[0]:+.2f} ({s_wit[1]:+.2f}) | '
                  f'sd of ED response between {cm[f"re_{c}"].std():.3f} within {(re - me).std():.3f}; truth between {cm[f"rt_{c}"].std():.3f} within {(rt - mt).std():.3f} | '
                  f'rms error: chrono {rms(rt):.3f}, row ED {rms(rt - re):.3f}, cell-median ED {rms(rt - me):.3f}, zone-median ED {rms(rt - zmed):.3f}, 0.5 x cell-median dev + zone median {rms(rt - zmed - 0.5 * (me - zmed)):.3f}')
