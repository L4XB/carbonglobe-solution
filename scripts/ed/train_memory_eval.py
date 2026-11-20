"""Memory of the start state after 40 window years, training world (rows of train_height_eval.py):
    log(truth40 / ED40) = m x log(truth0 / ED0)       per zone x seed age
If the model and the window forcing are the same, the truth's year-40 state is ED40 x (y0 / e0)^m whatever the reason for the start mismatch;
m = 1 is the hybrid decode, m = 0 raw ED. Prints the least-squares m per zone x age (through the origin and with an intercept) and the rms error of
raw / hybrid / chrono / memory decode (m per zone x band, fitted on the other half of the cells = 2-fold)."""
import sys, json
import numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1] + '/height_eval_rows.csv')
zone = {}
for st, lst in json.load(open('work/edv3/train_hv_cells.json')).items():
    for _, c in lst: zone[int(c)] = st
d['zone'] = [zone.get(int(c), 'other') for c in d.cell]
EPS = {'h': 0.1, 'a': 0.01}
for c in ('h', 'a'):
    e = EPS[c]
    d[f's_{c}'] = np.log((d[f'y0_{c}'] + e) / (np.maximum(d[f'e0_{c}'], 0) + e)); d[f'f_{c}'] = np.log((d[f'y40_{c}'] + e) / (np.maximum(d[f'e40_{c}'], 0) + e))
    d[f'chr_{c}'] = d[f'rt_{c}'] - d[f're_{c}']        # error of the chrono decode (truth response - ED response)
rms = lambda v: float(np.sqrt(np.mean(np.square(v)))) if len(v) else float('nan')
cells = np.array(sorted(d.cell.unique())); fold = {c: i % 2 for i, c in enumerate(cells)}; d['fold'] = d.cell.map(fold)
for c, cn in (('h', 'HEIGHT'), ('a', 'AGB')):
    print(f'{cn}: memory m per zone x seed age (slope through the origin | with intercept: slope, intercept); gate |start mismatch| < 0.5')
    for z in ('boreal', 'cool', 'warm', 'moist'):
        line = f'   {z:6s}'
        for a in (1, 10, 30, 50, 100, 150, 200, 300, 500):
            g = d[(d.zone == z) & (d.age == a) & (d[f's_{c}'].abs() < 0.5)]
            if len(g) < 8: continue
            m0 = float(np.sum(g[f's_{c}'] * g[f'f_{c}']) / np.sum(g[f's_{c}'] ** 2))
            line += f' | {a}: {m0:+.2f}'
        print(line)
    print('   rms error of the decodes on rows with |start mismatch| in 0.03..0.25 (2-fold memory decode): n | raw | hybrid | chrono | memory (m)')
    for z in ('boreal', 'cool', 'warm', 'moist'):
        for bn, lo, hi in (('10-30', 10, 30), ('50-100', 50, 100), ('150-500', 150, 500)):
            g = d[(d.zone == z) & (d.age >= lo) & (d.age <= hi) & (d[f's_{c}'].abs() < 0.25) & (d[f's_{c}'].abs() >= 0.03)]
            if len(g) < 16: continue
            err = np.zeros(len(g)); ms = []
            for k in (0, 1):
                tr = g[g.fold != k]; te = g.fold == k
                m = float(np.sum(tr[f's_{c}'] * tr[f'f_{c}']) / max(np.sum(tr[f's_{c}'] ** 2), 1e-12)); ms.append(m)
                err[te.to_numpy()] = (g[f'f_{c}'] - m * g[f's_{c}'])[te]
            print(f'      {z:6s} {bn:8s} n {len(g):4d} | {rms(g[f"f_{c}"]):.3f} | {rms(g[f"f_{c}"] - g[f"s_{c}"]):.3f} | {rms(g[f"chr_{c}"]):.3f} | {rms(err):.3f} (m {ms[0]:+.2f} / {ms[1]:+.2f})')
