#!/usr/bin/env python
"""Pseudo forced-fire root for sites where the MAIN PASS already is the forced run: its lookup matches y0 (|ratio| < 0.12) and its window effect meets
the target in both scenarios (|miss| < 0.06). Copies the main-pass site table into <out>/csv and writes the tune record that ff_rows.py expects.
    python scripts/ed/ff_mainpass_root.py work/ff/pos_sites_mac_mainpass.txt [work/edv3/ff_pos_mainpass]"""
import json, shutil, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sites = [int(x) for x in (ROOT / sys.argv[1]).read_text().split()]
out = ROOT / (sys.argv[2] if len(sys.argv) > 2 else 'work/edv3/ff_pos_mainpass'); (out / 'csv').mkdir(parents=True, exist_ok=True); (out / 'tune').mkdir(exist_ok=True)
tq = np.load(ROOT / 'work/ff/targets_K.npz')['q']; pr = np.load(ROOT / 'work/ff/prior_lookup_ratio.npy')[:, 0]; qa = np.load(ROOT / 'work/ff/main_pass_old_effect.npy')
n = 0
for s in sites:
    src = next((ROOT / r / 'csv' / f'site{s:05d}.csv' for r in ('work/edv3/full_cx16', 'work/edv3/fire_cx17b', 'work/edv3/fire_cx17b_srv') if (ROOT / r / 'csv' / f'site{s:05d}.csv').exists()), None)
    if src is None or not np.isfinite(qa[s]).all():
        continue
    shutil.copy(src, out / 'csv' / f'site{s:05d}.csv')
    scs = ('ssp126', 'ssp585')
    json.dump({'site': s, 'train': False, 'source': 'main pass', 'target': {sc: float(tq[s, j]) for j, sc in enumerate(scs)}, 'lookup': [{'L': 3e-8, 'fa': float(pr[s]), 'source': 'main pass'}],
               'windows': [{'it': 0, 'sc': sc, 'c': 1.0, 'qa': float(qa[s, j]), 'qh': float('nan')} for j, sc in enumerate(scs)], 'L': 3e-8, 'lookup_fa': float(pr[s]),
               'c': {sc: 1.0 for sc in scs}, 'miss': {sc: float(abs(qa[s, j] - tq[s, j])) for j, sc in enumerate(scs)}}, open(out / 'tune' / f'site{s:05d}.json', 'w'), indent=1)
    n += 1
print(f'{n} of {len(sites)} sites written to {out}')
