#!/usr/bin/env python
"""How well do the tent fits of node_q.py EXTRAPOLATE to the node when no pixels lie near it? Hold-out test at sites that do have central pixels.

For burnable sites with >= 20 existing-forest pixels within +-0.05 deg of the node: the pixels within a hole of +-H deg around the node are removed,
the plain and the biomass-weighted tent fit are redone, and their centre values are compared with the central median (the truth of this test).
Also: the cell median without the hole pixels, and the ring-1 median (pixels of the own cell only).
    python scripts/ornl/node_hole_test.py 0.1 [max_sites]
"""
import sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import node_q as nq
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]; D = ROOT / 'work/ornl2522'
H = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1; nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 400
sites = np.load(D / 'node_win_sites.npy'); z = np.load(D / 'q_node.npz')
mx = np.load(ROOT / 'work/fire/max_factor_rule833.npy'); cl = np.load(ROOT / 'work/collapse_cluster_labels.npy') == 0
cand = np.flatnonzero(((mx[sites] >= 30) | cl[sites]) & (z['all_n_cen'][:, 1] >= 20))
rng = np.random.default_rng(1); cand = np.sort(rng.choice(cand, size=min(nmax, len(cand)), replace=False))
L = {k: np.load(D / f'node_win_{k}.npy', mmap_mode='r') for k in ('r0', 'r1', 's126', 's585')}
f = lambda a: np.where(a == -32768, np.nan, a.astype(np.float32) * np.float32(0.01))
hole = nq.DCEN <= H / 0.5
rows = []
for i in cand:
    r0 = f(L['r0'][i]); r1 = f(L['r1'][i]); okx = np.isfinite(r0) & np.isfinite(r1) & (r0 > 0.3) & (r1 > 0)
    lq = [np.where(okx & (f(L[sc][i]) > 0), np.log(np.maximum(f(L[sc][i]), 1e-6) / np.maximum(r1, 1e-6)), np.nan) for sc in ('s126', 's585')]
    e = nq.estimate(okx & ~hole, lq, do_rat=True)
    for j in range(2):
        rows.append((i, j, z['all_central'][i, j], e['cell'][j], e['plain'][j], e['rat'][j], e['rat_se'][j], e['plain_se'][j], e['neff'][j], z['all_cell'][i, j], int(mx[sites[i]] >= 1e3)))
r = np.array(rows, float); ok = np.isfinite(r[:, 2]) & np.isfinite(r[:, 4])
print(f'hole +-{H} deg: {len(cand)} sites, {int(ok.sum())} site x scenario with a fit; sd of the central truth {np.nanstd(r[ok, 2]):.3f}')
def st(x, t, nm):
    k = np.isfinite(x) & np.isfinite(t); d = x[k] - t[k]
    return f'   {nm:34s} n {int(k.sum()):4d} rms {np.sqrt(np.mean(d ** 2)):.4f} robust {1.4826 * np.median(np.abs(d - np.median(d))):.4f} mean {d.mean():+.4f} corr {np.corrcoef(x[k], t[k])[0, 1]:.3f} slope truth on est {np.polyfit(x[k], t[k], 1)[0]:.2f}'
t = r[:, 2]
print(st(r[:, 9], t, 'cell median (with the hole pixels)')); print(st(r[:, 3], t, 'cell median without the hole'))
print(st(r[:, 4], t, 'plain tent fit without the hole')); print(st(r[:, 5], t, 'rational fit without the hole'))
g = np.isfinite(r[:, 6]) & (r[:, 6] < 0.03)
print(st(np.where(g, r[:, 5], np.nan), t, 'rational fit, se < 0.03')); print(st(np.where(g, r[:, 4], np.nan), t, 'plain fit, same rows'))
print(st(np.where(g, 0.5 * (r[:, 4] + r[:, 5]), np.nan), t, 'mean of plain and rational'))
for lo, hi in ((0, 5), (5, 20), (20, 80), (80, 1e9)):
    k = (r[:, 8] >= lo) & (r[:, 8] < hi)
    if k.sum() > 20: print(f'   neff {lo}-{hi}:' + st(np.where(k, r[:, 4], np.nan), t, ' plain') + ' |' + st(np.where(k, r[:, 5], np.nan), t, ' rational'))
