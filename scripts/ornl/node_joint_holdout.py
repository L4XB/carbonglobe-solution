#!/usr/bin/env python
"""Hold-out test of the joint node fit (node_q_joint.py): at burnable sites WITH mature pixels near the node, the MATURE pixels inside a hole of
+-H deg around the node are removed (the non-mature ones stay), the joint fit is redone and compared with the central mature median (truth).
Emulates sites whose mature forest sits away from the node. Also printed: the mature cell median without the hole pixels (what v4 would use),
the all-pixel central median (node_q 'all' central) and the mature-only plain fit without the hole.
    python scripts/ornl/node_joint_holdout.py 0.1 [max_sites]"""
import sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import node_q as nq, node_q_joint as nj
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]; D = ROOT / 'work/ornl2522'
H = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1; nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 500
sites = np.load(D / 'node_win_sites.npy'); zm = np.load(D / 'q_node_mat.npz'); zn = np.load(D / 'q_node.npz')
mx = np.load(ROOT / 'work/fire/max_factor_rule833.npy'); cl = np.load(ROOT / 'work/collapse_cluster_labels.npy') == 0
cand = np.flatnonzero(((mx[sites] >= 30) | cl[sites]) & (zm['mat_n_cen'][:, 1] >= 12))
rng = np.random.default_rng(2); cand = np.sort(rng.choice(cand, size=min(nmax, len(cand)), replace=False))
L = {k: np.load(D / f'node_win_{k}.npy', mmap_mode='r') for k in ('r0', 'r1', 's126', 's585')}
f = lambda a: np.where(a == -32768, np.nan, a.astype(np.float32) * np.float32(0.01))
hole = nq.DCEN <= H / 0.5; cellm = nq.DCEN <= 0.5
rows = []
for i in cand:
    r0 = f(L['r0'][i]); r1 = f(L['r1'][i]); okx = np.isfinite(r0) & np.isfinite(r1) & (r0 > 0.3) & (r1 > 0)
    g = np.where(okx, np.log(np.maximum(r1, 1e-6) / np.maximum(r0, 1e-6)), np.nan); mat = okx & (r0 > 0.5) & (g < 0.2) & (g > -0.3)
    for j, sc in enumerate(('s126', 's585')):
        s1 = f(L[sc][i]); m = okx & np.isfinite(s1) & (s1 > 0)
        truth = zm['mat_central'][i, j]
        keep = m & ~(mat & hole)                       # mature pixels inside the hole removed
        matk = mat & keep
        if matk.sum() < 12 or keep.sum() < 30: continue
        lq = np.where(m, np.log(np.maximum(s1, 1e-6) / np.maximum(r1, 1e-6)), np.nan)
        w = nq.W9[:, keep].T; lam, se, beta, res = nj.fit(w, ~mat[keep], lq[keep])
        mc = matk & cellm; cellmed = float(np.median(lq[mc])) if mc.sum() >= 6 else np.nan
        wm = nq.W9[:, matk].T; rho, se_p, _ = nq.fit_plain(wm, np.exp(lq[matk])); neff_m = float(np.sum(wm[:, 4] ** 2))
        allc = m & ~mat & (nq.DCEN <= 0.1); allcen = float(np.median(lq[allc])) if allc.sum() >= 8 else np.nan
        rows.append((truth, lam, se, beta, cellmed, np.log(max(rho[4], 1e-3)), neff_m, allcen, float(np.sum(w[:, 4] ** 2)), int(matk.sum()), int(mx[sites[i]] >= 1e3)))
r = np.array(rows, float); t = r[:, 0]
def st(x, nm, k=None):
    k = (np.isfinite(x) & np.isfinite(t)) if k is None else (k & np.isfinite(x) & np.isfinite(t)); d = x[k] - t[k]
    return f'   {nm:46s} n {int(k.sum()):4d} rms {np.sqrt(np.mean(d ** 2)):.4f} robust {1.4826 * np.median(np.abs(d - np.median(d))):.4f} mean {d.mean():+.4f} corr {np.corrcoef(x[k], t[k])[0, 1]:.3f}'
print(f'hole +-{H} deg for the mature pixels: {len(cand)} sites, {len(r)} site x scenario; sd of the truth {np.nanstd(t):.3f}')
good = (r[:, 2] < 0.03)
print(st(r[:, 1], 'joint fit (all pixels + state offset)')); print(st(r[:, 1], 'joint fit, se < 0.03', good))
print(st(r[:, 4], 'mature cell median without the hole (v4)')); print(st(r[:, 4], '   same rows as joint se < 0.03', good & np.isfinite(r[:, 4])))
print(st(r[:, 5], 'mature-only plain fit without the hole')); print(st(r[:, 5], '   with n_eff >= 20', r[:, 6] >= 20))
print(st(r[:, 7], 'non-mature central median (state-contaminated)')); print(st(r[:, 7] - r[:, 3], 'non-mature central median - beta'))
both = good & np.isfinite(r[:, 4])
print(st(0.5 * (r[:, 1] + r[:, 4]), 'mean of joint and mature cell median', both))
for nm, k in (('fire-possible', r[:, 10] == 1), ('class b / cluster', r[:, 10] == 0)):
    print(f' {nm}:'); print(st(r[:, 1], 'joint fit, se < 0.03', good & k)); print(st(r[:, 4], 'mature cell median without the hole', good & k & np.isfinite(r[:, 4])))
