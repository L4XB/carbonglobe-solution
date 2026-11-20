#!/usr/bin/env python
"""Exact training cell of every test site: global search over all 54152 training cells with the per-channel affine soil
maps of scripts/geo/cell_identity.py (work/test_cell_match.npz: a, b). Writes work/test_cell_exact.npz:
cell (index into the training arrays, -1 = no exact match), resid, second, n_exact (cells within tol), lat, lon.
    python scripts/geo/cell_identity_global.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io  # noqa: E402

SOIL = np.arange(129, 136)
z = np.load(ROOT / "work/test_cell_match.npz"); a, b = z["a"], z["b"]
xt, _ = io.open_test("ssp126")
te = np.asarray(xt[:, 0, 0, :][:, SOIL], dtype=np.float64)
tr = np.asarray(io.open_train_x()[:, 0, 0, :][:, SOIL], dtype=np.float64)
tll = np.load(ROOT / "work/edv3/train_latlon.npy")
sd = tr.std(0); sd[sd == 0] = 1.0
use = [0, 1, 2, 3, 4, 5]                      # soil depth is nearly constant: leave it out of the distance
P = (a * tr + b)[:, use] / (a * sd)[use]      # predicted test values per training cell, sd units
Q = te[:, use] / (a * sd)[use]
n = len(te)
cell = np.full(n, -1); resid = np.full(n, np.nan); second = np.full(n, np.nan); n_exact = np.zeros(n, int)
for lo in range(0, n, 256):
    q = Q[lo:lo + 256]
    d = np.sqrt(((q[:, None, :] - P[None, :, :]) ** 2).mean(-1))        # (256, 54152)
    o = np.argpartition(d, 1, axis=1)[:, :2]
    for i in range(len(q)):
        j0, j1 = o[i]; 
        if d[i, j0] > d[i, j1]: j0, j1 = j1, j0
        cell[lo + i] = j0; resid[lo + i] = d[i, j0]; second[lo + i] = d[i, j1]; n_exact[lo + i] = int((d[i] < 1e-5).sum())
tol = 1e-5
ex = resid < tol
print(f"exact matches (resid < {tol}): {ex.sum()} of {n}; unique-exact {(n_exact == 1).sum()}, multiple {(n_exact > 1).sum()}; no exact {(~ex).sum()}")
print("residual quantiles of the non-exact:", np.round(np.quantile(resid[~ex], [0, 0.25, 0.5, 0.75, 1]), 5) if (~ex).any() else "-")
print("distinct cells among exact matches:", len(np.unique(cell[ex])), " (duplicates:", ex.sum() - len(np.unique(cell[ex])), ")")
z3 = np.load(ROOT / "work/test_latlon_nex3.npz")
dd = np.hypot(tll[cell, 0] - z3["lat"], (((tll[cell, 1] - z3["lon"] + 180) % 360) - 180) * np.cos(np.radians(tll[cell, 0])))
print(f"distance exact cell - nex3 position: median {np.nanmedian(dd[ex]):.3f} deg, p90 {np.nanpercentile(dd[ex], 90):.3f}, p99 {np.nanpercentile(dd[ex], 99):.2f}, > 1.5 deg: {(dd[ex] > 1.5).sum()}")
m = np.load(ROOT / "cache/mask2d.npy"); flat = np.flatnonzero(m.ravel() > 0); lab = m.ravel()[flat]
print("paper split of the matched cells (1 train / 2 val / 3 test):", {int(k): int((lab[cell[ex]] == k).sum()) for k in (1, 2, 3)})
np.savez(ROOT / "work/test_cell_exact.npz", cell=np.where(ex, cell, -1), cell_best=cell, resid=resid, second=second, n_exact=n_exact,
         lat=tll[cell, 0], lon=tll[cell, 1], exact=ex)
