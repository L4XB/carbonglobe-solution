#!/usr/bin/env python
"""Is the test lookup (test_y0) the training lookup of the same cell? For the exactly matched test sites compare the
released test_y0 (site, 15 ages, 7 vars) with the training trajectories of the same cell and seed age: year 0 (each
month and the annual mean) and, for reference, year 40. Per variable: best affine fit test = a * train + b, R2, and
the share of rows within 1e-3 relative.
    python scripts/geo/y0_identity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io  # noqa: E402

VARS = ["height", "agb", "soil", "lai", "gpp", "npp", "rh"]
AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
z = np.load(ROOT / "work/test_cell_exact.npz"); cell = z["cell"]; ok = cell >= 0
sites = np.where(ok)[0]; cells = cell[ok]
order = np.argsort(cells)
_, y0a = io.open_test("ssp126"); _, y0b = io.open_test("ssp585")
y0a = np.asarray(y0a, dtype=np.float64); y0b = np.asarray(y0b, dtype=np.float64)
print("test_y0 identical between scenarios? max |diff|", np.abs(y0a - y0b).max())
cache = ROOT / "work/train_twin_states.npz"
if cache.exists():
    c = np.load(cache); tr0 = c["tr0"]; tr40 = c["tr40"]; tr1 = c["tr1"]
else:
    tr0 = np.zeros((len(sites), 15, 12, 7), np.float32); tr40 = np.zeros_like(tr0); tr1 = np.zeros_like(tr0)
    for k, a in enumerate(AGES):
        y = io.open_train_y(a)
        blk = np.asarray(y[cells[order]][:, [0, 1, 40]])               # (n, 3, 12, 7)
        inv = np.empty_like(order); inv[order] = np.arange(len(order))
        blk = blk[inv]
        tr0[:, k] = blk[:, 0]; tr1[:, k] = blk[:, 1]; tr40[:, k] = blk[:, 2]
        print("loaded age", a, flush=True)
    np.savez(cache, tr0=tr0, tr1=tr1, tr40=tr40, sites=sites, cells=cells)
te = y0a[sites]                                                         # (n, 15, 7)
for v, name in enumerate(VARS):
    y = te[:, :, v].ravel()
    best = None
    for label, arr in [(f"y0 month {m + 1}", tr0[:, :, m, v]) for m in range(12)] + [("y0 annual mean", tr0[:, :, :, v].mean(2)), ("y1 annual mean", tr1[:, :, :, v].mean(2)), ("y40 annual mean", tr40[:, :, :, v].mean(2)), ("y40 month 12", tr40[:, :, 11, v])]:
        x = arr.ravel().astype(np.float64)
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ coef; r2 = 1 - r.var() / y.var()
        if best is None or r2 > best[1]:
            best = (label, r2, coef, np.abs(r))
        if label in ("y0 month 1", "y0 month 12", "y0 annual mean", "y40 annual mean"):
            print(f"  {name:6s} vs {label:16s}: a {coef[0]:.5f} b {coef[1]:+.5f} R2 {r2:.6f}")
    label, r2, coef, ar = best
    print(f"{name:6s} BEST {label}: test = {coef[0]:.6f} * train + {coef[1]:+.6f}, R2 {r2:.7f}, |resid| median {np.median(ar):.2e} p90 {np.percentile(ar, 90):.2e} p99 {np.percentile(ar, 99):.2e} (y sd {y.std():.3f})")
