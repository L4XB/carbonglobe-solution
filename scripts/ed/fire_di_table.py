#!/usr/bin/env python
"""Yearly ED fire dryness index (replica of calcSiteDrynessIndex, scripts/ed/fire_proxy_calib.dryness_avg) for every
exactly matched test site: calibrated test forcing (both scenarios) and the twin's training forcing.
-> work/cal/fire_di.npz: di_126, di_585, di_train (n_sites, 40) in ED units, sites, cells
    python scripts/ed/fire_di_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "ed"))
from cgb import io  # noqa: E402
import fire_proxy_calib as fp  # noqa: E402

ex = np.load(ROOT / "work/test_cell_exact.npz"); ok = ex["cell"] >= 0
sites = np.where(ok)[0]; cells = ex["cell"][ok]; lat = ex["lat"][ok]
m = np.load(ROOT / "work/cal/cal_maps_cal.npz"); al, be = m["alpha"][:2], m["beta"][:2]
out = {k: np.zeros((len(sites), 40)) for k in ("di_126", "di_585", "di_train")}
xtr = io.open_train_x()
order = np.argsort(cells)
for lo in range(0, len(sites), 500):
    idx = order[lo:lo + 500]
    tr = np.asarray(xtr[cells[idx]][:, :, :, :2], dtype=np.float64)
    for j, i in enumerate(idx):
        out["di_train"][i] = fp.dryness_avg(tr[j, :, :, 0] - 273.15, tr[j, :, :, 1], fp.b_factors(float(lat[i])))
for sc, key in (("ssp126", "di_126"), ("ssp585", "di_585")):
    xt, _ = io.open_test(sc)
    for lo in range(0, len(sites), 500):
        blk = (np.asarray(xt[sites[lo:lo + 500]][:, :, :, :2], dtype=np.float64) - be) / al
        for j in range(blk.shape[0]):
            i = lo + j
            out[key][i] = fp.dryness_avg(blk[j, :, :, 0] - 273.15, np.clip(blk[j, :, :, 1], 0, None), fp.b_factors(float(lat[i])))
np.savez(ROOT / "work/cal/fire_di.npz", sites=sites, cells=cells, **out)
for k, v in out.items():
    print(k, "median of site-max", np.median(v.max(1)).round(0), " share of site-years > 65k:", np.mean(v > 65000).round(4))
