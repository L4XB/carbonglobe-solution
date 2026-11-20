#!/usr/bin/env python
"""NEX-GDDP-CMIP6 daily NetCDF -> monthly means stacked over years: work/nexgddp/<var>_<scen>_<y0>_<y1>_monthly.npy
(years, 12, 600, 1440) float32; pr in mm/month, tas in K. Uses h5py (scratchpad venv).
    <venv>/bin/python scripts/nex/nex_monthly.py VAR SCEN Y0 Y1 [suffix]"""
import sys
from pathlib import Path
import h5py
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
var, scen, y0, y1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); suf = sys.argv[5] if len(sys.argv) > 5 else ""
out = np.zeros((y1 - y0 + 1, 12, 600, 1440), np.float32)
for k, y in enumerate(range(y0, y1 + 1)):
    with h5py.File(ROOT / f"work/nexgddp/{var}_day_GFDL-ESM4_{scen}_r1i1p1f1_gr1_{y}{suf}.nc", "r") as f:
        x = f[var][:]
    x = np.where(x > 1e19, np.nan, x)
    leap = x.shape[0] == 366
    days = np.array([31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]); e = np.concatenate([[0], np.cumsum(days)])
    m = np.stack([np.nanmean(x[e[i]:e[i + 1]], axis=0) for i in range(12)])
    if var == "pr":
        m = m * 86400 * days[:, None, None]
    out[k] = m
    print(y, "ok", flush=True)
np.save(ROOT / f"work/nexgddp/{var}_{scen}_{y0}_{y1}_monthly.npy", out)
print("wrote", out.shape)
