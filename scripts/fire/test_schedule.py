#!/usr/bin/env python
"""Yearly fire ignition factors of the TEST windows under two climate hypotheses, plus the emulated old-stand outcome.
  fut : the fire sees the windows' own monthly climate (released test T / pr mapped back with the twin calibration)
  mer : the fire sees the exact twin's MERRA-2 training years (year k -> k)
Writes work/fire/test_W.npz: W_fut (2, n_site, 40), W_mer (n_site, 40), sites-level emulated log ratios, twin truth.
    python scripts/fire/test_schedule.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "fire"))
from cgb import io, shift  # noqa: E402
import emulator as em  # noqa: E402

DIV = np.array([10.90645, 3.39567]); M = 1.14114


def main() -> None:
    z = np.load(ROOT / "work/cal/cal_maps_cal.npz"); al, be = z["alpha"][:2].astype(float), z["beta"][:2].astype(float)
    ex = np.load(ROOT / "work/test_cell_exact.npz"); cell, lat = ex["cell"], ex["lat"]
    ns = len(cell)
    xtr = io.open_train_x()
    Wf = np.zeros((2, ns, 40)); Wm = np.full((ns, 40), np.nan); Xf = np.zeros((2, ns, 40, 12), dtype=np.float32)
    xs = {sc: io.open_test(sc)[0] for sc in ("ssp126", "ssp585")}
    for s in range(ns):
        la = float(lat[s]) if np.isfinite(lat[s]) else 0.0
        prevT = prevP = None
        if cell[s] >= 0:
            tw = np.asarray(xtr[int(cell[s]), :, :, :2], dtype=np.float64)
            Wm[s] = em.xroll(tw[:, :, 0] - 273.15, tw[:, :, 1], la).mean(1)
            prevT, prevP = tw[-1, :, 0] - 273.15, tw[-1, :, 1]
        for j, sc in enumerate(("ssp126", "ssp585")):
            x = np.asarray(xs[sc][s, :, :, :2], dtype=np.float64)
            T = (x[:, :, 0] - be[0]) / al[0] - 273.15; P = np.clip((x[:, :, 1] - be[1]) / al[1], 0.0, None)
            xr = em.xroll(T, P, la, prevT, prevP); Wf[j, s] = xr.mean(1); Xf[j, s] = xr
        if s % 1000 == 0: print("site", s, flush=True)
    out = ROOT / "work/fire"; out.mkdir(exist_ok=True)
    np.savez(out / "test_W.npz", W_fut=Wf, W_mer=Wm, cell=cell, lat=lat)
    print("saved", out / "test_W.npz")


if __name__ == "__main__":
    main()
