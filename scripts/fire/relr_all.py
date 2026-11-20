#!/usr/bin/env python
"""As-delivered monthly climate + shipped wet-month rule: yearly fire factors and the emulated old-stand outcome for
EVERY test site (work/fire/test_W_relr.npz: W (2, n_site, 40) with the rule, W0 without; L (2, n_site) emulated log
ratio of old agb with CO2-faster recovery)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "fire"))
from cgb import io  # noqa: E402
import emulator as em  # noqa: E402
import build_dir as bd  # noqa: E402


def main() -> None:
    ex = np.load(ROOT / "work/test_cell_exact.npz"); lat = ex["lat"]; ns = len(lat)
    xs = {sc: io.open_test(sc)[0] for sc in ("ssp126", "ssp585")}
    W = np.zeros((2, ns, 40)); W0 = np.zeros((2, ns, 40))
    for j, sc in enumerate(("ssp126", "ssp585")):
        x = np.asarray(xs[sc][:, :, :, :2], dtype=np.float64)
        for s in range(ns):
            la = float(lat[s]) if np.isfinite(lat[s]) else 0.0
            T = x[s, :, :, 0] - 273.15; P = x[s, :, :, 1]
            xr = em.xroll(T, P, la); W0[j, s] = xr.mean(1); W[j, s] = (xr * (P < 8.33)).mean(1)
        print(sc, "done", flush=True)
    g = bd.groups(); a0 = g["a0"]
    L = np.zeros((2, ns))
    for j, (rg, aq) in enumerate(((1.5, 1.10), (2.0, 1.20))):
        a = a0.copy()
        for k in range(40):
            r = np.minimum(bd.PRM["cap"], bd.PRM["c"] * a * W[j][:, k])
            a = a * np.exp(-r + em.regrowth(np.log(np.maximum(a0 * aq, 1e-6) / np.maximum(a, 1e-6)), rg))
        L[j] = np.log(np.maximum(a, 0.02) / np.maximum(a0, 0.02))
    np.savez(ROOT / "work/fire/test_W_relr.npz", W=W, W0=W0, L=L, a0=a0)
    print("saved; sites with an emulated loss below -0.1:", int((L.min(0) < -0.1).sum()), " cluster mean", L[:, g["cl"]].mean(1).round(3))


if __name__ == "__main__":
    main()
