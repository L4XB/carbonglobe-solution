#!/usr/bin/env python
"""Training-world test of an ALLOMETRIC height decode: the year-40 height of a row from its year-40 agb and the cell's own y0 chronosequence.

For a seed of age a the truth's year-40 agb A40 is mapped to the "biomass-equivalent age" a* on the cell's y0 profile (A0(a*) = A40, log-log interpolation
over the 15 seed ages) and the height is read there: H = H0(a*). Idea: under more CO2 / another climate a young stand runs faster along the same
height-biomass path. Compared with the chronosequence at age + 40 and reported by seed age and by the truth's agb change; also with a noisy A40 (the
test would use the product's agb, ~4 % off).

    python scripts/heights/allometric_decode_train.py [n_cells]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io  # noqa: E402

AGES = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]); LA = np.log(AGES)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 54152
    rng = np.random.default_rng(42)
    H0 = np.empty((n, 15)); A0 = np.empty((n, 15)); H40 = np.empty((n, 15)); A40 = np.empty((n, 15))
    for j, a in enumerate(AGES):
        Y = io.open_train_y(int(a))
        y = np.asarray(Y[:n, [0, 40], 11, :2], dtype=np.float64)          # (cells, 2 times, 2 vars)
        H0[:, j] = y[:, 0, 0]; A0[:, j] = y[:, 0, 1]; H40[:, j] = y[:, 1, 0]; A40[:, j] = y[:, 1, 1]
    ok_cell = np.isfinite(H0).all(axis=1) & np.isfinite(A40).all(axis=1) & (A0[:, 10] > 1.0)                      # forest cells
    eh, ea = 0.1, 0.01
    x40 = np.log(np.minimum(AGES + 40, 500))
    rows = []
    for i in np.where(ok_cell)[0]:
        lA = np.log(A0[i] + ea); lH = np.log(H0[i] + eh)
        # monotone envelope of the agb profile (running maximum) so that the inverse exists; flat parts map to the first age that reaches the value
        lAm = np.maximum.accumulate(lA)
        chrono_h = np.exp(np.interp(x40, LA, lH)) - eh; chrono_a = np.exp(np.interp(x40, LA, lA)) - ea
        for j in range(7):                                                # seed ages 1 .. 100
            a40 = A40[i, j]; h40 = H40[i, j]
            if not (a40 > 0.05 and h40 > 0.5):
                continue
            def inv(v):
                lv = np.log(v + ea)
                if lv >= lAm[-1]: return LA[-1], True
                k = int(np.searchsorted(lAm, lv))
                if k == 0: return LA[0], False
                d = lAm[k] - lAm[k - 1]
                return (LA[k - 1] + (lv - lAm[k - 1]) / d * (LA[k] - LA[k - 1]) if d > 1e-9 else LA[k]), False
            la_star, sat = inv(a40)
            h_allo = np.exp(np.interp(la_star, LA, lH)) - eh
            la_n, _ = inv(a40 * np.exp(rng.normal(0, 0.04)))
            h_allo_n = np.exp(np.interp(la_n, LA, lH)) - eh
            rows.append((i, AGES[j], np.log((h40 + eh) / (chrono_h[j] + eh)), np.log((h40 + eh) / (h_allo + eh)), np.log((h40 + eh) / (h_allo_n + eh)),
                         np.log((a40 + ea) / (chrono_a[j] + ea)), sat, np.exp(la_star), h40))
    r = np.array(rows, dtype=float)
    print(f"cells {int(ok_cell.sum())}, rows {len(r)}")
    f = lambda v: float(np.sqrt(np.mean(v ** 2)))
    print("seed age | n | rms log error of the height: chronosequence at age + 40 | allometric (true agb) | allometric (agb with 4 % noise) | bias of the allometric decode | share saturated")
    for a in AGES[:7]:
        m = r[:, 1] == a
        print(f"   {int(a):3d} | {int(m.sum()):6d} | {f(r[m, 2]):.4f} | {f(r[m, 3]):.4f} | {f(r[m, 4]):.4f} | {np.mean(r[m, 3]):+.4f} | {np.mean(r[m, 6]):.2f}")
    print("by the truth's agb effect against the chronosequence (seed ages 10-100):")
    m0 = r[:, 1] >= 10
    for lo, hi in ((-9, -0.5), (-0.5, -0.2), (-0.2, -0.05), (-0.05, 0.05), (0.05, 0.2), (0.2, 9)):
        m = m0 & (r[:, 5] >= lo) & (r[:, 5] < hi)
        if m.sum() > 50:
            print(f"   agb effect [{lo:+.2f}, {hi:+.2f}) n {int(m.sum()):6d}: chrono {f(r[m, 2]):.4f} | allometric {f(r[m, 3]):.4f} | noisy {f(r[m, 4]):.4f} | bias {np.mean(r[m, 3]):+.4f} | truth height effect {np.mean(r[m, 2]):+.4f}")
    np.save(ROOT / "work/heights/allometric_train_rows.npy", r.astype(np.float32))


if __name__ == "__main__":
    main()
