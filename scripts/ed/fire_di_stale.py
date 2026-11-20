#!/usr/bin/env python
"""ED's fire dryness index as SHIPPED (calcSiteDrynessIndex without the January reset): when January is wet its index
keeps the value of the last dry January (zero until the first one), and a dry February starts from that stale value.

    dryness_stale(T, P, b) -> (DI_avg per year with the stale January, DI_avg per year with January reset [the patched
    build_dryfix behaviour], stale January value per year)
T (years, 12) deg C, P (years, 12) mm/month, b = latitude-band factors (fire_proxy_calib.b_factors).
Run as a script: validation against the organisers' TRAINING trajectories of the tropical forest twins (age 300 yearly
agb): share of loss years (ratio < 0.94) explained, Spearman of the 40-year ratio with burn-year counts, per threshold.
    python scripts/ed/fire_di_stale.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "ed"))


def dryness_stale(T: np.ndarray, P: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Tc = np.maximum(T, 0.0001)
    I = np.sum((Tc / 5.0) ** 1.51, axis=1, keepdims=True)
    a = 67.5e-8 * I ** 3 - 77.1e-6 * I ** 2 + 0.0179 * I + 0.492
    pet = 1.62 * b[None, :] * (10.0 * Tc / I) ** a * 120.0          # mm/yr
    pr = np.maximum(P * 12.0, 1e-9)
    dry = pet / pr > 1.0
    inc = 30.0 * (pet - pr)
    ny = T.shape[0]
    out_s = np.zeros(ny); out_f = np.zeros(ny); jan = np.zeros(ny)
    di_s = np.zeros(12)                                             # persistent (shipped code)
    for y in range(ny):
        di_f = np.zeros(12)
        if dry[y, 0]:
            d0 = inc[y, 0]
            for m2 in range(11, 0, -1):
                if dry[y, m2]:
                    d0 += inc[y, m2]
                else:
                    break
            di_s[0] = d0; di_f[0] = d0
        for m in range(1, 12):
            di_s[m] = di_s[m - 1] + inc[y, m] if dry[y, m] else 0.0
            di_f[m] = di_f[m - 1] + inc[y, m] if dry[y, m] else 0.0
        out_s[y] = di_s.mean(); out_f[y] = di_f.mean(); jan[y] = di_s[0]
    return out_s, out_f, jan


def main() -> None:
    from scipy.stats import spearmanr
    from cgb import io
    import fire_proxy_calib as fp
    z = np.load(ROOT / "work/cal/fire_di.npz"); sites = z["sites"]; cells = z["cells"]
    tw = np.load(ROOT / "work/train_twin_states.npz"); t0 = tw["tr0"][:, 7:, 0, 1].astype(float).mean(1); t40 = tw["tr40"].mean(2)[:, 7:, 1].astype(float).mean(1)
    ex = np.load(ROOT / "work/test_cell_exact.npz"); lat = ex["lat"][sites]
    forest = (t0 > 5) & (np.abs(lat) < 30); fi = np.where(forest)[0]
    ratio = np.log(np.maximum(t40, 0.05) / np.maximum(t0, 0.05))
    xtr = io.open_train_x(); y300 = io.open_train_y(300)
    DS = np.zeros((len(sites), 40)); DF = np.zeros((len(sites), 40)); JAN = np.zeros((len(sites), 40)); YR = np.ones((len(sites), 40))
    for i in fi:
        tr = np.asarray(xtr[int(cells[i]), :, :, :2], dtype=np.float64)
        DS[i], DF[i], JAN[i] = dryness_stale(tr[:, :, 0] - 273.15, tr[:, :, 1], fp.b_factors(float(lat[i])))
        a = np.asarray(y300[int(cells[i]), :, 0, 1], dtype=np.float64)
        YR[i] = a[1:] / np.maximum(a[:-1], 1e-6)
    np.savez(ROOT / "work/cal/fire_di_stale_train.npz", sites=sites, di_stale=DS, di_fixed=DF, jan=JAN, yearly_ratio=YR, forest=forest)
    loss = YR[forest] < 0.94
    print(f"{forest.sum()} tropical forest twins; loss years (yearly agb ratio < 0.94): {loss.mean():.3f} of site-years; sites with a stale January > 0 at some point: {np.mean((JAN[forest] > 0).any(1)):.2f}")
    for nm, D in (("January reset (patched build)", DF[forest]), ("stale January (shipped code)", DS[forest])):
        print(f"== {nm}")
        for thr in (40, 50, 60, 65, 70, 80):
            hit = D > thr * 1000
            # the loss shows up in the same model year or the next: compare with loss[y] | loss[y + 1]
            lossn = loss.copy(); lossn[:, :-1] |= loss[:, 1:]
            hitp = hit.copy(); hitp[:, 1:] |= hit[:, :-1]
            prec = (hit & lossn).sum() / max(hit.sum(), 1); rec = (loss & hitp).sum() / max(loss.sum(), 1)
            F = hit.sum(1)
            print(f"   thr {thr}k: site-years above {hit.mean():.3f}; precision (a loss in that or the next year) {prec:.2f}, recall (losses with an exceedance that or the previous year) {rec:.2f}; "
                  f"Spearman(40-year ratio, count) {spearmanr(ratio[forest], F)[0]:+.3f}")
    i = int(np.where(cells == 43296)[0][0])
    np.set_printoptions(linewidth=250, precision=0, suppress=True)
    print("cell 43296 yearly agb ratio x100:", YR[i] * 100); print("   DI fixed / 1000 :", DF[i] / 1000); print("   DI stale / 1000 :", DS[i] / 1000); print("   stale Jan / 1000:", JAN[i] / 1000)


if __name__ == "__main__":
    main()
