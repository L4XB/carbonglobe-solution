#!/usr/bin/env python
"""Are the test sites training cells? For each test site take the training cells within --rad deg of its NEX position
(work/test_latlon_nex3.npz) and compare the seven static soil channels (129-135) under a per-channel affine map
test = a_c * train + b_c (fitted robustly, iterated with the best candidate per site).
Prints the fitted maps and how many sites have a near-exact candidate; writes work/test_cell_match.npz
(cell = index of the best training cell, resid = its normalised rms residual, second = runner-up residual).
    python scripts/geo/cell_identity.py [--rad 1.5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io  # noqa: E402

SOIL = np.arange(129, 136)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rad", type=float, default=1.5)
    ap.add_argument("--pos", default="work/test_latlon_nex3.npz")
    args = ap.parse_args()
    xt, _ = io.open_test("ssp126")
    te = np.asarray(xt[:, 0, 0, :][:, SOIL], dtype=np.float64)                   # (6171, 7)
    te_last = np.asarray(xt[:, 39, 11, :][:, SOIL], dtype=np.float64)
    print("test soil channels static in time? max |first - last| per channel:", np.abs(te - te_last).max(0).round(6))
    xtr = io.open_train_x()
    tr = np.asarray(xtr[:, 0, 0, :][:, SOIL], dtype=np.float64)                  # (54152, 7)
    tll = np.load(ROOT / "work/edv3/train_latlon.npy")
    z = np.load(ROOT / args.pos); pos = np.stack([z["lat"], z["lon"]], 1)
    n = len(te)
    cands = []
    for i in range(n):
        la, lo = pos[i]
        if not np.isfinite(la):
            cands.append(np.array([], int)); continue
        dl = np.abs(((tll[:, 1] - lo + 180) % 360) - 180) * np.cos(np.radians(la))
        cands.append(np.where((np.abs(tll[:, 0] - la) <= args.rad) & (dl <= args.rad))[0])
    print("candidates per site: median", int(np.median([len(c) for c in cands])), " sites without candidates", sum(len(c) == 0 for c in cands))
    sd = tr.std(0); sd[sd == 0] = 1.0
    # start: nearest cell
    best = np.full(n, -1)
    for i in range(n):
        c = cands[i]
        if len(c):
            la, lo = pos[i]
            d = np.hypot(tll[c, 0] - la, (((tll[c, 1] - lo + 180) % 360) - 180) * np.cos(np.radians(la)))
            best[i] = c[np.argmin(d)]
    a = np.ones(7); b = np.zeros(7)
    for it in range(6):
        ok = best >= 0
        for k in range(7):
            x = tr[best[ok], k]; y = te[ok, k]
            if x.std() < 1e-12:
                a[k] = 1.0; b[k] = float(np.median(y - x)); continue
            aa, bb = np.polyfit(x, y, 1)
            for _ in range(5):                                   # trimmed refits
                r = y - (aa * x + bb); keep = np.abs(r) <= np.quantile(np.abs(r), 0.6)
                aa, bb = np.polyfit(x[keep], y[keep], 1)
            a[k], b[k] = aa, bb
        resid = np.full(n, np.nan); second = np.full(n, np.nan)
        for i in range(n):
            c = cands[i]
            if not len(c):
                continue
            r = np.sqrt((((te[i] - (a * tr[c] + b)) / (a * sd)) ** 2).mean(1))
            o = np.argsort(r); best[i] = c[o[0]]; resid[i] = r[o[0]]; second[i] = r[o[1]] if len(c) > 1 else np.nan
        q = np.nanpercentile(resid, [10, 25, 50, 75, 90])
        print(f"iter {it}: a {a.round(5)} b {b.round(5)} | best residual (sd units) p10 {q[0]:.5f} p25 {q[1]:.5f} median {q[2]:.5f} p75 {q[3]:.5f} p90 {q[4]:.5f}; "
              f"share < 0.001: {np.nanmean(resid < 1e-3):.3f}, < 0.01: {np.nanmean(resid < 1e-2):.3f}; runner-up median {np.nanmedian(second):.4f}")
    np.savez(ROOT / "work/test_cell_match.npz", cell=best, resid=resid, second=second, a=a, b=b)
    d = np.hypot(tll[best, 0] - pos[:, 0], (((tll[best, 1] - pos[:, 1] + 180) % 360) - 180) * np.cos(np.radians(pos[:, 0])))
    print("distance best cell - NEX position (deg): median", np.nanmedian(d).round(3), "p90", np.nanpercentile(d, 90).round(3))
    for k, name in enumerate(["k_sat", "sat_theta", "res_theta", "L", "n", "m", "soil_depth"]):
        r = te[:, k] - (a[k] * tr[best, k] + b[k])
        print(f"  ch {SOIL[k]} {name:10s}: train sd {sd[k]:.5g}; |resid| median {np.nanmedian(np.abs(r)):.3g}, p90 {np.nanpercentile(np.abs(r), 90):.3g}")


if __name__ == "__main__":
    main()
