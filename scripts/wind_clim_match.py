#!/usr/bin/env python
"""Geolocation key: the test's 24 hourly wind(?) channels (36 + 4h) are a fixed monthly-hourly climatology, identical
every year, while in training they vary by year. If the test climatology is the multi-year mean of the same reanalysis,
a test site matches the training site at (or near) its location. Builds the training sites' 40-year means of those
channels (work/train_seas_clim.npy, (54152, 12, 24)) and the test sites' nearest training sites in that space
(work/test_seas_nn.npz: index, distance, second distance; distances in units of the per-channel training sd).

    ./.venv/bin/python scripts/wind_clim_match.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io  # noqa: E402

SEAS = list(range(36, 129, 4))


def main() -> None:
    out = ROOT / "work" / "train_seas_clim.npy"
    if not out.exists():
        X = io.open_train_x()                                  # (54152, 40, 12, 136)
        clim = np.zeros((X.shape[0], 12, len(SEAS)), dtype=np.float32)
        for i in range(0, X.shape[0], 1000):
            blk = np.asarray(X[i:i + 1000], dtype=np.float32)[..., SEAS]   # (n, 40, 12, 24)
            clim[i:i + 1000] = blk.mean(axis=1)
            if i % 10000 == 0:
                print("train rows", i, flush=True)
        np.save(out, clim)
    clim = np.load(out).reshape(-1, 12 * len(SEAS)).astype(np.float64)
    xt, _ = io.open_test("ssp126")
    tc = np.asarray(xt[:, 0, :, :], dtype=np.float64)[..., SEAS].reshape(-1, 12 * len(SEAS))   # (6171, 288)
    x5, _ = io.open_test("ssp585")
    t5 = np.asarray(x5[:, 0, :, :], dtype=np.float64)[..., SEAS].reshape(-1, 12 * len(SEAS))
    print("test seasonal identical across scenarios:", bool(np.array_equal(tc, t5)))
    sd = clim.std(axis=0) + 1e-9
    A = clim / sd; T = tc / sd
    a2 = (A ** 2).sum(1)
    nn = np.zeros(len(T), dtype=np.int64); d1 = np.zeros(len(T)); d2 = np.zeros(len(T))
    for i in range(0, len(T), 500):
        t = T[i:i + 500]
        d = (t ** 2).sum(1)[:, None] + a2[None, :] - 2 * t @ A.T
        d = np.sqrt(np.maximum(d, 0) / A.shape[1])            # rms distance per channel-month, in sd units
        o = np.argsort(d, axis=1)[:, :2]
        nn[i:i + 500] = o[:, 0]; d1[i:i + 500] = d[np.arange(len(t)), o[:, 0]]; d2[i:i + 500] = d[np.arange(len(t)), o[:, 1]]
    # training -> training reference: nearest other training site for a sample
    rng = np.random.default_rng(42); smp = rng.choice(len(A), 2000, replace=False)
    dd = (A[smp] ** 2).sum(1)[:, None] + a2[None, :] - 2 * A[smp] @ A.T
    dd = np.sqrt(np.maximum(dd, 0) / A.shape[1]); dd[np.arange(len(smp)), smp] = np.inf
    ref = dd.min(1)
    np.savez(ROOT / "work" / "test_seas_nn.npz", nn=nn, d1=d1, d2=d2)
    q = [0, .01, .1, .5, .9, .99]
    print("test->train nearest rms distance quantiles", np.quantile(d1, q).round(4))
    print("test second-nearest", np.quantile(d2, q).round(4))
    print("train->train nearest (other site)", np.quantile(ref, q).round(4))
    print("test sites with d1 < 0.01:", int((d1 < 0.01).sum()), " < 0.05:", int((d1 < 0.05).sum()), " unique nn:", len(set(nn.tolist())))


if __name__ == "__main__":
    main()
