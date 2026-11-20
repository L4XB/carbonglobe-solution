#!/usr/bin/env python
"""Allometric relaxation directions for the competition's mid and old stands.

The competition's initial stands sit off the training allometry: taller than the training relation
height ~ f(agb, soil, lai, age, climate) gives for their biomass (mean ratio 1.07-1.10 in the mid/old
bands, driven by a tail of tall, nearly empty stands) and ~5% thinner than agb ~ g(height, ...). The board's
mid/old height slopes (0.94 / 0.89) and the smaller-than-predicted agb losses match a relaxation back onto
the allometry over the forty years. This writes the allometric predictions h_allo, a_allo for every test
row (training slot-0 relations, fitted on all training sites) and two probe files that move the mid+old
rows onto them in one column each.

    <lgb venv>/bin/python scripts/allometry_relax.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import cache, io, latent, shift  # noqa: E402

AGES = np.array(io.SEED_AGES)
SOIL = [129, 130, 131, 132, 133, 134, 135]


def rows(u0: np.ndarray, cl: np.ndarray, target: int) -> np.ndarray:
    """u0 (15, n, 4) physical h, agb, soil, lai; the features exclude the target column."""
    keep = [k for k in range(4) if k != target]
    n = u0.shape[1]
    return np.concatenate([np.concatenate([u0[j][:, keep], np.full((n, 1), AGES[j], np.float32), cl], 1) for j in range(15)]).astype(np.float32)


def main() -> None:
    y_mean = io.load_stats()["y_mean"]
    sea = latent.seasonal_block(cache.open_x_season())
    xa = cache.open_x_annual()
    cl = np.concatenate([sea[:, :24], np.asarray(xa[:, 0, SOIL])], 1).astype(np.float32)
    u0 = np.asarray(cache.open_traj()[:, :, 0, :4], dtype=np.float32)
    models = {}
    for target in (0, 1):
        X = rows(u0, cl, target); y = u0[:, :, target].reshape(-1)
        veg = u0[:, :, 0].reshape(-1) > 1.0
        models[target] = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=255, subsample=0.8, subsample_freq=1,
                                           colsample_bytree=0.8, verbose=-1, n_jobs=6).fit(X[veg], y[veg])
        print(f"fitted the {'height' if target == 0 else 'agb'} allometry on {veg.sum():,} rows", flush=True)
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    keys = shift.template_keys(); sc = keys["scenario"].to_numpy(); site = keys["site"].to_numpy(); age = keys["age"].to_numpy()
    allo = np.zeros_like(R)
    for s in io.SCENARIOS:
        y0 = io.open_test(s)[1].astype(np.float64)
        dec = np.stack([1.14114 * (y0[:, :, k] - shift.FLOORS[k]) for k in range(3)] + [1.0 * (y0[:, :, 3] - shift.FLOORS[3])], 2)
        xt = np.load(ROOT / "work" / "cache" / f"test_{s}_annual.npy", mmap_mode="r")
        st = latent.seasonal_block(np.load(ROOT / "work" / "cache" / f"test_{s}_season.npy"))
        clt = np.concatenate([st[:, :24], np.asarray(xt[:, 0, SOIL])], 1).astype(np.float32)
        u0t = np.transpose(dec, (1, 0, 2)).astype(np.float32)
        for target in (0, 1):
            p = models[target].predict(rows(u0t, clt, target)).reshape(15, -1).T  # (6171, 15)
            m = sc == s
            allo[m, target] = np.clip(p[site[m], np.searchsorted(AGES, age[m])], 0, None) / y_mean[target]
    bare = R[:, 0] < 1.0 / y_mean[0]
    allo[bare] = R[bare]
    np.save(ROOT / "work" / "cache" / "allometry_test_scaled.npy", allo)
    band = shift.band_masks(age); mo = band["mid"] | band["old"]
    for target, name in ((0, "height"), (1, "agb")):
        P = R.copy(); P[mo, target] = allo[mo, target]
        path = ROOT / "results" / f"relaxprobe_midold_{name}.csv"
        shift.write_predictions(P, path)
        d = P - R
        print(f"{path.name}: E[(P-R)^2] {np.mean(d**2):.5f}; mid mean change {d[band['mid'], target].mean():+.4f}, old {d[band['old'], target].mean():+.4f}; "
              f"slope of P on R: mid {np.sum(P[band['mid'], target]*R[band['mid'], target])/np.sum(R[band['mid'], target]**2):.3f} old {np.sum(P[band['old'], target]*R[band['old'], target])/np.sum(R[band['old'], target]**2):.3f}")


if __name__ == "__main__":
    main()
