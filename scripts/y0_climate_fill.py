#!/usr/bin/env python
"""Climate-predicted seedling state for the competition sites.

The competition's age-1 initial state is a younger seedling than the training's slot-0 age-1
state (height 0.93x, agb 0.49x, lai 0.60x of what the training climate relation predicts,
while ages >= 10 match it to 1%). A model trained on slot-0 seedlings reads a small seedling as
a harsh site. This writes, per scenario, the training-like age-1 state (height, agb, soil,
lai; physical units) predicted from the site's climatology, soil and first forcing year, so
``predict_latent.py --y0-fill-age1`` can start the age-1 rollout from an in-distribution state.

    <lightgbm venv>/bin/python scripts/y0_climate_fill.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgb import cache, holdout, io, latent  # noqa: E402


def first_year(ann: np.ndarray) -> np.ndarray:
    a = np.asarray(ann[:, 0, :], dtype=np.float32)
    hourly = np.stack([a[:, s + 4 * np.arange(24)].mean(1) for s in (33, 34, 35, 36)], 1)
    return np.concatenate([a[:, :8], hourly], 1)


def main() -> None:
    split = holdout.make_split(seed=42)
    xa = cache.open_x_annual()
    X = np.concatenate([latent.seasonal_block(cache.open_x_season()), np.asarray(xa[:, 0, [129, 130, 131, 132, 133, 134, 135]]), first_year(xa)], 1).astype(np.float32)
    y = np.asarray(cache.open_traj()[0, :, 0, :4], dtype=np.float64)  # age 1, slot 0: height, agb, soil, lai
    out = {}
    for scenario in io.SCENARIOS:
        xt = np.load(cache.CACHE_DIR / f"test_{scenario}_annual.npy", mmap_mode="r")
        Xt = np.concatenate([latent.seasonal_block(np.load(cache.CACHE_DIR / f"test_{scenario}_season.npy")), np.asarray(xt[:, 0, [129, 130, 131, 132, 133, 134, 135]]), first_year(xt)], 1).astype(np.float32)
        out[scenario] = np.zeros((Xt.shape[0], 4))
        for k in range(4):
            m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
            m.fit(X, y[:, k])  # all training sites: this is a deployment fit, the holdout check lives in the scratchpad script
            out[scenario][:, k] = np.clip(m.predict(Xt), [1.5, 0.0, 0.0, 0.0][k], None)
        print(f"{scenario}: predicted age-1 state means {out[scenario].mean(0).round(4)} (training age-1 slot-0 means {y.mean(0).round(4)})")
        np.save(cache.CACHE_DIR / f"y0_clim_age1_{scenario}.npy", out[scenario].astype(np.float32))
    print("wrote work/cache/y0_clim_age1_<scenario>.npy")


if __name__ == "__main__":
    main()
