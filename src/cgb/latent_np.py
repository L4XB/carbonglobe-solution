"""Torch-free copies of the two feature helpers the direct GBDT needs from :mod:`cgb.latent`.

``cgb.latent`` imports torch at module level; an interpreter with LightGBM but without torch can still build the
direct-GBDT features through this module. Keep in sync with ``cgb.latent.seasonal_block`` / ``age_features``.
"""
from __future__ import annotations

import numpy as np

from . import cache, io

HOURLY_GROUPS = {"hus": 33, "ta_h": 34, "rsds": 35, "wind": 36}


def seasonal_block(x_season: np.ndarray) -> np.ndarray:
    """``(n_sites, 72)``: monthly climatology of T_air, precip and the four daily-mean hourly groups."""
    n = x_season.shape[0]
    cols = [np.asarray(x_season[:, cache.D_TAIR, :]), np.asarray(x_season[:, cache.D_PRECIP, :])]
    for start in HOURLY_GROUPS.values():
        idx = start + 4 * np.arange(24)
        cols.append(np.asarray(x_season[:, idx, :]).mean(axis=1))
    return np.concatenate([c.reshape(n, 12) for c in cols], axis=1).astype(np.float32)


def age_features(stats: dict[str, np.ndarray]) -> np.ndarray:
    """``(15, 15)`` per seed age: age/500, age-mean state / y_mean, 40 * age slope / y_mean."""
    tri = np.load(io.CACHE / "age_triplet.npz")
    y_mean = stats["y_mean"]
    return np.concatenate([(tri["age"] / 500.0)[:, None], tri["age_mean"] / y_mean, 40.0 * tri["age_slope"] / y_mean],
                          axis=1).astype(np.float32)
