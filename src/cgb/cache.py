"""Load the working caches built by ``scripts/cache_arrays.py``.

The three caches are the whole training set with the axes the task does not use
removed -- the degenerate month axis of ``glob_Y``, and the month axis of the
forcing split into an annual series and a seasonal climatology:

``y_traj``   (15, 54152, 41, 7) float32   trajectories, seed age x site x slot x var
``x_annual`` (54152, 40, 136)   float32   drivers averaged over each forcing year
``x_season`` (54152, 136, 12)   float32   drivers averaged over the 40 years

Nothing here fits or scores anything; it only opens arrays and builds the feature
blocks the ladder and the rollout study share, so both are looking at exactly the
same numbers.
"""

from __future__ import annotations

import numpy as np

from .io import ROOT, SEED_AGES, load_stats

CACHE_DIR = ROOT / "work" / "cache"

#: Drivers used as named aggregates. Identified by value, not by a layout claim:
#: driver 0 has a training mean of 283.1 (air temperature in K), driver 1 a mean
#: of 59.8 (precipitation flux), driver 8 a mean of 385.4 and is site-invariant
#: and rising (CO2 in ppm). Checked in ``scripts/baseline_ladder.py``.
D_TAIR, D_PRECIP, D_CO2 = 0, 1, 8

#: The eight drivers measured to be constant over time at a site, i.e. static site
#: properties rather than forcing (``notes/instrument.md`` section 6).
STATIC_DRIVERS = (32, 129, 130, 131, 132, 133, 134, 135)


def open_traj(mmap: bool = True) -> np.ndarray:
    """Return ``(15, 54152, 41, 7)`` trajectories, memory-mapped by default."""
    return np.load(CACHE_DIR / "y_traj.npy", mmap_mode="r" if mmap else None)


def open_x_annual(mmap: bool = True) -> np.ndarray:
    """Return ``(54152, 40, 136)`` annual-mean drivers."""
    return np.load(CACHE_DIR / "x_annual.npy", mmap_mode="r" if mmap else None)


def open_x_season(mmap: bool = True) -> np.ndarray:
    """Return ``(54152, 136, 12)`` 40-year-mean seasonal cycles."""
    return np.load(CACHE_DIR / "x_season.npy", mmap_mode="r" if mmap else None)


def scoring_divisors() -> np.ndarray:
    """``[height, agb]`` global means -- the competition's scaling constants."""
    return np.asarray(load_stats()["y_mean"][:2], dtype=np.float64)


def state_block(slot: int) -> np.ndarray:
    """``(15, 54152, 7)`` state at one year slot, all seed ages, physical units."""
    traj = open_traj()
    return np.asarray(traj[:, :, slot, :], dtype=np.float64)


def climate_summary() -> dict[str, np.ndarray]:
    """Per-site 40-year aggregates of the forcing, plus per-site linear trends.

    ``mean`` is the 40-year mean of each driver and ``trend`` its ordinary
    least-squares slope per year, both ``(54152, 136)``. A trend is included
    because a 40-step rollout is driven by where the climate *goes*, not only by
    where it sits, and CO2 in particular rises monotonically across the window.
    """
    x = open_x_annual()
    n_sites = x.shape[0]
    years = np.arange(40, dtype=np.float64)
    years -= years.mean()
    denom = float((years**2).sum())

    mean = np.empty((n_sites, 136), dtype=np.float64)
    trend = np.empty((n_sites, 136), dtype=np.float64)
    step = 4000
    for lo in range(0, n_sites, step):
        hi = min(lo + step, n_sites)
        block = np.asarray(x[lo:hi], dtype=np.float64)  # (chunk, 40, 136)
        mean[lo:hi] = block.mean(axis=1)
        trend[lo:hi] = np.tensordot(years, block, axes=(0, 1)) / denom
    return {"mean": mean, "trend": trend}


def age_index(age: int) -> int:
    return SEED_AGES.index(age)
