"""Drought-mortality events in the old stands: definition, features, detector.

``notes/theory.md`` Sec. 4 measured that at seed ages >= 150 the agb loss is not a
drift but a set of single-year collapses at a tenth of the sites, synchronised in
forcing years and among grid neighbours (a 47-84x lift). This module holds the
shared definitions the ``scripts/mort_*.py`` measurements build on, so every one of
them counts the same thing.

An **event** is a year-over-year drop in agb of more than :data:`EVENT_THR` kgC m-2.
The threshold is the one ``theory_mortality_events.py`` used; ``mort_census.py``
checks whether the distribution of annual changes actually has a gap there.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import CACHE, SCENARIOS

#: A single-year agb drop larger than this, in kgC m-2, is an event.
EVENT_THR = 1.0
#: Alternative, relative definition: a drop larger than this fraction of last year's agb.
REL_THR = 0.15
#: A stand with less agb than this at year 0 is treated as bare.
BARE_AGB = 0.5


def events(agb: np.ndarray, *, thr: float = EVENT_THR) -> np.ndarray:
    """Boolean ``(..., 40)`` event mask from an ``(..., 41)`` agb trajectory.

    ``out[..., t]`` is True when forcing year ``t + 1`` dropped agb by more than
    ``thr`` (state slot ``t`` to slot ``t + 1``).
    """
    agb = np.asarray(agb)
    if agb.shape[-1] != 41:
        raise ValueError(f"expected 41 year slots on the last axis, got {agb.shape}")
    drop = agb[..., :-1] - agb[..., 1:]
    return (drop > thr) & (agb[..., :-1] > BARE_AGB)


def site_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(row, col, lat, lon)`` of every training site, in site order.

    Site ``k`` is the ``k``-th nonzero cell of ``cache/mask2d.npy`` in row-major
    order, the convention the organisers' ``to_grid`` helper uses; the grid is
    0.5 degrees with row 0 at the north pole and column 0 at -180.
    """
    mask = np.load(CACHE / "mask2d.npy")
    flat = np.flatnonzero(mask.ravel() > 0)
    row, col = np.divmod(flat, mask.shape[1])
    lat = 90.0 - 0.5 * row - 0.25
    lon = -180.0 + 0.5 * col + 0.25
    return row, col, lat, lon


# ---- monthly forcing cache (built by scripts/mort_cache_monthly.py) ------------

#: Where the monthly reductions live; a work product, rebuilt from the 14 GB forcing.
MONTHLY_DIR = CACHE.parent / "work" / "cache"
#: Channel order of the monthly cache.
MONTHLY_NAMES: tuple[str, ...] = (
    "ta_m", "pr", "tsl1", "hus", "ta_h", "rsds", "wind", "ta_hmax", "hus_min"
)
N_MONTHLY = len(MONTHLY_NAMES)


def monthly_path(which: str) -> Path:
    """Path of the monthly cache for ``"train"``, ``"ssp126"`` or ``"ssp585"``."""
    if which not in ("train", *SCENARIOS):
        raise ValueError(f"unknown forcing set {which!r}")
    return MONTHLY_DIR / f"mort_monthly_{which}.npy"


def open_monthly(which: str) -> np.ndarray:
    """Memory-map the ``(n_sites, 40, 12, N_MONTHLY)`` monthly cache."""
    arr = np.load(monthly_path(which), mmap_mode="r")
    if arr.shape[1:] != (40, 12, N_MONTHLY):
        raise AssertionError(f"{monthly_path(which).name} is {arr.shape}")
    return arr
