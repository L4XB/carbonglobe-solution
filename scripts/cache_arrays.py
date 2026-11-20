#!/usr/bin/env python
"""Collapse the 25 GB of training arrays into two working caches.

Everything downstream of this script -- the baseline ladder, the rollout drift study
and the error decomposition -- needs the same two objects over and over, and both are
small enough to hold in RAM once the axes the task does not use are removed:

``work/cache/y_traj.npy``   (15, 54152, 41, 7) float32, 932 MB
    Every seed age's full 41-slot trajectory, month axis dropped. Dropping it is
    exact, not an approximation: the twelve months of ``glob_Y`` are bit-identical.
    That is re-asserted here on a sample rather than inherited as a claim, because
    the whole ladder silently loses meaning if it ever stops being true.

``work/cache/x_annual.npy`` (54152, 40, 136) float32, 1.18 GB
    Each driver averaged over the twelve months of its forcing year. This *is* a
    reduction -- 24 of the 136 drivers are seasonal-only and their annual mean throws
    the season away -- so the seasonal cycle is preserved separately in

``work/cache/x_season.npy`` (54152, 136, 12) float32, 354 MB
    the same drivers averaged over the 40 years instead, one value per calendar
    month. Between the two, a model can see both the year-to-year forcing and the
    site's seasonality without ever touching the 14 GB original again.

Run once:  ./.venv/bin/python scripts/cache_arrays.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import io  # noqa: E402

CACHE = io.ROOT / "work" / "cache"
CHUNK = 4000


def assert_month_degenerate(rng: np.random.Generator) -> None:
    """Re-prove that ``glob_Y``'s month axis carries nothing, on random cells."""
    for age in (1, 100, 500):
        y = io.open_train_y(age)
        sites = np.sort(rng.choice(y.shape[0], size=64, replace=False))
        for slot in (0, 1, 20, 40):
            block = np.asarray(y[sites, slot])  # (64, 12, 7)
            spread = np.abs(block - block[:, :1, :]).max()
            if spread != 0.0:
                raise AssertionError(
                    f"glob_Y{age:03d} slot {slot}: months differ by up to {spread!r}; "
                    "the month axis is no longer degenerate and dropping it is now lossy"
                )
    print("  month axis: bit-identical across all 12 months (3 ages x 4 slots x 64 sites)")


def build_y_traj() -> Path:
    out = CACHE / "y_traj.npy"
    arr = np.lib.format.open_memmap(
        out, mode="w+", dtype=np.float32, shape=(len(io.SEED_AGES), io.TRAIN_Y_SHAPE[0], 41, 7)
    )
    for j, age in enumerate(io.SEED_AGES):
        t0 = time.time()
        y = io.open_train_y(age)
        for lo in range(0, y.shape[0], CHUNK):
            hi = min(lo + CHUNK, y.shape[0])
            arr[j, lo:hi] = y[lo:hi, :, 0, :]
        print(f"  glob_Y{age:03d} -> slot 0..40, month 0   {time.time() - t0:5.1f}s", flush=True)
    arr.flush()
    return out


def build_x() -> tuple[Path, Path]:
    x = io.open_train_x()
    n_sites = x.shape[0]
    annual = np.lib.format.open_memmap(
        CACHE / "x_annual.npy", mode="w+", dtype=np.float32, shape=(n_sites, 40, 136)
    )
    season = np.lib.format.open_memmap(
        CACHE / "x_season.npy", mode="w+", dtype=np.float32, shape=(n_sites, 136, 12)
    )
    t0 = time.time()
    for lo in range(0, n_sites, CHUNK):
        hi = min(lo + CHUNK, n_sites)
        block = np.asarray(x[lo:hi], dtype=np.float32)  # (chunk, 40, 12, 136)
        annual[lo:hi] = block.mean(axis=2)
        season[lo:hi] = block.mean(axis=1).transpose(0, 2, 1)
        if lo % (CHUNK * 4) == 0:
            print(f"  sites {lo:6d}..{hi:6d}   {time.time() - t0:5.1f}s", flush=True)
    annual.flush()
    season.flush()
    return CACHE / "x_annual.npy", CACHE / "x_season.npy"


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    print("checking the assumption that makes this cache exact:")
    assert_month_degenerate(rng)

    print("\nbuilding work/cache/y_traj.npy  (15, 54152, 41, 7):")
    p = build_y_traj()
    print(f"  wrote {p} ({p.stat().st_size / 1e6:.0f} MB)")

    print("\nbuilding work/cache/x_annual.npy and x_season.npy from the 14 GB forcing:")
    a, s = build_x()
    print(f"  wrote {a} ({a.stat().st_size / 1e6:.0f} MB)")
    print(f"  wrote {s} ({s.stat().st_size / 1e6:.0f} MB)")

    # Cheap sanity: the caches must reproduce the originals exactly on random cells.
    y = io.open_train_y(100)
    traj = np.load(CACHE / "y_traj.npy", mmap_mode="r")
    j = io.SEED_AGES.index(100)
    sites = np.sort(rng.choice(y.shape[0], size=32, replace=False))
    if not np.array_equal(np.asarray(y[sites, :, 0, :]), np.asarray(traj[j, sites])):
        raise AssertionError("y_traj cache disagrees with glob_Y100")
    x = io.open_train_x()
    ann = np.load(CACHE / "x_annual.npy", mmap_mode="r")
    s32 = np.sort(rng.choice(x.shape[0], size=8, replace=False))
    ref = np.asarray(x[s32], dtype=np.float32).mean(axis=2)
    got = np.asarray(ann[s32])
    err = float(np.abs(ref - got).max())
    if err > 0:
        raise AssertionError(f"x_annual cache disagrees with glob_X_fea by {err!r}")
    print("\n  spot check: both caches reproduce the originals exactly")


if __name__ == "__main__":
    main()
