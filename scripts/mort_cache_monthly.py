#!/usr/bin/env python
"""One pass over the 14 GB forcing to keep the monthly series a drought index needs.

``work/cache/x_annual.npy`` averages each driver over its twelve months, which is
exactly the reduction a drought-mortality trigger cannot survive: a dry season is a
run of months, not a low annual mean. This script keeps, at monthly resolution and
for every site, the handful of drivers a water-balance or heat index is built from:

    0  ta_m    monthly air temperature (K)                 driver 0
    1  pr      monthly precipitation                       driver 1
    2  tsl1    soil temperature, top layer                 driver 2
    3  hus     specific humidity, daily mean of 24 hours   drivers 33 + 4k
    4  ta_h    air temperature, daily mean of 24 hours     drivers 34 + 4k
    5  rsds    shortwave down, daily mean of 24 hours      drivers 35 + 4k
    6  wind    wind speed, daily mean of 24 hours          drivers 36 + 4k
    7  ta_hmax air temperature, daily maximum of 24 hours  max over drivers 34 + 4k
    8  hus_min specific humidity, daily minimum            min over drivers 33 + 4k

Written as ``work/cache/mort_monthly_train.npy`` (54152, 40, 12, 9) float32 and
``mort_monthly_{ssp126,ssp585}.npy`` (6171, 40, 12, 9) for the two scored
scenarios, so the same index can be evaluated on the test forcing.

The hourly block layout (interleaved q, T, SW, wind at 24 hours from driver 33) is
the one ``cgb.latent.HOURLY_GROUPS`` uses and ``notes/theory.md`` Sec. 1.1 sources
to the benchmark's ``utils/preprocess.py``.

    ./.venv/bin/python scripts/mort_cache_monthly.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import io  # noqa: E402
from cgb import mortality as mort  # noqa: E402

CHUNK = 2000


def reduce_block(block: np.ndarray) -> np.ndarray:
    """``(n, 40, 12, 136)`` -> ``(n, 40, 12, 9)`` per the layout in the module docstring."""
    hus = block[..., 33 + 4 * np.arange(24)]
    ta_h = block[..., 34 + 4 * np.arange(24)]
    rsds = block[..., 35 + 4 * np.arange(24)]
    wind = block[..., 36 + 4 * np.arange(24)]
    return np.stack(
        [
            block[..., 0],
            block[..., 1],
            block[..., 2],
            hus.mean(axis=-1),
            ta_h.mean(axis=-1),
            rsds.mean(axis=-1),
            wind.mean(axis=-1),
            ta_h.max(axis=-1),
            hus.min(axis=-1),
        ],
        axis=-1,
    ).astype(np.float32)


def build(x: np.ndarray, out: Path) -> None:
    n = x.shape[0]
    arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=(n, 40, 12, mort.N_MONTHLY))
    t0 = time.time()
    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        arr[lo:hi] = reduce_block(np.asarray(x[lo:hi], dtype=np.float32))
        print(f"  {out.name}: sites {lo:6d}..{hi:6d}  {time.time() - t0:6.1f}s", flush=True)
    arr.flush()
    del arr


def main() -> None:
    mort.MONTHLY_DIR.mkdir(parents=True, exist_ok=True)
    build(io.open_train_x(), mort.monthly_path("train"))
    for sc in io.SCENARIOS:
        x, _ = io.open_test(sc)
        build(x, mort.monthly_path(sc))
    # spot check against the original on random cells
    x = io.open_train_x()
    m = np.load(mort.monthly_path("train"), mmap_mode="r")
    rng = np.random.default_rng(0)
    sites = np.sort(rng.choice(x.shape[0], size=4, replace=False))
    ref = reduce_block(np.asarray(x[sites], dtype=np.float32))
    err = float(np.abs(ref - np.asarray(m[sites])).max())
    if err > 0:
        raise AssertionError(f"monthly cache disagrees with glob_X_fea by {err!r}")
    print("spot check: cache reproduces the reduction exactly on 4 random sites")


if __name__ == "__main__":
    main()
