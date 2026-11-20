"""Measure what is actually in the two competition ``.npz`` files.

Streams over sites so the 1.6 GB forcing array is never fully resident. Every
number printed comes from the file; nothing here is a guess about the physics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import io  # noqa: E402

CHUNK = 256


def driver_stats(x: np.ndarray) -> dict[str, np.ndarray]:
    """Streaming min/max/mean/var per driver, plus the axes each driver varies on."""
    n_sites, n_years, n_months, n_drv = x.shape
    lo = np.full(n_drv, np.inf, dtype=np.float64)
    hi = np.full(n_drv, -np.inf, dtype=np.float64)
    total = np.zeros(n_drv, dtype=np.float64)
    total_sq = np.zeros(n_drv, dtype=np.float64)
    n_nan = np.zeros(n_drv, dtype=np.int64)
    # Per-driver: does it change within a site across (year, month)? Does its
    # site-mean change across sites?
    varies_in_time = np.zeros(n_drv, dtype=bool)
    varies_in_month = np.zeros(n_drv, dtype=bool)
    varies_in_year = np.zeros(n_drv, dtype=bool)
    site_means = np.empty((n_sites, n_drv), dtype=np.float64)

    count = 0
    for start in range(0, n_sites, CHUNK):
        block = np.asarray(x[start : start + CHUNK], dtype=np.float64)
        flat = block.reshape(-1, n_drv)
        n_nan += np.isnan(flat).sum(axis=0)
        lo = np.minimum(lo, np.nanmin(flat, axis=0))
        hi = np.maximum(hi, np.nanmax(flat, axis=0))
        total += np.nansum(flat, axis=0)
        total_sq += np.nansum(flat * flat, axis=0)
        count += flat.shape[0]
        site_means[start : start + block.shape[0]] = np.nanmean(
            block.reshape(block.shape[0], -1, n_drv), axis=1
        )
        per_site = block.reshape(block.shape[0], -1, n_drv)
        varies_in_time |= (per_site.max(axis=1) != per_site.min(axis=1)).any(axis=0)
        # month variation with the year held fixed, and year variation with the
        # month held fixed -- a driver can be seasonal, secular, or both.
        varies_in_month |= (block.max(axis=2) != block.min(axis=2)).any(axis=(0, 1))
        varies_in_year |= (block.max(axis=1) != block.min(axis=1)).any(axis=(0, 1))

    mean = total / count
    var = np.maximum(total_sq / count - mean * mean, 0.0)
    return {
        "min": lo,
        "max": hi,
        "mean": mean,
        "std": np.sqrt(var),
        "n_nan": n_nan,
        "varies_in_time": varies_in_time,
        "varies_in_month": varies_in_month,
        "varies_in_year": varies_in_year,
        "site_means": site_means,
    }


def main() -> None:
    stats = io.load_stats()
    x_mean, x_std = stats["x_mean"], stats["x_std"]

    per_scenario = {}
    for scenario in io.SCENARIOS:
        x, y0 = io.open_test(scenario)
        print(f"\n================ {scenario} ================")
        print(f"test_x  {x.shape} {x.dtype}   test_y0 {y0.shape} {y0.dtype}")
        s = driver_stats(x)
        per_scenario[scenario] = (s, y0)

        constant = s["std"] == 0.0
        site_var = s["site_means"].std(axis=0)
        print(f"drivers total                 : {x.shape[-1]}")
        print(f"  NaN cells                   : {int(s['n_nan'].sum())}")
        print(f"  globally constant (std==0)  : {int(constant.sum())}  -> {np.flatnonzero(constant).tolist()}")
        print(f"  constant within every site  : {int((~s['varies_in_time']).sum())}")
        print(f"  vary across months          : {int(s['varies_in_month'].sum())}")
        print(f"  vary across years           : {int(s['varies_in_year'].sum())}")
        print(f"  identical at every site     : {int((site_var == 0).sum())}")
        static = (~s["varies_in_time"]) & (site_var > 0)
        print(f"  static per site (site-only) : {int(static.sum())}  -> {np.flatnonzero(static).tolist()[:40]}")

        # Does the shipped training normalisation describe these drivers? If the
        # ordering or the units differed, z would not be near zero-mean/unit-sd.
        z_mean = (s["mean"] - x_mean) / np.where(x_std == 0, 1.0, x_std)
        z_std = s["std"] / np.where(x_std == 0, 1.0, x_std)
        print(f"  z-mean vs training stats    : median {np.median(z_mean):+.3f}  "
              f"p95 |z| {np.percentile(np.abs(z_mean), 95):.3f}  max |z| {np.abs(z_mean).max():.3f}")
        print(f"  z-std  vs training stats    : median {np.median(z_std):.3f}  "
              f"min {z_std.min():.3f}  max {z_std.max():.3f}")

        print("\n  y0 by seed age (mean over sites):")
        print(f"    {'age':>5}  " + "  ".join(f"{n:>9}" for n in io.STATE_NAMES))
        for j, age in enumerate(io.SEED_AGES):
            row = y0[:, j, :].mean(axis=0)
            print(f"    {age:>5}  " + "  ".join(f"{v:9.4f}" for v in row))
        print(f"    y0 range   height [{y0[...,0].min():.4f}, {y0[...,0].max():.4f}]  "
              f"agb [{y0[...,1].min():.4f}, {y0[...,1].max():.4f}]")
        zero_sites = int((np.abs(y0).sum(axis=(1, 2)) == 0).sum())
        print(f"    sites whose whole y0 is zero: {zero_sites}")
        # Is y0 monotone in seed age, per site?
        for name, k in (("height", 0), ("agb", 1)):
            diffs = np.diff(y0[:, :, k], axis=1)
            print(f"    {name}: nondecreasing in age at {float((diffs >= 0).all(axis=1).mean()):.4%} of sites")

    (s126, y126), (s585, y585) = per_scenario["ssp126"], per_scenario["ssp585"]
    print("\n================ ssp126 vs ssp585 ================")
    print(f"test_y0 identical across scenarios: {bool(np.array_equal(y126, y585))}")
    same = np.isclose(s126["mean"], s585["mean"], rtol=1e-9, atol=0.0) & np.isclose(
        s126["std"], s585["std"], rtol=1e-9, atol=0.0
    )
    print(f"drivers with identical mean+std   : {int(same.sum())} of {same.size}")
    differ = np.flatnonzero(~same)
    print(f"drivers that differ               : {differ.tolist()}")
    for d in differ[:12]:
        print(f"  driver {d:3d}  126 mean {s126['mean'][d]:12.5f} std {s126['std'][d]:10.5f}   "
              f"585 mean {s585['mean'][d]:12.5f} std {s585['std'][d]:10.5f}")


if __name__ == "__main__":
    main()
