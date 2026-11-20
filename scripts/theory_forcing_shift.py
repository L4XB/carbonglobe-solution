"""How far the two future scenarios move the drivers away from the training climate,
and what the shipped initial state ``test_y0`` actually is.

Part A reads a handful of drivers from both competition files (memory-mapped, one
pass each) and compares them with the training normalisation statistics in
``cache/data_stats.npz``. Part B characterises ``test_y0`` against the training
year-0 states at the same seed ages. Everything printed is measured.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import io  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=200)

# driver indices, from the block layout in the benchmark repo's utils/preprocess.py
# (clm 8, co2 24, dst 1, mch 96, soi 7) and the means in data_stats.npz
T_AIR, PRECIP, CO2_0, DST, SOIL0 = 0, 1, 8, 32, 129
DEC = 11


def part_a() -> None:
    stats = io.load_stats()
    x_mean, x_std = stats["x_mean"], stats["x_std"]
    print("== A. forcing shift, per scenario: site-mean of selected drivers by forcing year ==")
    print(f"   training stats  T_air mean {x_mean[T_AIR]:.2f} sd {x_std[T_AIR]:.2f}; precip mean {x_mean[PRECIP]:.2f} sd {x_std[PRECIP]:.2f}; CO2[hour0] mean {x_mean[CO2_0]:.2f} sd {x_std[CO2_0]:.2f}")
    per = {}
    for sc in io.SCENARIOS:
        x, _ = io.open_test(sc)
        n = x.shape[0]
        t = np.zeros(40)
        p = np.zeros(40)
        c = np.zeros(40)
        cmax = np.zeros(40)
        for s in range(0, n, 512):
            blk = np.asarray(x[s : s + 512], dtype=np.float64)  # (b, 40, 12, 136)
            t += blk[:, :, :, T_AIR].mean(axis=2).sum(axis=0)
            p += blk[:, :, :, PRECIP].mean(axis=2).sum(axis=0)
            co2 = blk[:, :, :, CO2_0 : CO2_0 + 24].mean(axis=(2, 3))
            c += co2.sum(axis=0)
            cmax = np.maximum(cmax, co2.max(axis=0))
        per[sc] = (t / n, p / n, c / n, cmax)
        print(f"\n   {sc}: year   T_air   precip   CO2(site-mean)  CO2(site-max)")
        for yr in (0, 9, 19, 29, 39):
            print(f"        {yr:>4}  {per[sc][0][yr]:7.2f}  {per[sc][1][yr]:7.2f}  {per[sc][2][yr]:9.2f}       {per[sc][3][yr]:8.2f}")
    d = per["ssp585"]
    e = per["ssp126"]
    print("\n   ssp585 minus ssp126 at year 39: T_air %+.2f K, precip %+.2f, CO2 %+.1f ppm" % (d[0][39] - e[0][39], d[1][39] - e[1][39], d[2][39] - e[2][39]))
    # training CO2 envelope: mean +- 2 sd of the hour-0 CO2 driver
    hi = x_mean[CO2_0] + 2 * x_std[CO2_0]
    print(f"   training CO2 envelope (mean+2sd) ~ {hi:.1f} ppm; ssp126 exceeds it from year "
          f"{int(np.argmax(e[2] > hi)) if (e[2] > hi).any() else 'never'}, ssp585 from year {int(np.argmax(d[2] > hi)) if (d[2] > hi).any() else 'never'}")
    # is the disturbance driver constant, and are the static drivers site constants
    x, _ = io.open_test("ssp126")
    dst = np.asarray(x[:, :, 0, DST])
    print(f"\n   driver {DST} ('dst'): unique values in test {np.unique(dst)[:5]} (count {len(np.unique(dst))})")
    st = np.asarray(x[:, 0, 0, SOIL0:])
    print(f"   static drivers 129..135 first row {st[0]}; last static driver unique count {len(np.unique(st[:, -1]))}")


def part_b() -> None:
    print("\n== B. what is test_y0? ==")
    x126, y126 = io.open_test("ssp126")
    _, y585 = io.open_test("ssp585")
    print(f"   identical across scenarios: {np.array_equal(y126, y585)}")
    names = io.STATE_NAMES
    floor = y126.min(axis=(0, 1))
    at_floor = np.isclose(y126, floor)
    print(f"   per-variable minimum       : {floor}")
    print(f"   fraction of rows at minimum: {at_floor.mean(axis=(0, 1)).round(4)}")
    print("   -> a hard floor hit by ~9% (pools) and ~53% (rh) of rows is a transformed zero, not a physical value")
    for age in (1, 10, 30):
        p = io.TRAIN / f"glob_Y{age:03d}.npy"
        if not p.exists() or (io.TRAIN / f"glob_Y{age:03d}.npy.kaggle-partial").exists():
            continue
        tr = np.asarray(np.load(p, mmap_mode="r")[:, 0, DEC, :], dtype=np.float64)
        te = y126[:, io.SEED_AGES.index(age), :].astype(np.float64)
        print(f"\n   seed age {age}: quantiles [5,25,50,75,95]% of training year-0 (raw) vs test_y0")
        for k, n in enumerate(names):
            print(f"     {n:>6}  train {np.percentile(tr[:, k], [5, 25, 50, 75, 95]).round(3)}   test {np.percentile(te[:, k], [5, 25, 50, 75, 95]).round(3)}   train zero-fraction {(tr[:, k] <= 0).mean():.3f}")
        # fit raw = a*test + b through the upper quantiles, anchored so that the floor maps to raw 0
        print("     slope a s.t. raw ~= a*(test - floor), from the 75/95/99 % quantiles:")
        for k, n in enumerate(names):
            qs = np.array([75, 95, 99])
            a = np.percentile(tr[:, k], qs) / np.maximum(np.percentile(te[:, k], qs) - floor[k], 1e-9)
            print(f"     {n:>6}  a = {a.round(3)}")


if __name__ == "__main__":
    part_a()
    part_b()
