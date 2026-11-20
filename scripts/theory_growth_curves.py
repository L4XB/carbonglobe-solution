"""Structural facts about the training trajectories that the theory note relies on.

Reads only the seed-age files that are fully downloaded (checked by size), so it
can run while the rest of the training set is still being pulled. Every number
is a measurement on ``data/train/glob_Y{age}.npy``; nothing is fitted.

Questions answered, in order:
  1. mean 41-year trajectory per seed age (the growth curve the emulator must reproduce)
  2. the 40-year change y40 - y0, in the competition's metric units, split by seed
     age and by "bare vs vegetated" site -- i.e. where a persistence-like model
     loses its squared error
  3. how tight the carbon balance d(agb+soil) ~ NPP - Rh is across a year, and the
     spread of NPP/GPP (carbon-use efficiency), which decides whether the flux
     variables are usable as constraints
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import io  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=200)

# Metric: MSE after dividing height by 10.90645 and agb by 3.39567 (README).
METRIC_MEAN = {"height": 10.90645, "agb": 3.39567}
DEC = 11


def complete_age_files() -> list[int]:
    """Seed ages whose glob_Y file is fully present (no .kaggle-partial marker, exact size)."""
    ages = []
    expected = 128 + int(np.prod(io.TRAIN_Y_SHAPE)) * 4
    for age in io.SEED_AGES:
        p = io.TRAIN / f"glob_Y{age:03d}.npy"
        if p.exists() and not (io.TRAIN / f"glob_Y{age:03d}.npy.kaggle-partial").exists():
            if os.path.getsize(p) == expected:
                ages.append(age)
    return ages


def annual_dec(age: int) -> np.ndarray:
    """December state of every year slot: (54152, 41, 7)."""
    y = np.load(io.TRAIN / f"glob_Y{age:03d}.npy", mmap_mode="r")
    return np.asarray(y[:, :, DEC, :], dtype=np.float64)


def main() -> None:
    ages = complete_age_files()
    print(f"complete seed-age files: {ages}")
    names = io.STATE_NAMES

    # ---- 1. growth curves -------------------------------------------------
    print("\n== 1. mean December trajectory per seed age (years 0,1,2,5,10,20,30,40) ==")
    for age in ages:
        y = annual_dec(age)
        veg = y[:, 40, 0] > 0  # vegetated at the end of the run
        print(f"\nseed age {age}: sites {y.shape[0]}, vegetated at yr40 {veg.mean():.3f}")
        print(f"   {'yr':>3} " + " ".join(f"{n:>8}" for n in names))
        for t in (0, 1, 2, 5, 10, 20, 30, 40):
            print(f"   {t:>3} " + " ".join(f"{v:8.3f}" for v in y[veg, t, :].mean(0)) + "   (vegetated sites)")
        # annual increments of height over the run, vegetated sites
        dh = np.diff(y[veg, :, 0], axis=1).mean(0)
        print(f"   mean annual height increment yrs 1-5 {dh[:5].mean():.3f}, 6-20 {dh[5:20].mean():.3f}, 21-40 {dh[20:].mean():.3f}")
        da = np.diff(y[veg, :, 1], axis=1).mean(0)
        print(f"   mean annual agb    increment yrs 1-5 {da[:5].mean():.3f}, 6-20 {da[5:20].mean():.3f}, 21-40 {da[20:].mean():.3f}")

    # ---- 2. where a persistence model loses its squared error --------------
    print("\n== 2. (y40 - y0)^2 / mean^2 : the squared relative 40-year change, by seed age and site class ==")
    print("   this is the per-row loss of 'predict y0' -- the loss floor any model must beat, before")
    print("   climate is even looked at; split into bare (height40==0) and vegetated sites")
    for age in ages:
        y = annual_dec(age)
        for k, n in ((0, "height"), (1, "agb")):
            d = (y[:, 40, k] - y[:, 0, k]) / METRIC_MEAN[n]
            veg = y[:, 40, 0] > 0
            sq = d**2
            share_top10 = np.sort(sq)[::-1][: len(sq) // 10].sum() / sq.sum()
            print(
                f"   age {age:>3} {n:>6}: mean sq rel change {sq.mean():.4f}  "
                f"(vegetated {sq[veg].mean():.4f}, bare {sq[~veg].mean():.4f}, bare share of sites {(~veg).mean():.3f}); "
                f"top-10% of sites carry {share_top10:.2f} of the sum; mean rel change {d.mean():+.3f}, sd {d.std():.3f}"
            )
        # what fraction of the 40-yr change is explained by the age-mean slope alone
        # (the DeepED 'age triplet' prior): residual after subtracting the mean change
        for k, n in ((0, "height"), (1, "agb")):
            d = (y[:, 40, k] - y[:, 0, k]) / METRIC_MEAN[n]
            veg = y[:, 40, 0] > 0
            resid = d[veg] - d[veg].mean()
            print(f"   age {age:>3} {n:>6}: vegetated sites, sq rel change {np.mean(d[veg]**2):.4f} -> after removing the mean change {np.mean(resid**2):.4f}")

    # ---- 3. carbon balance and flux ratios ---------------------------------
    print("\n== 3. carbon-balance coupling, seed age = oldest complete file ==")
    age = ages[-1]
    y = np.load(io.TRAIN / f"glob_Y{age:03d}.npy", mmap_mode="r")
    # annual sums of the monthly fluxes for years 1..40, and pool changes over the same years
    flux = np.asarray(y[:, 1:, :, 4:7], dtype=np.float64).sum(axis=2)  # (N, 40, 3): gpp, npp, rh annual totals
    pools = np.asarray(y[:, :, DEC, :4], dtype=np.float64)  # (N, 41, 4): height, agb, soil, lai at December
    d_agb = np.diff(pools[:, :, 1], axis=1)
    d_soil = np.diff(pools[:, :, 2], axis=1)
    npp, rh, gpp = flux[..., 1], flux[..., 2], flux[..., 0]
    veg = pools[:, 40, 0] > 0
    x = (npp - rh)[veg].ravel()
    z = (d_agb + d_soil)[veg].ravel()
    a, b = np.polyfit(x, z, 1)
    r2 = np.corrcoef(x, z)[0, 1] ** 2
    print(f"   d(agb+soil) vs annual (NPP - Rh): slope {a:.3f}, intercept {b:.4f}, R^2 {r2:.3f}  (vegetated sites, {veg.sum()} sites x 40 yrs)")
    x2 = npp[veg].ravel()
    a2, b2 = np.polyfit(x2, d_agb[veg].ravel(), 1)
    r22 = np.corrcoef(x2, d_agb[veg].ravel())[0, 1] ** 2
    print(f"   d(agb) vs annual NPP: slope {a2:.3f}, intercept {b2:.4f}, R^2 {r22:.3f}")
    cue = npp[veg] / np.where(gpp[veg] > 1e-6, gpp[veg], np.nan)
    print(f"   NPP/GPP (annual, vegetated): median {np.nanmedian(cue):.3f}, IQR [{np.nanpercentile(cue, 25):.3f}, {np.nanpercentile(cue, 75):.3f}]")
    # unit sanity: December monthly GPP vs annual total
    dec_gpp = np.asarray(y[:, 1:, DEC, 4], dtype=np.float64)
    print(f"   annual GPP total / December GPP (median over vegetated site-years): {np.nanmedian(gpp[veg] / np.where(dec_gpp[veg] > 1e-6, dec_gpp[veg], np.nan)):.2f}")
    print(f"   -> the 'gpp' state in y0 is a December monthly value, not an annual mean, if this ratio is far from 1")
    # does agb ever fall? (mortality / disturbance events)
    falls = (d_agb[veg] < -0.05).mean()
    print(f"   fraction of vegetated site-years with agb dropping by >0.05: {falls:.4f}; largest single-year agb drop {d_agb[veg].min():.3f}")


if __name__ == "__main__":
    main()
