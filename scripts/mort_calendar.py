#!/usr/bin/env python
"""Which calendar years are the event years, and where on the globe do they fall?

``mort_census.py`` found the old-stand collapses piled into forcing years 1-3, 7,
16-17 and 34-35. If forcing year 1 is 1981 (the paper says MERRA-2, 1981 onwards)
those are the 1982-83, 1987, 1997-98 and 2015-16 El Nino droughts. Two checks:

  A. pin the calendar from the CO2 driver (Mauna Loa: 340 ppm in 1981, 414 in 2020);
  B. the geography of each big event year -- an El Nino drought sits in Amazonia,
     Indonesia, Australia and southern Africa; a spin-up shock in year 1 would sit
     everywhere old stands sit.

    ./.venv/bin/python scripts/mort_calendar.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, io  # noqa: E402
from cgb import mortality as mort  # noqa: E402

np.set_printoptions(precision=3, suppress=True, linewidth=200)


def main() -> None:
    # ---- A. calendar --------------------------------------------------------
    xa = cache.open_x_annual()
    co2 = np.asarray(xa[:2000, :, 8:32], dtype=np.float64).mean(axis=(0, 2))  # (40,) site-invariant
    print("== A. CO2 (mean of the 24 hourly drivers) by forcing year, first 2000 sites ==")
    print("  " + " ".join(f"y{t + 1:02d}:{co2[t]:.1f}" for t in range(0, 40, 3)))
    print("  Mauna Loa annual means: 1981 340.1, 1990 354.4, 2000 369.6, 2010 389.9, 2020 414.2")
    print("  -> forcing year 1 = 1981 if y01 ~ 340 and y40 ~ 414; y10 ~ 354, y20 ~ 370, y30 ~ 390")

    # ---- B. geography of the event years ------------------------------------
    traj = cache.open_traj()
    old = [j for j, a in enumerate(io.SEED_AGES) if a >= 150]
    agb = np.asarray(traj[old, :, :, 1], dtype=np.float32)  # (8, N, 41)
    ev = mort.events(agb)  # (8, N, 40)
    veg_all = (agb[:, :, 0] > mort.BARE_AGB).all(axis=0)
    k = ev.sum(axis=0)  # (N, 40)
    site_ev = (k >= 4) & veg_all[:, None]  # a site-year where at least half the old ages collapsed
    _, _, lat, lon = mort.site_grid()
    print(f"\n== B. site-years with an event at >= 4 of the 8 old ages: {site_ev.sum()} over {veg_all.sum()} sites vegetated at all old ages ==")
    regions = {
        "Amazonia (lat -20..10, lon -80..-45)": (lat > -20) & (lat < 10) & (lon > -80) & (lon < -45),
        "C. Africa (lat -15..10, lon 5..35)": (lat > -15) & (lat < 10) & (lon > 5) & (lon < 35),
        "S. Africa (lat -35..-15, lon 10..40)": (lat > -35) & (lat < -15) & (lon > 10) & (lon < 40),
        "SE Asia/Indonesia (lat -12..20, lon 90..150)": (lat > -12) & (lat < 20) & (lon > 90) & (lon < 150),
        "Australia (lat -45..-12, lon 110..155)": (lat > -45) & (lat < -12) & (lon > 110) & (lon < 155),
        "N. America (lat 25..70, lon -170..-50)": (lat > 25) & (lat < 70) & (lon > -170) & (lon < -50),
        "Europe (lat 35..70, lon -10..40)": (lat > 35) & (lat < 70) & (lon > -10) & (lon < 40),
        "Siberia/E. Asia (lat 30..75, lon 40..180)": (lat > 30) & (lat < 75) & (lon > 40) & (lon < 180),
        "India (lat 5..30, lon 65..90)": (lat > 5) & (lat < 30) & (lon > 65) & (lon < 90),
    }
    per_year = site_ev.sum(axis=0)
    print("  site-events per forcing year: " + " ".join(f"{t + 1}:{per_year[t]}" for t in range(40)))
    print("  region: vegetated old sites | share of each year's site-events (calendar year in header)")
    years = [1, 2, 3, 4, 6, 7, 11, 16, 17, 18, 21, 24, 25, 26, 28, 34, 35, 39]
    hdr = f"  {'region':44s} {'n':>5s} " + " ".join(f"{1980 + t:>5d}" for t in years)
    print(hdr)
    for name, m in regions.items():
        mv = m & veg_all
        row = [site_ev[mv, t - 1].sum() / max(per_year[t - 1], 1) for t in years]
        print(f"  {name:44s} {mv.sum():5d} " + " ".join(f"{r:5.2f}" for r in row))
    # rate per vegetated site in each region and year (how *intense* the year was there)
    print("\n  fraction of the region's vegetated old sites hit, by year:")
    print(hdr)
    for name, m in regions.items():
        mv = m & veg_all
        row = [site_ev[mv, t - 1].mean() if mv.any() else np.nan for t in years]
        print(f"  {name:44s} {mv.sum():5d} " + " ".join(f"{r:5.2f}" for r in row))
    trop = np.abs(lat) < 23.5
    print(f"\n  tropical share of site-events, by year: " + " ".join(
        f"{1980 + t}:{site_ev[trop & veg_all, t - 1].sum() / max(per_year[t - 1], 1):.2f}" for t in years))
    # how many sites are hit in year 1 that are never hit again, vs hit repeatedly
    n_hits = site_ev.sum(axis=1)
    y1 = site_ev[:, 0]
    print(f"\n  sites hit in year 1: {y1.sum()}; of those, hit in no other year: {(y1 & (n_hits == 1)).sum()}, hit in >= 3 other years: {(y1 & (n_hits >= 4)).sum()}")
    print(f"  sites hit at least once: {(n_hits > 0).sum()}; distribution of hits per site: " + " ".join(
        f"{i}:{(n_hits == i).sum()}" for i in range(1, 8)) + f" >=8:{(n_hits >= 8).sum()}")


if __name__ == "__main__":
    main()
