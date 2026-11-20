"""Are the large biomass losses in old stands predictable from forcing, or noise?

The persistence loss at seed ages >= 100 is carried by ~10% of sites (see
theory_growth_curves.py). This script asks whether those losses look like a
deterministic response to weather (events cluster in particular years and among
grid neighbours) or like an independent stochastic disturbance process (a flat
rate, no neighbour concordance). Site coordinates come from ``cache/mask2d.npy``
in the row-major order the tutorial's ``to_grid`` helper uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import io  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=200)
DEC = 11
AGB_MEAN = 3.39567


def site_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(row, col, lat) of every site in site order; site k is the k-th nonzero mask cell."""
    mask = np.load(io.CACHE / "mask2d.npy")
    flat = np.flatnonzero(mask.ravel() > 0)
    row, col = np.divmod(flat, mask.shape[1])
    lat = 90.0 - 0.5 * row - 0.25
    return row, col, lat


def main() -> None:
    row, col, lat = site_grid()
    mask = np.load(io.CACHE / "mask2d.npy")
    site_of = np.full(mask.shape, -1, dtype=np.int64)
    site_of[row, col] = np.arange(row.size)

    for age in (100, 500):
        y = np.load(io.TRAIN / f"glob_Y{age:03d}.npy", mmap_mode="r")
        agb = np.asarray(y[:, :, DEC, 1], dtype=np.float64)  # (N, 41)
        veg = agb[:, 0] > 0.5
        d = np.diff(agb, axis=1)  # (N, 40) year-over-year change
        thr = -1.0
        ev = (d < thr) & veg[:, None]
        print(f"\n== seed age {age}: 'event' = agb drops by more than {abs(thr)} kgC/m^2 in one year; vegetated sites {veg.sum()} ==")
        print(f"   site-years with an event: {ev.sum()} ({ev.sum() / (veg.sum() * 40):.4f} of vegetated site-years)")
        per_year = ev.sum(0)
        print(f"   events per forcing year (1..40): {per_year.tolist()}")
        print(f"   ratio max/median year: {per_year.max() / max(np.median(per_year), 1):.2f}  (a flat stochastic rate would give ~1-1.5)")
        per_site = ev.sum(1)
        print(f"   sites with >=1 event: {(per_site > 0).sum()}, >=3 events: {(per_site >= 3).sum()}, max events at one site: {per_site.max()}")
        # neighbour concordance: for each event (site, year), do the 4-neighbours also have an event that year?
        conc = []
        base = ev[veg].mean()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r2 = np.clip(row + dr, 0, mask.shape[0] - 1)
            c2 = (col + dc) % mask.shape[1]
            nb = site_of[r2, c2]
            ok = (nb >= 0) & veg & veg[np.where(nb >= 0, nb, 0)]
            s, t = np.nonzero(ev & ok[:, None])
            conc.append(ev[nb[s], t].mean() if s.size else np.nan)
        print(f"   P(neighbour has an event in the same year | site has one): {np.nanmean(conc):.3f}  vs base rate {base:.4f}  -> lift x{np.nanmean(conc) / base:.1f}")
        # where does the 40-year persistence loss for agb sit geographically?
        loss = ((agb[:, 40] - agb[:, 0]) / AGB_MEAN) ** 2
        trop = np.abs(lat) < 23.5
        print(f"   share of the age-{age} agb persistence loss from tropical sites (|lat|<23.5): {loss[trop].sum() / loss.sum():.2f} (they are {trop.mean():.2f} of sites)")
        big = loss > np.percentile(loss, 90)
        print(f"   top-10% loss sites: mean |lat| {np.abs(lat[big]).mean():.1f}, mean agb0 {agb[big, 0].mean():.2f} vs all-site mean agb0 {agb[:, 0].mean():.2f}; "
              f"fraction of those with >=1 event {(per_site[big] > 0).mean():.2f}")
        # how much of the top-10% loss is a smooth decline vs a step: compare y40-y0 with the sum of event-year drops
        step = np.where(ev, d, 0.0).sum(1)
        tot = agb[:, 40] - agb[:, 0]
        print(f"   among top-10% loss sites: median (sum of event drops)/(total change) = {np.median(step[big] / np.where(np.abs(tot[big]) > 1e-6, tot[big], np.nan)):.2f}")


if __name__ == "__main__":
    main()
