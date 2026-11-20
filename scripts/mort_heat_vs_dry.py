#!/usr/bin/env python
"""Does heat without a precipitation deficit trigger a collapse in the training runs?

``mort_test_forcing.py`` found that the scenario forcing has a tenth of the
training set's precipitation-drought years but more heat and water-balance
anomalies, concentrated in the last decade of ssp585; ``mort_test_rate.py`` found
the latent model predicting 65 % more loss on the biggest stands there. Whether
that is a real response or an extrapolation turns on one table: the event rate in
training site-years that are hot but not dry, dry but not hot, both, or neither.
If ED collapses stands on heat alone, the scenario predictions are supported by
training evidence; if it needs the deficit, they are not.

Cut-offs are quantiles over all vegetated site-years; 'late' restricts to forcing
years 31-40, the band where the scenario extremes sit.

    ./.venv/bin/python scripts/mort_heat_vs_dry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, io  # noqa: E402
from cgb import mortality as mort  # noqa: E402
from mort_trigger import build_indices  # noqa: E402


def main() -> None:
    traj = cache.open_traj()
    old = [j for j, a in enumerate(io.SEED_AGES) if a >= 150]
    agb = np.asarray(traj[old, :, :, 1], dtype=np.float32)
    ev = mort.events(agb)
    veg_all = (agb[:, :, 0] > mort.BARE_AGB).all(axis=0)
    site_ev = (ev.sum(axis=0) >= 4) & veg_all[:, None]
    rel = (agb[:, :, :-1] - agb[:, :, 1:]) / np.where(agb[:, :, :-1] > mort.BARE_AGB, agb[:, :, :-1], np.nan)
    rel_drop = np.nanmean(rel, axis=0)
    ix = build_indices(np.asarray(mort.open_monthly("train")))
    base = float(site_ev[veg_all].mean())
    print(f"vegetated sites {veg_all.sum()}, base event rate per site-year {base:.4f}")
    for dry_key, hot_key in (("spi12", "tanom"), ("spi12", "vpd6"), ("wb12", "tanom")):
        dry = ix[dry_key][veg_all]
        hot = ix[hot_key][veg_all]
        lab = site_ev[veg_all]
        rd = rel_drop[veg_all]
        for q in (0.90, 0.97):
            dq = np.nanquantile(dry, q)
            hq = np.nanquantile(hot, q)
            print(f"\n== dry = {dry_key} > q{q:.2f} ({dq:.3f}), hot = {hot_key} > q{q:.2f} ({hq:.3f}) ==")
            print(f"  {'cell':18s} {'site-years':>10s} {'event rate':>10s} {'lift':>6s} {'mean rel drop':>13s} | {'late (y31-40)':>13s} {'rate':>7s} {'lift':>6s}")
            late = np.zeros_like(lab)
            late[:, 30:] = True
            for name, m in (
                ("dry & hot", (dry > dq) & (hot > hq)),
                ("dry, not hot", (dry > dq) & ~(hot > hq)),
                ("hot, not dry", ~(dry > dq) & (hot > hq)),
                ("neither", ~(dry > dq) & ~(hot > hq)),
            ):
                m = m & np.isfinite(dry) & np.isfinite(hot)
                r = float(lab[m].mean())
                ml = m & late
                rl = float(lab[ml].mean()) if ml.any() else float("nan")
                print(f"  {name:18s} {m.sum():10d} {r:10.4f} {r / base:6.1f} {np.nanmean(rd[m]):+13.4f} | {ml.sum():13d} {rl:7.4f} {rl / base:6.1f}")
    # the same with a stricter notion of 'not dry': precipitation at or above the site's median
    dry = ix["spi12"][veg_all]
    hot = ix["tanom"][veg_all]
    lab = site_ev[veg_all]
    wet = dry <= np.nanmedian(dry, axis=1, keepdims=True)  # the site's wetter half of years
    for q in (0.90, 0.97, 0.99):
        hq = np.nanquantile(hot, q)
        m = wet & (hot > hq)
        print(f"\n  hot (tanom > q{q:.2f} = {hq:+.2f} K) in the site's wetter half of years: {m.sum()} site-years, event rate {lab[m].mean():.4f} (lift x{lab[m].mean() / base:.1f})")
    # how hot are the hottest training years, in absolute anomaly terms, against ssp585's late years
    print(f"\n  training tanom quantiles over vegetated site-years: q90 {np.nanquantile(hot, 0.9):+.2f}, q99 {np.nanquantile(hot, 0.99):+.2f}, max {np.nanmax(hot):+.2f} K")
    for sc in io.SCENARIOS:
        ixs = build_indices(np.asarray(mort.open_monthly(sc)))
        t = ixs["tanom"]
        print(f"  {sc}: tanom in years 31-40: mean {np.nanmean(t[:, 30:]):+.2f}, q90 {np.nanquantile(t[:, 30:], 0.9):+.2f}, q99 {np.nanquantile(t[:, 30:], 0.99):+.2f}; "
              f"share of site-years 31-40 beyond the training q99: {(t[:, 30:] > np.nanquantile(hot, 0.99)).mean():.4f}")


if __name__ == "__main__":
    main()
