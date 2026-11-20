#!/usr/bin/env python
"""Does the scored forcing carry the droughts the training runs were hit by?

The events are triggered by a cumulative water-balance deficit with heat on top
(``mort_trigger.py``). The training forcing is MERRA-2 reanalysis, 1982-2021, with
the El Nino droughts in it; the scored forcing is two climate-model scenarios.
Whether an emulator that has learned "12-month deficit beyond X -> collapse" fires
the right number of times on the test runs depends on how often X is exceeded
there. This script measures that, on the same indices, for

  - the 6,171 held-out training sites (``cgb.holdout.make_split(seed=42)``), the
    population the offline score is measured on, and
  - the 6,171 competition sites under ssp126 and ssp585.

Indices are relative to each site's own 40-year climatology, so a different site
sample does not by itself move them; what moves them is the variability of the
forcing. The absolute versions are reported as well, because a stand's carbon
balance responds to the absolute deficit, not to the anomaly.

    ./.venv/bin/python scripts/mort_test_forcing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import holdout, io  # noqa: E402
from cgb import mortality as mort  # noqa: E402
from mort_trigger import build_indices  # noqa: E402

np.set_printoptions(precision=3, suppress=True, linewidth=200)
METRICS = io.ROOT / "work" / "metrics"
KEYS = ("wb12", "wb6", "spi12", "vpd6", "vpd12", "tanom", "dryrun")


def summarise(ix: dict[str, np.ndarray], thr: dict[str, dict[str, float]] | None) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for k in KEYS:
        x = ix[k]
        fin = np.isfinite(x)
        site_max = np.nanmax(x, axis=1)
        d = {
            "mean": float(np.nanmean(x)),
            "sd": float(np.nanstd(x)),
            "q90": float(np.nanquantile(x, 0.90)),
            "q99": float(np.nanquantile(x, 0.99)),
            "site_max_median": float(np.median(site_max)),
            "site_max_q90": float(np.quantile(site_max, 0.90)),
        }
        if thr is not None:
            for q in ("q90", "q99"):
                d[f"frac_over_train_{q}"] = float((x[fin] > thr[k][q]).mean())
            d["frac_sites_over_train_q99"] = float((site_max > thr[k]["q99"]).mean())
        out[k] = d
    return out


def main() -> None:
    split = holdout.make_split(seed=42)
    hold = split.holdout
    m_train = mort.open_monthly("train")
    ix_hold = build_indices(np.asarray(m_train[hold]))
    ix_hold.pop("_pet_over_p")
    ref = summarise(ix_hold, None)
    thr = {k: {"q90": ref[k]["q90"], "q99": ref[k]["q99"]} for k in KEYS}
    ref = summarise(ix_hold, thr)
    per = {"holdout": ref}
    for sc in io.SCENARIOS:
        ix = build_indices(np.asarray(mort.open_monthly(sc)))
        ix.pop("_pet_over_p")
        per[sc] = summarise(ix, thr)

    print("== drought / heat indices, per site-year unless stated; holdout = 6,171 held-out training sites ==")
    for k in KEYS:
        print(f"\n  {k}")
        print(f"    {'set':8s} {'mean':>8s} {'sd':>8s} {'q90':>8s} {'q99':>8s} {'site-max med':>12s} {'site-max q90':>12s} {'>train q90':>10s} {'>train q99':>10s} {'sites>q99':>9s}")
        for name in ("holdout", *io.SCENARIOS):
            d = per[name][k]
            print(f"    {name:8s} {d['mean']:8.3f} {d['sd']:8.3f} {d['q90']:8.3f} {d['q99']:8.3f} {d['site_max_median']:12.3f} {d['site_max_q90']:12.3f} "
                  f"{d['frac_over_train_q90']:10.3f} {d['frac_over_train_q99']:10.4f} {d['frac_sites_over_train_q99']:9.3f}")
    print("\n  reading: '>train q99' is the fraction of site-years beyond the holdout's own 99th percentile (0.010 by construction there);")
    print("  a scenario at 0.003 has a third of the training set's extreme drought years, at 0.03 three times as many.")

    # year-to-year synchrony: how many sites share the same extreme year (the El Nino signature)
    print("\n== synchrony: share of sites whose most stressed wb12 year falls in the busiest single year ==")
    for name, ix in (("holdout", ix_hold),):
        top = np.nanargmax(np.where(np.isfinite(ix["wb12"]), ix["wb12"], -np.inf), axis=1)
        c = np.bincount(top, minlength=40)
        print(f"  {name:8s} busiest year {int(np.argmax(c)) + 1:2d} holds {c.max() / len(top):.3f} of sites; top-3 years {np.sort(c)[::-1][:3] / len(top)}")
    for sc in io.SCENARIOS:
        ix = build_indices(np.asarray(mort.open_monthly(sc)))
        top = np.nanargmax(np.where(np.isfinite(ix["wb12"]), ix["wb12"], -np.inf), axis=1)
        c = np.bincount(top, minlength=40)
        print(f"  {sc:8s} busiest year {int(np.argmax(c)) + 1:2d} holds {c.max() / len(top):.3f} of sites; top-3 years {np.sort(c)[::-1][:3] / len(top)}")

    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "mort_test_forcing.json").write_text(json.dumps(per, indent=1))
    print(f"\nwrote {METRICS / 'mort_test_forcing.json'}")


if __name__ == "__main__":
    main()
