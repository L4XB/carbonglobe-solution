#!/usr/bin/env python
"""Lookup fit of spin-only ED roots (work/edv3/sp_<cfg>/csv/site*.csv) against the decoded test y0, per scenario.

Per root: median |log(ED / y0)|, share within 10 %, median log ratio (bias) for height, agb, lai, gpp, soil by band
(rows with y0 above a small threshold), and the SCENARIO CONTRAST log(x585 / x126): corr and slope ED vs y0.
    python scripts/ed/spin_fit.py work/edv3/sp_cxs_cent400 [more roots] [--common]
--common restricts every root to the sites present in all of them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = {"young": (1, 10, 20, 30), "mid": (50, 70, 100), "old": (150, 200, 250, 300, 350, 400, 450, 500)}
VARS = [("height", "spin_height_m", "y0_height", 0.5), ("agb", "spin_agb_kgC", "y0_agb", 0.2), ("lai", "spin_lai", "y0_lai", 0.2),
        ("gpp", "spin_gpp", "y0_gpp", 0.1), ("soil", "spin_soil", "y0_soil", 0.3)]


def load(root: Path) -> pd.DataFrame:
    fr = [pd.read_csv(f) for f in sorted((root / "csv").glob("site*.csv"))]
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    common = "--common" in sys.argv
    data = {Path(r).name: load(Path(r)) for r in args}
    data = {k: v for k, v in data.items() if len(v)}
    if common:
        keep = set.intersection(*[set(v.site.unique()) for v in data.values()])
        data = {k: v[v.site.isin(keep)] for k, v in data.items()}
    for name, d in data.items():
        d = d.copy(); d["band"] = d.age.map({a: b for b, ages in BANDS.items() for a in ages})
        print(f"== {name}: {d.site.nunique()} sites")
        for b in ("young", "mid", "old", "all"):
            x = d if b == "all" else d[d.band == b]
            cells = []
            for nm, e, y, thr in VARS:
                g = (x[e] > thr) & (x[y] > thr)
                lr = np.log(x.loc[g, e]) - np.log(x.loc[g, y])
                cells.append(f"{nm} med|log| {np.median(np.abs(lr)):.3f} w10 {np.mean(np.abs(lr) < 0.0953):.2f} bias {np.median(lr):+.3f}")
            print(f"  {b:5s} " + " | ".join(cells))
        w = d.pivot_table(index=["site", "age"], columns="scenario", values=[v[1] for v in VARS] + [v[2] for v in VARS])
        cells = []
        for nm, e, y, thr in VARS:
            g = (w[(e, "ssp126")] > thr) & (w[(y, "ssp126")] > thr) & (w[(e, "ssp585")] > thr) & (w[(y, "ssp585")] > thr) & (w.index.get_level_values("age") >= 50)
            ce = np.log(w.loc[g, (e, "ssp585")] / w.loc[g, (e, "ssp126")]); cy = np.log(w.loc[g, (y, "ssp585")] / w.loc[g, (y, "ssp126")])
            # site-level means (ages >= 50)
            ce = ce.groupby("site").mean(); cy = cy.groupby("site").mean()
            sl = np.polyfit(cy, ce, 1)[0] if cy.std() > 0 else np.nan
            cells.append(f"{nm} corr {np.corrcoef(ce, cy)[0, 1]:+.2f} slope {sl:+.2f} (sd y0 {cy.std():.3f}, ED {ce.std():.3f}; mean y0 {cy.mean():+.3f}, ED {ce.mean():+.3f})")
        print("  scenario contrast log(585/126), site means of ages >= 50: " + " | ".join(cells))


if __name__ == "__main__":
    main()
