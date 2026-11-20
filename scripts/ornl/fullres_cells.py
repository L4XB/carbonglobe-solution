#!/usr/bin/env python
"""Full-resolution (0.01 deg) ORNL DAAC 2522 pixels of every test site's EXACT 0.5-deg cell (50 x 50 pixels), for the
layers the row predictions need. Decodes only the tiles the cells touch (LRU cache per file).
Writes work/ornl2522/cells_full.npz: sites (n,), <layer> (n, 2500) int16 (x 0.01 = kg C m-2, nodata -32768) for
  r0 = REF existing 2021, r1 = REF existing 2060, s126 / s585 = GFDL CLIM_CO2 existing 2060, nr = REF new 2060, n126 / n585 = GFDL new 2060.
    <scratchpad>/venvs/geotiff/bin/python scripts/ornl/fullres_cells.py
"""
from __future__ import annotations

import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "work" / "ornl2522"
LAYERS = {"r0": ("Biomass_REF_existing_forest_2021_2030", 0), "r1": ("Biomass_REF_existing_forest_2051_2060", 9),
          "s126": ("Biomass_GFDL-ESM4_SSP126_CLIM_CO2_existing_forest_2051_2060", 9), "s585": ("Biomass_GFDL-ESM4_SSP585_CLIM_CO2_existing_forest_2051_2060", 9),
          "nr": ("Biomass_REF_new_forest_2051_2060", 9), "n126": ("Biomass_GFDL-ESM4_SSP126_CLIM_CO2_new_forest_2051_2060", 9),
          "n585": ("Biomass_GFDL-ESM4_SSP585_CLIM_CO2_new_forest_2051_2060", 9)}


def main() -> None:
    ex = np.load(ROOT / "work/test_cell_exact.npz"); lat, lon, cellid = ex["lat"], ex["lon"], ex["cell"]
    sites = np.array([s for s in range(len(lat)) if np.isfinite(lat[s]) and cellid[s] >= 0])
    order = sites[np.lexsort((lon[sites], -lat[sites]))]          # row-major: neighbours share tiles
    out = {k: np.full((len(sites), 2500), -32768, np.int16) for k in LAYERS}
    pos = {int(s): i for i, s in enumerate(sites)}
    t0 = time.time()
    for name, (stem, band) in LAYERS.items():
        with tifffile.TiffFile(D / f"{stem}.tif") as tif:
            p = tif.pages[0]; H, W, B = p.shape; tl, tw = p.tilelength, p.tilewidth; ntc = (W + tw - 1) // tw
            fh = tif.filehandle; cache: OrderedDict[int, np.ndarray] = OrderedDict()

            def tile(idx: int) -> np.ndarray:
                if idx in cache:
                    cache.move_to_end(idx); return cache[idx]
                if p.databytecounts[idx] == 0:
                    seg = np.full((tl, tw), -32768, np.int16)
                else:
                    fh.seek(p.dataoffsets[idx]); data = fh.read(p.databytecounts[idx])
                    seg = np.asarray(p.decode(data, idx)[0]).reshape(tl, tw, B)[:, :, band].copy()
                cache[idx] = seg
                if len(cache) > 400:
                    cache.popitem(last=False)
                return seg

            for s in order:
                r0 = int(round((80.0 - (lat[s] + 0.25)) / 0.01)); c0 = int(round((lon[s] - 0.25 + 180.0) / 0.01))
                if r0 < 0 or r0 + 50 > H or c0 < 0 or c0 + 50 > W:
                    continue
                win = np.full((50, 50), -32768, np.int16)
                for tr in range(r0 // tl, (r0 + 49) // tl + 1):
                    for tc in range(c0 // tw, (c0 + 49) // tw + 1):
                        seg = tile(tr * ntc + tc)
                        ra, rb = max(r0, tr * tl), min(r0 + 50, (tr + 1) * tl); ca, cb = max(c0, tc * tw), min(c0 + 50, (tc + 1) * tw)
                        win[ra - r0:rb - r0, ca - c0:cb - c0] = seg[ra - tr * tl:rb - tr * tl, ca - tc * tw:cb - tc * tw]
                out[name][pos[int(s)]] = win.ravel()
        print(f"{name}: done after {time.time() - t0:.0f} s; valid share {np.mean(out[name] != -32768):.3f}", flush=True)
    np.savez(D / "cells_full.npz", sites=sites, **out)
    print("wrote", D / "cells_full.npz")


if __name__ == "__main__":
    main()
