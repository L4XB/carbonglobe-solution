#!/usr/bin/env python
"""Pixel windows of the ORNL DAAC 2522 rasters centred on the NODE of every exact test site (node = centre of the competition's 0.5-deg cell).

scripts/ornl/grid_edges.py / node_profiles.py (11-11 04:00 UTC): the 0.01-deg product is a bilinear interpolation between 0.5-deg lookup-table nodes that
sit at the cell centres, so a cell's pixels mix the node's own response with its neighbours' (own weight 1 at the node, 0.5 at the cell edge, 0.25 at the
corner). The window here is the full support of the node's tent weight: 100 x 100 pixels (node +- 0.5 deg), node between pixels 49 and 50 in both axes.

Writes work/ornl2522/node_win_<layer>.npy (n_sites, 100, 100) int16 (raw values, scale 0.01, nodata -32768) and node_win_sites.npy.
    <scratchpad>/venvs/geotiff/bin/python scripts/ornl/node_windows.py
"""
from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "work" / "ornl2522"
LAYERS = {"r0": ("Biomass_REF_existing_forest_2021_2030", 0), "r1": ("Biomass_REF_existing_forest_2051_2060", 9),
          "s126": ("Biomass_GFDL-ESM4_SSP126_CLIM_CO2_existing_forest_2051_2060", 9), "s585": ("Biomass_GFDL-ESM4_SSP585_CLIM_CO2_existing_forest_2051_2060", 9),
          "n1": ("Biomass_REF_new_forest_2051_2060", 9), "n126": ("Biomass_GFDL-ESM4_SSP126_CLIM_CO2_new_forest_2051_2060", 9),
          "n585": ("Biomass_GFDL-ESM4_SSP585_CLIM_CO2_new_forest_2051_2060", 9)}
HALF = 50


def main() -> None:
    ex = np.load(ROOT / "work/test_cell_exact.npz"); lat, lon, cellid = ex["lat"], ex["lon"], ex["cell"]
    sites = np.array([s for s in range(len(lat)) if np.isfinite(lat[s]) and cellid[s] >= 0])
    proc = sites[np.lexsort((lon[sites], -lat[sites]))]; pos = {int(s): i for i, s in enumerate(sites)}
    np.save(D / "node_win_sites.npy", sites)
    t0 = time.time(); n = 2 * HALF
    for name, (stem, band) in LAYERS.items():
        out = np.full((len(sites), n, n), -32768, np.int16)
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
                if len(cache) > 600:
                    cache.popitem(last=False)
                return seg

            for s in proc:
                r0 = int(round((80.0 - lat[s]) / 0.01)) - HALF; c0 = int(round((lon[s] + 180.0) / 0.01)) - HALF
                w = out[pos[int(s)]]
                ra0, rb0 = max(r0, 0), min(r0 + n, H); ca0, cb0 = max(c0, 0), min(c0 + n, W)
                if ra0 >= rb0 or ca0 >= cb0:
                    continue
                for tr in range(ra0 // tl, (rb0 - 1) // tl + 1):
                    for tc in range(ca0 // tw, (cb0 - 1) // tw + 1):
                        seg = tile(tr * ntc + tc)
                        ra, rb = max(ra0, tr * tl), min(rb0, (tr + 1) * tl); ca, cb = max(ca0, tc * tw), min(cb0, (tc + 1) * tw)
                        w[ra - r0:rb - r0, ca - c0:cb - c0] = seg[ra - tr * tl:rb - tr * tl, ca - tc * tw:cb - tc * tw]
        np.save(D / f"node_win_{name}.npy", out)
        print(f"{name}: {len(sites)} windows after {time.time() - t0:.0f} s; valid share {np.mean(out != -32768):.3f}", flush=True)


if __name__ == "__main__":
    main()
