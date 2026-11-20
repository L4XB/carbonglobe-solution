#!/usr/bin/env python
"""Pixel profiles of the ORNL DAAC 2522 layers through the NODE (= centre of the competition's 0.5-deg cell) of a few sites.

grid_edges.py showed kinks of log(SSP / REF) exactly at the cell centres: the 0.01-deg product is a bilinear interpolation between 0.5-deg nodes.
This prints, for the given sites, the pixel values of several layers along the row and the column through the node (3-pixel-wide strips,
100 pixels = the node's own cell and half of each neighbour), to see in which space (biomass or log) the interpolation is linear.

    <scratchpad>/venvs/geotiff/bin/python scripts/ornl/node_profiles.py 1750 3911 3984
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "work" / "ornl2522"
LAYERS = {"new_ref60": ("Biomass_REF_new_forest_2051_2060", 9), "new_585_60": ("Biomass_GFDL-ESM4_SSP585_CLIM_CO2_new_forest_2051_2060", 9),
          "ex_ref21": ("Biomass_REF_existing_forest_2021_2030", 0), "ex_ref60": ("Biomass_REF_existing_forest_2051_2060", 9),
          "ex_585_60": ("Biomass_GFDL-ESM4_SSP585_CLIM_CO2_existing_forest_2051_2060", 9)}
HALF = 75          # pixels on each side of the node


def window(tif: tifffile.TiffFile, band: int, r0: int, c0: int, n: int) -> np.ndarray:
    p = tif.pages[0]; H, W, B = p.shape; tl, tw = p.tilelength, p.tilewidth; ntc = (W + tw - 1) // tw
    out = np.full((n, n), -32768, np.int16); fh = tif.filehandle
    for tr in range(max(r0, 0) // tl, (min(r0 + n, H) - 1) // tl + 1):
        for tc in range(max(c0, 0) // tw, (min(c0 + n, W) - 1) // tw + 1):
            idx = tr * ntc + tc
            if p.databytecounts[idx] == 0:
                continue
            fh.seek(p.dataoffsets[idx]); seg = np.asarray(p.decode(fh.read(p.databytecounts[idx]), idx)[0]).reshape(tl, tw, B)[:, :, band]
            ra, rb = max(r0, tr * tl), min(r0 + n, (tr + 1) * tl); ca, cb = max(c0, tc * tw), min(c0 + n, (tc + 1) * tw)
            out[ra - r0:rb - r0, ca - c0:cb - c0] = seg[ra - tr * tl:rb - tr * tl, ca - tc * tw:cb - tc * tw]
    return np.where(out == -32768, np.nan, out.astype(np.float32) * 0.01)


def main() -> None:
    sites = [int(s) for s in sys.argv[1:]]
    ex = np.load(ROOT / "work/test_cell_exact.npz"); lat, lon = ex["lat"], ex["lon"]
    wins = {}
    for name, (stem, band) in LAYERS.items():
        with tifffile.TiffFile(D / f"{stem}.tif") as tif:
            for s in sites:
                r0 = int(round((80.0 - lat[s]) / 0.01)) - HALF; c0 = int(round((lon[s] + 180.0) / 0.01)) - HALF      # node = corner between pixels HALF - 1 and HALF
                wins[(name, s)] = window(tif, band, r0, c0, 2 * HALF)
    np.savez(D / "node_profiles.npz", **{f"{k[0]}_{k[1]}": v for k, v in wins.items()})
    for s in sites:
        print(f"site {s} lat {lat[s]} lon {lon[s]} (node between pixels {HALF - 1} and {HALF}; neighbour nodes at {HALF - 51}|{HALF - 50} and {HALF + 49}|{HALF + 50})")
        for name in LAYERS:
            w = wins[(name, s)]
            colp = np.nanmedian(w[HALF - 2:HALF + 2, :], axis=0); rowp = np.nanmedian(w[:, HALF - 2:HALF + 2], axis=1)
            print(f"  {name:10s} W->E: " + " ".join("  ." if not np.isfinite(v) else f"{v:.1f}" for v in colp[HALF - 60:HALF + 60:3]))
            print(f"  {name:10s} N->S: " + " ".join("  ." if not np.isfinite(v) else f"{v:.1f}" for v in rowp[HALF - 60:HALF + 60:3]))


if __name__ == "__main__":
    main()
