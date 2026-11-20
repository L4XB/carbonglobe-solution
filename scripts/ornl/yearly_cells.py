#!/usr/bin/env python
"""Yearly ORNL trajectories (all 10 bands of the 2021_2030 and 2051_2060 files) for a list of test sites: median over the old-growth pixels
(REF 2021 > 1, REF growth 2021 -> 2060 < 0.12) of the biomass, per year, for REF / SSP126 / SSP585 (existing forest).
    <geotiff venv python> scripts/ornl/yearly_cells.py <site file> <out.npz>"""
from __future__ import annotations
import sys
from collections import OrderedDict
from pathlib import Path
import numpy as np
import tifffile
ROOT = Path(__file__).resolve().parents[2]; D = ROOT / "work" / "ornl2522"
sites = np.array([int(x) for x in open(sys.argv[1]).read().split()]); outp = sys.argv[2]
ex = np.load(ROOT / "work/test_cell_exact.npz"); lat, lon = ex["lat"], ex["lon"]
order = sites[np.lexsort((lon[sites], -lat[sites]))]
KIND = __import__("os").environ.get("YEARLY_KIND", "existing")          # "new": the new-forest layers (regrowth from bare ground in 2021)
FILES = {"ref": f"Biomass_REF_{KIND}_forest", "s126": f"Biomass_GFDL-ESM4_SSP126_CLIM_CO2_{KIND}_forest", "s585": f"Biomass_GFDL-ESM4_SSP585_CLIM_CO2_{KIND}_forest"}
pos = {int(s): i for i, s in enumerate(sites)}
cube = {k: np.full((len(sites), 20, 2500), -32768, np.int16) for k in FILES}          # 2021..2030, 2051..2060
for k, stem in FILES.items():
    for di, dec in enumerate(("2021_2030", "2051_2060")):
        with tifffile.TiffFile(D / f"{stem}_{dec}.tif") as tif:
            p = tif.pages[0]; H, W, B = p.shape; tl, tw = p.tilelength, p.tilewidth; ntc = (W + tw - 1) // tw; fh = tif.filehandle
            cache: OrderedDict[int, np.ndarray] = OrderedDict()
            def tile(idx):
                if idx in cache:
                    cache.move_to_end(idx); return cache[idx]
                if p.databytecounts[idx] == 0: seg = np.full((tl, tw, B), -32768, np.int16)
                else:
                    fh.seek(p.dataoffsets[idx]); seg = np.asarray(p.decode(fh.read(p.databytecounts[idx]), idx)[0]).reshape(tl, tw, B)
                cache[idx] = seg
                if len(cache) > 60: cache.popitem(last=False)
                return seg
            for s in order:
                r0 = int(round((80.0 - (lat[s] + 0.25)) / 0.01)); c0 = int(round((lon[s] - 0.25 + 180.0) / 0.01))
                if r0 < 0 or r0 + 50 > H or c0 < 0 or c0 + 50 > W: continue
                win = np.full((50, 50, B), -32768, np.int16)
                for tr in range(r0 // tl, (r0 + 49) // tl + 1):
                    for tc in range(c0 // tw, (c0 + 49) // tw + 1):
                        seg = tile(tr * ntc + tc)
                        ra, rb = max(r0, tr * tl), min(r0 + 50, (tr + 1) * tl); ca, cb = max(c0, tc * tw), min(c0 + 50, (tc + 1) * tw)
                        win[ra - r0:rb - r0, ca - c0:cb - c0] = seg[ra - tr * tl:rb - tr * tl, ca - tc * tw:cb - tc * tw]
                cube[k][pos[int(s)], di * 10:(di + 1) * 10] = win.reshape(2500, B).T[:10]
        print(k, dec, "done", flush=True)
np.savez_compressed(outp, sites=sites, **cube)
