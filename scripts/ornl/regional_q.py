#!/usr/bin/env python
"""Regional old-growth effect around every burnable test site from the full-resolution ORNL DAAC 2522 rasters (GFDL-ESM4 CLIM_CO2 against REF).

The site-level q_old (scripts/ornl/q_proxies.py: median log(B2060_ssp / B2060_REF) over the old-growth pixels of the site's own 0.5-deg cell,
REF 2021 > 1.0 kgC, REF growth in (-0.2, 0.12)) is regionally coherent (corr of a site's q with its neighbours' mean within 300 km 0.65-0.77,
notes 11-11 01:00 UTC), and 735 fire-possible site x scenario have no old-growth pixels of their own (proxies, board kappa 0.6). Here the SAME strict
pixel definition is applied to the 5 x 5 block of 0.5-deg cells around each site (250 x 250 pixels, +-1.25 deg): per site x scenario the median
effect and the pixel count in the own cell (ring 0), the 8 cells around it (ring 1) and the next 16 (ring 2).

Writes work/ornl2522/q_regional.npz: sites (n,), q (n, 2 scenarios, 3 rings), npx (n, 2, 3), b0 (median REF 2021 biomass of the old-growth pixels),
also q_all (all existing-forest pixels, REF 2021 > 0.3).
    <scratchpad>/venvs/geotiff/bin/python scripts/ornl/regional_q.py
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
          "s126": ("Biomass_GFDL-ESM4_SSP126_CLIM_CO2_existing_forest_2051_2060", 9), "s585": ("Biomass_GFDL-ESM4_SSP585_CLIM_CO2_existing_forest_2051_2060", 9)}
K = 2; N = 50 * (2 * K + 1)          # block of (2K + 1)^2 cells, N x N pixels


def main() -> None:
    ex = np.load(ROOT / "work/test_cell_exact.npz"); lat, lon, cellid = ex["lat"], ex["lon"], ex["cell"]
    mx = np.load(ROOT / "work/fire/max_factor_rule833.npy"); cl = np.load(ROOT / "work/collapse_cluster_labels.npy") == 0
    sites = np.array([s for s in range(len(lat)) if np.isfinite(lat[s]) and cellid[s] >= 0 and (mx[s] >= 30 or cl[s])])
    order = np.argsort(np.lexsort((lon[sites], -lat[sites])))      # not used for indexing; sites processed row-major below
    proc = sites[np.lexsort((lon[sites], -lat[sites]))]
    pos = {int(s): i for i, s in enumerate(sites)}
    win = {k: np.full((len(sites), N, N), -32768, np.int16) for k in LAYERS}
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
                if len(cache) > 600:
                    cache.popitem(last=False)
                return seg

            for s in proc:
                r0 = int(round((80.0 - (lat[s] + 0.25 + 0.5 * K)) / 0.01)); c0 = int(round((lon[s] - 0.25 - 0.5 * K + 180.0) / 0.01))
                w = win[name][pos[int(s)]]
                ra0, rb0 = max(r0, 0), min(r0 + N, H); ca0, cb0 = max(c0, 0), min(c0 + N, W)
                if ra0 >= rb0 or ca0 >= cb0:
                    continue
                for tr in range(ra0 // tl, (rb0 - 1) // tl + 1):
                    for tc in range(ca0 // tw, (cb0 - 1) // tw + 1):
                        seg = tile(tr * ntc + tc)
                        ra, rb = max(ra0, tr * tl), min(rb0, (tr + 1) * tl); ca, cb = max(ca0, tc * tw), min(cb0, (tc + 1) * tw)
                        w[ra - r0:rb - r0, ca - c0:cb - c0] = seg[ra - tr * tl:rb - tr * tl, ca - tc * tw:cb - tc * tw]
        print(f"{name}: read after {time.time() - t0:.0f} s; valid share {np.mean(win[name] != -32768):.3f}", flush=True)
    # ring index of every pixel of the block: 0 = own cell, 1 = the 8 cells around, 2 = the outer 16
    ci = np.arange(N) // 50 - K
    ring = np.maximum(np.abs(ci)[:, None], np.abs(ci)[None, :])
    f = lambda a: np.where(a == -32768, np.nan, a.astype(np.float32) * np.float32(0.01))
    q = np.full((len(sites), 2, 3), np.nan, np.float32); npx = np.zeros((len(sites), 2, 3), np.int32); q_all = np.full((len(sites), 2, 3), np.nan, np.float32)
    b0 = np.full((len(sites), 2, 3), np.nan, np.float32)        # median REF 2021 biomass of the old-growth pixels (state check against the site's y0)
    # STATE-MATCHED old-growth pixels: REF 2021 biomass within 0.6..1.7 of the site's own y0 old-growth level (work/ornl2522/yold_site.npy), per ring and
    # for rings 1 + 2 together (index 3): the regional forest that looks like the site's lookup, not the regional median forest
    yold = np.load(D / "yold_site.npy")
    q_sm = np.full((len(sites), 2, 4), np.nan, np.float32); n_sm = np.zeros((len(sites), 2, 4), np.int32)
    for i in range(len(sites)):
        r0 = f(win["r0"][i]); r1 = f(win["r1"][i])
        okx = np.isfinite(r0) & np.isfinite(r1) & (r0 > 0.3) & (r1 > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(okx, np.log(np.maximum(r1, 1e-6) / np.maximum(r0, 1e-6)), np.nan)
        old = okx & (r0 > 1.0) & (g < 0.12) & (g > -0.2)
        for j, sc in enumerate(("s126", "s585")):
            s1 = f(win[sc][i]); m = okx & np.isfinite(s1) & (s1 > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                lq = np.where(m, np.log(np.maximum(s1, 1e-6) / np.maximum(r1, 1e-6)), np.nan)
            for rg in range(3):
                mo = old & m & (ring == rg); npx[i, j, rg] = int(mo.sum())
                if mo.sum() >= 12:
                    q[i, j, rg] = float(np.median(lq[mo])); b0[i, j, rg] = float(np.median(r0[mo]))
                ma = m & (ring == rg)
                if ma.sum() >= 12:
                    q_all[i, j, rg] = float(np.median(lq[ma]))
            yo = float(yold[sites[i], j])
            if yo > 1.0:
                sm = old & m & (r0 >= 0.6 * yo) & (r0 <= 1.7 * yo)
                for rg in range(4):
                    ms = sm & ((ring == rg) if rg < 3 else (ring >= 1)); n_sm[i, j, rg] = int(ms.sum())
                    if ms.sum() >= 12:
                        q_sm[i, j, rg] = float(np.median(lq[ms]))
    np.savez(D / "q_regional.npz", sites=sites, q=q, npx=npx, q_all=q_all, b0=b0, q_sm=q_sm, n_sm=n_sm)
    print(f"wrote {D / 'q_regional.npz'}: {len(sites)} sites; with an old-growth effect in ring 0 / 1 / 2: " + " / ".join(str(int(np.isfinite(q[:, 0, r]).sum())) for r in range(3)), flush=True)


if __name__ == "__main__":
    main()
