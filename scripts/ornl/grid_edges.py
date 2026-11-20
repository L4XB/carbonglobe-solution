#!/usr/bin/env python
"""On which grid does the ORNL DAAC 2522 product see its CLIMATE? Edge profile of the pixel effect around burnable test sites.

Per sample site the 5 x 5 block of 0.5-deg cells (250 x 250 pixels of 0.01 deg, the block starts on a cell boundary of the competition grid) is read
for REF 2060 and the two GFDL CLIM_CO2 2060 layers; lq = log(SSP / REF) per pixel. For every pixel boundary j (between columns j and j + 1, likewise
rows) the mean |lq[j + 1] - lq[j]| over all valid pixel pairs is accumulated. A climate grid shows as peaks of this profile at its cell edges:
0.5-deg forcing on the competition grid -> peaks at j = 49, 99, 149, 199 only; 0.25-deg forcing (NEX-GDDP native) -> also at 24, 74, 124, ...

    <scratchpad>/venvs/geotiff/bin/python scripts/ornl/grid_edges.py [n_sites]
Writes work/ornl2522/grid_edges.npz (profiles, per-site windows of lq for the first 40 sites).
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
LAYERS = {"r1": ("Biomass_REF_existing_forest_2051_2060", 9), "s126": ("Biomass_GFDL-ESM4_SSP126_CLIM_CO2_existing_forest_2051_2060", 9),
          "s585": ("Biomass_GFDL-ESM4_SSP585_CLIM_CO2_existing_forest_2051_2060", 9)}
K = 2; N = 50 * (2 * K + 1)


def main() -> None:
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    reg = np.load(D / "q_regional.npz"); sites_all = reg["sites"]; q = reg["q"]; npx = reg["npx"]
    ex = np.load(ROOT / "work/test_cell_exact.npz"); lat, lon = ex["lat"], ex["lon"]
    # sample: sites whose own cell and ring 1 hold many strict old-growth pixels and whose regional effect is not flat
    score = np.nanmax(np.abs(q[:, :, 1]), axis=1); ok = (npx[:, 0, 0] >= 300) & (npx[:, 0, 1] >= 2000) & np.isfinite(score) & (score > 0.1)
    cand = sites_all[ok]; rng = np.random.default_rng(42); sites = np.sort(rng.choice(cand, size=min(nmax, len(cand)), replace=False))
    print(f"{len(cand)} candidate sites, {len(sites)} sampled", flush=True)
    proc = sites[np.lexsort((lon[sites], -lat[sites]))]; pos = {int(s): i for i, s in enumerate(sites)}
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
        print(f"{name}: read after {time.time() - t0:.0f} s", flush=True)
    f = lambda a: np.where(a == -32768, np.nan, a.astype(np.float32) * np.float32(0.01))
    prof = np.zeros((2, 2, N - 1)); cnt = np.zeros((2, 2, N - 1))        # (scenario, axis: 0 = column edges, 1 = row edges, edge index)
    keep = []
    for i in range(len(sites)):
        r1 = f(win["r1"][i])
        for j, sc in enumerate(("s126", "s585")):
            s1 = f(win[sc][i]); m = np.isfinite(r1) & np.isfinite(s1) & (r1 > 0.5) & (s1 > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                lq = np.where(m, np.log(np.maximum(s1, 1e-6) / np.maximum(r1, 1e-6)), np.nan)
            dc = np.abs(lq[:, 1:] - lq[:, :-1]); dr = np.abs(lq[1:, :] - lq[:-1, :])
            prof[j, 0] += np.nansum(dc, axis=0); cnt[j, 0] += np.isfinite(dc).sum(axis=0)
            prof[j, 1] += np.nansum(dr, axis=1); cnt[j, 1] += np.isfinite(dr).sum(axis=1)
            if i < 40 and j == 1:
                keep.append(lq.astype(np.float32))
    prof = prof / np.maximum(cnt, 1)
    np.savez(D / "grid_edges.npz", prof=prof, cnt=cnt, sites=sites, lq=np.array(keep))
    for j, sc in enumerate(("ssp126", "ssp585")):
        for ax, an in enumerate(("column edges (longitude)", "row edges (latitude)")):
            pr = prof[j, ax]; base = np.median(pr)
            top = np.argsort(-pr)[:12]
            print(f"{sc} {an}: median |step| {base:.4f}; largest at edge index " + ", ".join(f"{int(t)} ({pr[t] / base:.1f}x)" for t in sorted(top)))
            for per, nm in ((50, "0.5 deg"), (25, "0.25 deg"), (10, "0.1 deg")):
                ph = np.array([pr[(np.arange(N - 1) % per) == k].mean() for k in range(per)])
                print(f"     folded at {nm}: max phase {int(np.argmax(ph))} ({ph.max() / np.median(ph):.2f}x the median phase); phases 24 / 49: {ph[24 % per] / np.median(ph):.2f} / {ph[49 % per] / np.median(ph):.2f}")


if __name__ == "__main__":
    main()
