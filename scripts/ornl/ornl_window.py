#!/usr/bin/env python
"""Read a lat/lon window of an ORNL DAAC 2522 GeoTIFF at full resolution (0.01 deg) by decoding only the tiles it
touches. Returns int16 -> kg C m-2 (x 0.01), nodata -> nan, shape (rows, cols, bands).
Needs tifffile + imagecodecs (scratchpad venv geotiff)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def read_window(path: Path, lat_hi: float, lat_lo: float, lon_lo: float, lon_hi: float, bands: list[int] | None = None,
                page: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with tifffile.TiffFile(path) as tif:
        p = tif.pages[page]
        H, W, B = p.shape
        res = 0.01 * (14000 / H)
        r0 = max(0, int(np.floor((80.0 - lat_hi) / res))); r1 = min(H, int(np.ceil((80.0 - lat_lo) / res)))
        c0 = max(0, int(np.floor((lon_lo + 180.0) / res))); c1 = min(W, int(np.ceil((lon_hi + 180.0) / res)))
        tl, tw = p.tilelength, p.tilewidth
        ntc = (W + tw - 1) // tw
        out = np.full((r1 - r0, c1 - c0, B), -32768, np.int16)
        fh = tif.filehandle
        decode = p.decode
        for tr in range(r0 // tl, (r1 - 1) // tl + 1):
            for tc in range(c0 // tw, (c1 - 1) // tw + 1):
                idx = tr * ntc + tc
                fh.seek(p.dataoffsets[idx]); data = fh.read(p.databytecounts[idx])
                seg, _, _ = decode(data, idx)
                seg = np.asarray(seg).reshape(tl, tw, B)
                ra, rb = max(r0, tr * tl), min(r1, (tr + 1) * tl); ca, cb = max(c0, tc * tw), min(c1, (tc + 1) * tw)
                out[ra - r0:rb - r0, ca - c0:cb - c0] = seg[ra - tr * tl:rb - tr * tl, ca - tc * tw:cb - tc * tw]
    v = np.where(out == -32768, np.nan, out.astype(np.float64) * 0.01)
    if bands is not None:
        v = v[..., bands]
    lat = 80.0 - (np.arange(r0, r1) + 0.5) * res; lon = -180.0 + (np.arange(c0, c1) + 0.5) * res
    return lat, lon, v
