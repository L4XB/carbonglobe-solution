#!/usr/bin/env python
"""Aggregate ORNL DAAC 2522 biomass GeoTIFFs (0.01 deg, one band per year of a decade) to a coarse grid.

For each file in work/ornl2522/*.tif: print the TIFF structure once, then for the requested bands (default: first and
last year of the decade) block-average to --deg (default 0.25 deg): mean biomass over valid pixels and the valid
(forest) fraction. Writes work/ornl2522/coarse_<file stem>.npz with lat/lon cell centres, mean[band], frac[band].
Runs with the scratchpad venv (tifffile + imagecodecs):
    <scratchpad>/venvs/geotiff/bin/python scripts/ornl/ornl_coarse.py [--deg 0.25] [--files a.tif,b.tif] [--info]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "work" / "ornl2522"


def georef(tif: tifffile.TiffFile) -> tuple[float, float, float, float]:
    """(lon0, lat0, dlon, dlat) of the pixel-corner grid from ModelTiepoint / ModelPixelScale."""
    p = tif.pages[0]
    scale = p.tags["ModelPixelScaleTag"].value
    tie = p.tags["ModelTiepointTag"].value
    return float(tie[3]), float(tie[4]), float(scale[0]), float(scale[1])


def info(path: Path) -> None:
    with tifffile.TiffFile(path) as tif:
        p = tif.pages[0]
        print(f"{path.name}: {len(tif.pages)} pages; page0 shape {p.shape} dtype {p.dtype} compression {p.compression!r} "
              f"tiled {p.is_tiled} tile {p.tilelength}x{p.tilewidth} spp {p.samplesperpixel} planar {p.planarconfig!r}")
        for k in ("GDAL_NODATA", "GDAL_METADATA", "ImageDescription"):
            if k in p.tags:
                print(f"   {k}: {str(p.tags[k].value)[:300]}")
        print("   georef (lon0, lat0, dlon, dlat):", georef(tif))
        series = tif.series[0]
        print(f"   series0 shape {series.shape} axes {series.axes}")


def coarse_band(arr: np.ndarray, nodata: float | None, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Block-average a (rows, cols) band over k x k pixel blocks: (mean over valid, valid fraction)."""
    r, c = arr.shape
    r2, c2 = r // k * k, c // k * k
    a = arr[:r2, :c2].astype(np.float32)
    valid = np.isfinite(a) & (a > 0)
    if nodata is not None:
        valid &= a != nodata
    a = np.where(valid, a, 0.0)
    s = a.reshape(r2 // k, k, c2 // k, k).sum(axis=(1, 3))
    n = valid.reshape(r2 // k, k, c2 // k, k).sum(axis=(1, 3))
    return np.where(n > 0, s / np.maximum(n, 1), np.nan), n / float(k * k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg", type=float, default=0.25)
    ap.add_argument("--files", default="")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--bands", default="first,last")
    ap.add_argument("--page", type=int, default=3, help="TIFF page (overview level) to read: 0 = 0.01 deg, k = 0.01 x 2^k deg (3 = 0.08 deg)")
    ap.add_argument("--scale", type=float, default=0.01, help="GDAL SCALE of the int16 values (kg C m-2 per count)")
    args = ap.parse_args()
    files = [D / f for f in args.files.split(",")] if args.files else sorted(D.glob("Biomass_*.tif"))
    for path in files:
        if args.info:
            info(path); continue
        out = D / f"coarse_{path.stem}.npz"
        if out.exists():
            print("have", out.name); continue
        with tifffile.TiffFile(path) as tif:
            lon0, lat0, dlon, dlat = georef(tif)
            nd = tif.pages[0].tags.get("GDAL_NODATA")
            nodata = float(nd.value) if nd is not None else None
            page = tif.pages[args.page]
            full = page.asarray()                                   # (rows, cols, bands) int16 at 0.01 x 2^page deg
            nb = full.shape[2] if full.ndim == 3 else 1
            pix = dlon * (2 ** args.page)
            want = sorted({0 if b == "first" else nb - 1 if b == "last" else int(b) for b in args.bands.split(",")})
            k = max(1, int(round(args.deg / pix)))
            dlon = dlat = pix
            means, fracs = {}, {}
            for b in want:
                raw = full[..., b] if full.ndim == 3 else full
                arr = np.where(raw == (nodata if nodata is not None else -32768), np.nan, raw.astype(np.float32) * args.scale)
                m, f = coarse_band(arr, None, k)
                means[b], fracs[b] = m, f
                print(f"{path.name} band {b}: raw {arr.shape} {arr.dtype} -> coarse {m.shape}; valid frac {np.nanmean(f):.3f}, "
                      f"mean biomass over valid {np.nanmean(m):.2f}", flush=True)
                del arr
            ny, nx = next(iter(means.values())).shape
            lat = lat0 - (np.arange(ny) + 0.5) * k * dlat
            lon = lon0 + (np.arange(nx) + 0.5) * k * dlon
            np.savez_compressed(out, lat=lat, lon=lon, bands=np.array(want), nbands=nb,
                                **{f"mean{b}": means[b] for b in want}, **{f"frac{b}": fracs[b] for b in want})
            print("wrote", out.name, flush=True)


if __name__ == "__main__":
    main()
