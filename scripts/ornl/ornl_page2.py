#!/usr/bin/env python
"""Extract single bands of ORNL DAAC 2522 GeoTIFFs at overview page 2 (0.04 deg; the overviews are nearest-neighbour
subsamples, so each page-2 pixel is one genuine 0.01-deg pixel and (2021, 2060) pairs stay per-pixel pairs).
Writes work/ornl2522/p2_<file stem>_b<band>.npy (int16, nodata -32768, x 0.01 = kg C m-2), shape (3500, 9000);
pixel (r, c) centre: lat 80 - 0.04 r - 0.005, lon -180 + 0.04 c + 0.005.
    <scratchpad>/venvs/geotiff/bin/python scripts/ornl/ornl_page2.py stem:band [stem:band ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile

D = Path(__file__).resolve().parents[2] / "work" / "ornl2522"

for arg in sys.argv[1:]:
    stem, band = arg.rsplit(":", 1)
    out = D / f"p2_{stem}_b{band}.npy"
    if out.exists():
        print("have", out.name); continue
    with tifffile.TiffFile(D / f"{stem}.tif") as tif:
        a = tif.pages[2].asarray()[..., int(band)].copy()
    np.save(out, a)
    print(out.name, a.shape, f"valid {np.mean(a != -32768):.3f}", flush=True)
