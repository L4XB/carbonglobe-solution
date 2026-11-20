#!/usr/bin/env python
"""Test forcing caches built from the exact twin's TRAINING forcing (x_annual / x_season of the matched training cell):
"what would the models predict for the test's start states under the cell's MERRA-2 1982-2021 forcing".
    --groups all          every channel from the twin (tag twin)
    --groups monthly      only the monthly drivers 0-7 (ta_m, pr, tsl1-6) from the twin, the rest from --base-tag (tag twinm)
Sites without an exact twin keep the base cache. Writes work/cache/test_<s>_annual_<tag>.npy / _season_<tag>.npy.
    python scripts/geo/make_twin_cache.py --groups all --tag twin
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--groups", default="all", choices=["all", "monthly", "hourly"])
ap.add_argument("--tag", default="twin")
ap.add_argument("--base-tag", default="", help="cache tag for the channels / sites that stay (default: as delivered)")
args = ap.parse_args()
ex = np.load(ROOT / "work/test_cell_exact.npz"); cell = ex["cell"]; ok = cell >= 0
xa = np.load(ROOT / "work/cache/x_annual.npy", mmap_mode="r"); xs = np.load(ROOT / "work/cache/x_season.npy", mmap_mode="r")
order = np.argsort(cell[ok]); inv = np.empty_like(order); inv[order] = np.arange(len(order))
ta = np.asarray(xa[cell[ok][order]])[inv]; ts = np.asarray(xs[cell[ok][order]])[inv]                    # (n_ok, 40, 136), (n_ok, 136, 12)
ch = {"all": np.arange(136), "monthly": np.arange(0, 8), "hourly": np.arange(33, 129)}[args.groups]
sfx = f"_{args.base_tag}" if args.base_tag else ""
for s in ("ssp126", "ssp585"):
    ann = np.load(ROOT / f"work/cache/test_{s}_annual{sfx}.npy").copy(); sea = np.load(ROOT / f"work/cache/test_{s}_season{sfx}.npy").copy()
    a0 = ann[ok][:, :, 0].mean(); 
    sub = ann[ok]; sub[:, :, ch] = ta[:, :, ch]; ann[ok] = sub
    sub = sea[ok]; sub[:, ch, :] = ts[:, ch, :]; sea[ok] = sub
    np.save(ROOT / f"work/cache/test_{s}_annual_{args.tag}.npy", ann); np.save(ROOT / f"work/cache/test_{s}_season_{args.tag}.npy", sea)
    print(f"{s}: tag {args.tag} ({args.groups}; base '{args.base_tag or 'as delivered'}'): matched sites {int(ok.sum())}; ta_m mean {a0:.2f} -> {ann[ok][:, :, 0].mean():.2f}; pr {np.load(ROOT / f'work/cache/test_{s}_annual{sfx}.npy')[ok][:, :, 1].mean():.2f} -> {ann[ok][:, :, 1].mean():.2f}")
