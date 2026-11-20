#!/usr/bin/env python
"""Rebuild the test forcing caches (annual means and 40-year seasonal climatology, as work/cache/test_<s>_annual.npy
and test_<s>_season.npy) from test_x with the release artefacts undone, so that models trained on the clean training
forcing see training-consistent test inputs:
    pr     additive floor 6.3517 mm/month removed (clip 0)                     [11-08]
    rsds   per-hour-channel night floor removed (work/edv3/rsds_channel_floors.json, clip 0)   [11-04]
    hus    negative values clipped at 0                                        [11-04]
    ta_h   shifted per (year, month) so the 24-hour mean equals ta_m           [11-04]
CO2 is left as delivered (a separate question). --check reproduces the original caches without repairs first.
    python scripts/repair_test_cache.py [--check] [--tag rep]
writes work/cache/test_<s>_annual_<tag>.npy and test_<s>_season_<tag>.npy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io  # noqa: E402

PR_FLOOR = 6.3517
PR_NEX_SCALE = 0.865
CH_PR = 1
CH_TA_M = 0
HUS = 33 + 4 * np.arange(24)
TA_H = 34 + 4 * np.arange(24)
RSDS = 35 + 4 * np.arange(24)


# 11-09: the released test temperatures sit below NEX-GDDP-CMIP6 GFDL-ESM4 (the forcing source; test year 0 = 2021) by a
# temperature-dependent amount (regional monthly climatologies, 1432 sites): test - NEX = +0.26 K below -10 C, -2.50 at
# -10..5, -2.74 at 5..15, -3.32 at 15..25, -3.84 at 25..35 C (NEX bins). "ta_nex" adds the inverse, interpolated in the
# TEST temperature, to ta_m, the six soil-temperature channels and the 24 hourly air temperatures.
TA_KNOTS_TEST_C = np.array([-24.7, -5.0, 7.3, 16.7, 26.2])
TA_KNOTS_SHIFT = np.array([-0.26, 2.50, 2.74, 3.32, 3.84])
TSL = [2, 3, 4, 5, 6, 7]


def ta_shift(t_k: np.ndarray) -> np.ndarray:
    return np.interp(t_k - 273.15, TA_KNOTS_TEST_C, TA_KNOTS_SHIFT)


def repair(x: np.ndarray, floors: np.ndarray, parts: set[str]) -> np.ndarray:
    """x: (n, 40, 12, 136) float64 -> repaired copy."""
    x = x.copy()
    if "ta_nex" in parts:
        for ch in [CH_TA_M] + TSL + list(TA_H):
            x[..., ch] = x[..., ch] + ta_shift(x[..., ch])
    if "hus_nex" in parts:   # 11-09: released specific humidity ~0.767 x NEX huss (regional 2021-2022 means)
        x[..., HUS] = np.clip(x[..., HUS], 0.0, None) / 0.767
    if "pr" in parts:
        x[..., CH_PR] = np.clip(x[..., CH_PR] - PR_FLOOR, 0.0, None)
    if "pr_nex" in parts:   # 11-09: (released pr - floor) ~ 0.865 x NEX pr per site (nex3 positions, wet months; IQR 0.83-0.90)
        x[..., CH_PR] = np.clip(x[..., CH_PR] - PR_FLOOR, 0.0, None) / PR_NEX_SCALE
    if "rsds" in parts:
        x[..., RSDS] = np.clip(x[..., RSDS] - floors[None, None, None, :], 0.0, None)
    if "hus" in parts:
        x[..., HUS] = np.clip(x[..., HUS], 0.0, None)
    if "ta_h" in parts:
        shift = x[..., CH_TA_M] - x[..., TA_H].mean(axis=-1)          # (n, 40, 12)
        x[..., TA_H] = x[..., TA_H] + shift[..., None]
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--tag", default="rep")
    ap.add_argument("--parts", default="pr,rsds,hus,ta_h")
    args = ap.parse_args()
    floors = np.asarray(json.load(open(ROOT / "work/edv3/rsds_channel_floors.json"))["floors"], dtype=np.float64)
    parts = set(args.parts.split(",")) if args.parts else set()
    for s in io.SCENARIOS:
        xt, _ = io.open_test(s)
        n = xt.shape[0]
        ann = np.empty((n, 40, 136), np.float32); sea = np.empty((n, 136, 12), np.float32)
        for lo in range(0, n, 512):
            blk = np.asarray(xt[lo:lo + 512], dtype=np.float64)
            if not args.check:
                blk = repair(blk, floors, parts)
            ann[lo:lo + 512] = blk.mean(axis=2)
            sea[lo:lo + 512] = blk.mean(axis=1).transpose(0, 2, 1)
        if args.check:
            a0 = np.load(ROOT / f"work/cache/test_{s}_annual.npy"); s0 = np.load(ROOT / f"work/cache/test_{s}_season.npy")
            print(f"{s}: annual max |diff| {np.abs(ann - a0).max():.2e}, season max |diff| {np.abs(sea - s0).max():.2e}")
            continue
        np.save(ROOT / f"work/cache/test_{s}_annual_{args.tag}.npy", ann); np.save(ROOT / f"work/cache/test_{s}_season_{args.tag}.npy", sea)
        a0 = np.load(ROOT / f"work/cache/test_{s}_annual.npy")
        print(f"{s}: wrote test_{s}_annual_{args.tag} / season; mean change pr {np.mean(ann[..., CH_PR] - a0[..., CH_PR]):+.3f}, "
              f"rsds daily mean {np.mean(ann[..., RSDS].mean(-1) - a0[..., RSDS].mean(-1)):+.3f}, hus {np.mean(ann[..., HUS].mean(-1) - a0[..., HUS].mean(-1)):+.2e}")


if __name__ == "__main__":
    main()
