#!/usr/bin/env python
"""Test forcing caches with EVERY channel mapped back onto the training scale through the exact-twin affine fits
(scripts/geo/calib_twin.py: released = alpha_c * train_cell + beta_c over 5994 exact twins, test 2021-2030 vs training
2012-2021):  x_cal = (released - beta_c) / alpha_c.
Exact anchors replace the regression where they exist:
    pr       beta = 6.3517 (the image of 0 mm), alpha = through-origin slope of (released - 6.3517) on the twin's pr
    rsds_h   beta = the channel's night floor (work/edv3/rsds_channel_floors.json), alpha through that point
    CO2      left as delivered (--co2 asis) : the models that read it were fitted and board-calibrated on that scale
    static soil channels: the exact maps (R2 = 1)
Writes work/cache/test_<s>_annual_<tag>.npy and test_<s>_season_<tag>.npy (default tag cal) and work/cal/cal_maps.npz.
    python scripts/geo/make_cal_cache.py [--tag cal] [--only 0,1,...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CH_PR = 1
RSDS = 35 + 4 * np.arange(24)
CO2 = np.arange(8, 32)
PR_FLOOR = 6.3517


def maps() -> tuple[np.ndarray, np.ndarray]:
    cal = np.load(ROOT / "work/cal/calib_twin.npz")
    alpha, beta = cal["alpha"].copy(), cal["beta"].copy()
    cte, ctr = cal["clim_test"].astype(np.float64), cal["clim_train"].astype(np.float64)
    # pr through the exact floor
    x = ctr[:, :, CH_PR].ravel(); y = cte[:, :, CH_PR].ravel() - PR_FLOOR
    alpha[CH_PR] = float(np.sum(x * y) / np.sum(x * x)); beta[CH_PR] = PR_FLOOR
    floors = np.asarray(json.load(open(ROOT / "work/edv3/rsds_channel_floors.json"))["floors"], dtype=np.float64)
    for h, c in enumerate(RSDS):
        x = ctr[:, :, c].ravel(); y = cte[:, :, c].ravel() - floors[h]
        alpha[c] = float(np.sum(x * y) / np.sum(x * x)); beta[c] = floors[h]
    alpha[CO2] = 1.0; beta[CO2] = 0.0
    alpha[32] = 1.0; beta[32] = 0.0
    return alpha, beta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cal")
    ap.add_argument("--only", default="", help="comma list of channel indices to repair (others as delivered); groups: monthly, soil, hus, ta_h, rsds, wind")
    args = ap.parse_args()
    alpha, beta = maps()
    if args.only:
        groups = {"monthly": list(range(0, 8)), "soil": list(range(129, 136)), "hus": list(33 + 4 * np.arange(24)), "ta_h": list(34 + 4 * np.arange(24)),
                  "rsds": list(RSDS), "wind": list(36 + 4 * np.arange(24)), "ta_m": [0], "pr": [1], "tsl": list(range(2, 8))}
        keep = set()
        for tok in args.only.split(","):
            keep |= set(groups[tok]) if tok in groups else {int(tok)}
        for c in range(136):
            if c not in keep:
                alpha[c], beta[c] = 1.0, 0.0
    np.savez(ROOT / f"work/cal/cal_maps_{args.tag}.npz", alpha=alpha, beta=beta)
    np.set_printoptions(linewidth=200, precision=4, suppress=True)
    print("monthly alpha", alpha[:8], "beta", beta[:8]); print("rsds alpha", alpha[RSDS]); print("rsds beta", beta[RSDS])
    for s in ("ssp126", "ssp585"):
        ann = np.load(ROOT / f"work/cache/test_{s}_annual.npy").astype(np.float64)      # (n, 40, 136)
        sea = np.load(ROOT / f"work/cache/test_{s}_season.npy").astype(np.float64)      # (n, 136, 12)
        ann_c = (ann - beta[None, None, :]) / alpha[None, None, :]
        sea_c = (sea - beta[None, :, None]) / alpha[None, :, None]
        np.save(ROOT / f"work/cache/test_{s}_annual_{args.tag}.npy", ann_c.astype(np.float32)); np.save(ROOT / f"work/cache/test_{s}_season_{args.tag}.npy", sea_c.astype(np.float32))
        print(f"{s}: ta_m mean {ann[..., 0].mean():.2f} -> {ann_c[..., 0].mean():.2f}; pr {ann[..., 1].mean():.2f} -> {ann_c[..., 1].mean():.2f}; "
              f"hus daily {ann[..., 33::4][..., :24].mean():.5f} -> {ann_c[..., 33::4][..., :24].mean():.5f}; rsds daily {ann[..., RSDS].mean():.1f} -> {ann_c[..., RSDS].mean():.1f}; "
              f"min pr {ann_c[..., 1].min():.2f}, min rsds {ann_c[..., RSDS].min():.2f}")


if __name__ == "__main__":
    main()
