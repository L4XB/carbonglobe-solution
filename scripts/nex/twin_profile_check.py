#!/usr/bin/env python
"""Position check through the seed-age lookup: the test y0 profile (decoded, ages 1-500) against the TRAINING y0 profile
of the nearest training cell (mask2d coordinates) for each candidate position set. If the organisers built the test
lookup like the training one, the right neighbour has a near-identical profile.
    python scripts/nex/twin_profile_check.py work/test_latlon_v2.npy work/test_latlon_nex2.npz ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io, shift  # noqa: E402

DIV = np.array([10.90645, 3.39567])
AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]


def load_pos(p: str) -> np.ndarray:
    z = np.load(ROOT / p)
    return np.stack([z["lat"], z["lon"]], 1) if hasattr(z, "files") else z


def main() -> None:
    tll = np.load(ROOT / "work/edv3/train_latlon.npy")
    cache = ROOT / "work/train_y0_agb_h.npy"
    if cache.exists():
        ty = np.load(cache)
    else:
        ty = np.zeros((tll.shape[0], len(AGES), 2))
        for k, a in enumerate(AGES):
            y = io.open_train_y(a)
            ty[:, k, 0] = np.asarray(y[:, 0, :, 0], dtype=np.float64).mean(axis=1)
            ty[:, k, 1] = np.asarray(y[:, 0, :, 1], dtype=np.float64).mean(axis=1)
        np.save(cache, ty)
    keys = shift.template_keys(); site = keys["site"].to_numpy(); age = keys["age"].to_numpy(); scen = keys["scenario"].to_numpy()
    R = shift.read_predictions(ROOT / "results/latent_ref_y0_m114114.csv").astype(np.float64) * DIV
    n = site.max() + 1
    tp = np.full((n, len(AGES), 2), np.nan)
    m = scen == "ssp126"
    for k, a in enumerate(AGES):
        mm = m & (age == a); tp[site[mm], k, :] = R[mm]
    la2, lo2 = np.radians(tll[:, 0]), np.radians(tll[:, 1])
    for p in sys.argv[1:]:
        pos = load_pos(p)
        nb = np.full(n, -1); dist = np.full(n, np.nan)
        for i in range(n):
            if not np.all(np.isfinite(pos[i])):
                continue
            la1, lo1 = np.radians(pos[i, 0]), np.radians(pos[i, 1])
            a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
            d = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1))); j = int(np.argmin(d)); nb[i] = j; dist[i] = d[j]
        ok = nb >= 0
        res = []
        for c, cn in ((1, "agb"), (0, "h")):
            x = tp[ok][:, :, c]; y = ty[nb[ok]][:, :, c]
            g = (x > 0.05) & (y > 0.05)
            lr = np.log(x[g]) - np.log(y[g])
            res.append(f"{cn}: corr(log) {np.corrcoef(np.log(x[g]), np.log(y[g]))[0, 1]:+.3f}, median |log ratio| {np.median(np.abs(lr)):.3f}, "
                       f"share within 10% {np.mean(np.abs(lr) < 0.0953):.3f}")
        print(f"{Path(p).name:22s} n {ok.sum()} median twin dist {np.nanmedian(dist):.0f} km | " + " | ".join(res))


if __name__ == "__main__":
    main()
