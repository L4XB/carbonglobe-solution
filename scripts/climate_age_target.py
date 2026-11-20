#!/usr/bin/env python
"""Climate-and-age analog: the training's year-40 state predicted from site climatology, soil and seed age only.

No initial state enters. On the test this says where a training-like stand of that age and climate would
be after forty years; its difference from the decoded persistence is a direction for the mid/old height
block, where the test's initial heights sit above every training relation and the truth relaxes.

    <lightgbm venv>/bin/python scripts/climate_age_target.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgb import cache, holdout, io, latent, shift  # noqa: E402


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, default=40, help="training year slot whose state is predicted (40 = year 40)")
    args = ap.parse_args()
    split = holdout.make_split(seed=42)
    y_mean = io.load_stats()["y_mean"].astype(np.float64)
    xa = cache.open_x_annual()
    site_x = np.concatenate([latent.seasonal_block(cache.open_x_season()), np.asarray(xa[:, 0, [129, 130, 131, 132, 133, 134, 135]])], 1).astype(np.float32)
    traj = cache.open_traj()
    n = site_x.shape[0]
    ages = np.array(io.SEED_AGES, dtype=np.float32)
    X = np.concatenate([np.concatenate([site_x, np.full((n, 1), a, np.float32)], 1) for a in ages], 0)
    Y = np.concatenate([np.asarray(traj[j, :, args.slot, :2], dtype=np.float64) / y_mean[:2] for j in range(15)], 0)
    site_of_row = np.tile(np.arange(n), 15)
    fit = np.isin(site_of_row, split.fit)
    params = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=42)
    boosters = [lgb.train(params, lgb.Dataset(X[fit], label=Y[fit, k]), num_boost_round=800) for k in range(2)]
    pub = np.isin(site_of_row, split.public)
    p = np.stack([b.predict(X[pub]) for b in boosters], 1)
    print(f"holdout public: MSE of the climate-age year-40 state {np.mean((p - Y[pub]) ** 2):.4f} (height {np.mean((p[:, 0] - Y[pub, 0]) ** 2):.4f}, agb {np.mean((p[:, 1] - Y[pub, 1]) ** 2):.4f})")
    keys = shift.template_keys()
    out = np.zeros((len(keys), 2))
    for scenario in io.SCENARIOS:
        sx = np.concatenate([latent.seasonal_block(np.load(cache.CACHE_DIR / f"test_{scenario}_season.npy")),
                             np.asarray(np.load(cache.CACHE_DIR / f"test_{scenario}_annual.npy", mmap_mode="r")[:, 0, [129, 130, 131, 132, 133, 134, 135]])], 1).astype(np.float32)
        m = keys["scenario"].to_numpy() == scenario
        site = keys["site"].to_numpy()[m]; age = keys["age"].to_numpy()[m].astype(np.float32)
        Xt = np.concatenate([sx[site], age[:, None]], 1)
        out[m] = np.stack([b.predict(Xt) for b in boosters], 1)
    tag = "y40" if args.slot == 40 else f"s{args.slot}"
    np.save(io.ROOT / "work" / "cache" / f"climate_age_{tag}_test_scaled.npy", out)
    if args.slot != 40:
        print(f"wrote work/cache/climate_age_{tag}_test_scaled.npy"); return
    R = shift.read_predictions(io.ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    masks = shift.band_masks(keys["age"].to_numpy())
    for b in ("young", "mid", "old"):
        mm = masks[b]
        print(f"test {b}: climate-age y40 mean h {out[mm, 0].mean():.3f} a {out[mm, 1].mean():.3f} vs R h {R[mm, 0].mean():.3f} a {R[mm, 1].mean():.3f}")
    F = R.copy()
    mo = masks["mid"] | masks["old"]
    F[mo, 0] = out[mo, 0]
    shift.write_predictions(F, io.ROOT / "results" / "climprobe_midold_height.csv")
    print("wrote results/climprobe_midold_height.csv")


if __name__ == "__main__":
    main()
