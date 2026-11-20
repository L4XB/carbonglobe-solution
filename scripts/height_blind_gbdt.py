#!/usr/bin/env python
"""Height-blind GBDT: the year-40 height from climate, seed age and the other initial pools, never the initial height.

The competition's mid/old initial heights sit above every training relation (taller for their agb, their
climate and their age) and the truth relaxes them partially (board: allometric relaxation lambda 0.44,
old height slope 0.891). A model that cannot see the anomalous initial height predicts where a
training-like stand with that agb, soil, lai, age and climate stands after forty years; its difference from
the decoded persistence is a direction for the mid/old height block that keeps the forty-year growth
dynamics the static allometry lacks. Features are ``direct_gbdt``'s site features plus the initial
agb/soil/lai and the age block; target the year-40 height level.

    <lightgbm venv>/bin/python scripts/height_blind_gbdt.py --tag hblind1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cgb import cache, holdout, io, latent, shift  # noqa: E402
import direct_gbdt  # noqa: E402

DECODE = np.array([1.14114, 1.14114, 1.14114, 1.0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="hblind1")
    ap.add_argument("--rounds", type=int, default=1200)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    direct_gbdt.USE_HOURLY = False
    stats = io.load_stats(); y_mean = stats["y_mean"].astype(np.float64); agef = latent.age_features(stats)
    split = holdout.make_split(seed=42)
    ann = cache.open_x_annual(); sea = cache.open_x_season(); traj = cache.open_traj()
    feat = np.concatenate([direct_gbdt.site_features(np.asarray(ann[lo:lo + 4000]), np.asarray(sea[lo:lo + 4000])) for lo in range(0, ann.shape[0], 4000)])
    u0 = np.transpose(np.asarray(traj[:, :, 0, :4], dtype=np.float64) / y_mean[:4], (1, 0, 2))
    h40 = np.transpose(np.asarray(traj[:, :, 40, 0], dtype=np.float64) / y_mean[0], (1, 0))  # (n, 15)
    u0b = u0.copy(); u0b[:, :, 0] = 0.0  # the initial height is hidden
    X = direct_gbdt.rows(feat, u0b, agef)
    y = np.concatenate([h40[:, j] for j in range(15)])
    site_of_row = np.tile(np.arange(ann.shape[0]), 15)
    fit = np.isin(site_of_row, split.fit)
    params = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=args.threads, verbose=-1, seed=args.seed)
    b = lgb.train(params, lgb.Dataset(X[fit], label=y[fit]), num_boost_round=args.rounds)
    ages = np.array(io.SEED_AGES)
    for fold, sites in (("public", split.public), ("private", split.private)):
        m = np.isin(site_of_row, sites)
        p = b.predict(X[m], num_threads=args.threads)
        h0 = np.concatenate([u0[:, j, 0] for j in range(15)])[m]
        age_row = np.repeat(ages, ann.shape[0])[m]
        for lab, sel in (("mid", (age_row >= 50) & (age_row <= 100)), ("old", age_row >= 150)):
            t = y[m][sel]
            print(f"  {fold} {lab}: height-blind MSE {np.mean((p[sel] - t) ** 2):.4f} | persistence {np.mean((h0[sel] - t) ** 2):.4f} | corr of (pred - h0) with (truth - h0) {np.corrcoef(p[sel] - h0[sel], t - h0[sel])[0, 1]:.3f}", flush=True)
    R = shift.read_predictions(io.ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    keys = shift.template_keys()
    out = R.copy()
    for scenario in io.SCENARIOS:
        ann_t = np.load(cache.CACHE_DIR / f"test_{scenario}_annual.npy"); sea_t = np.load(cache.CACHE_DIR / f"test_{scenario}_season.npy")
        y0 = io.open_test(scenario)[1].astype(np.float64)[:, :, :4]
        u0t = DECODE * (y0 - shift.FLOORS[:4]) / y_mean[:4]
        u0t[:, :, 0] = 0.0
        Xt = direct_gbdt.rows(direct_gbdt.site_features(ann_t, sea_t), u0t, agef)
        p = b.predict(Xt, num_threads=args.threads).reshape(15, -1).T  # (n, 15)
        m = keys["scenario"].to_numpy() == scenario
        site = keys["site"].to_numpy()[m]; ai = np.searchsorted(ages, keys["age"].to_numpy()[m])
        out[m, 0] = np.clip(p[site, ai], 0, None)
    np.save(io.ROOT / "work" / "cache" / f"{args.tag}_test_height.npy", out[:, 0])
    masks = shift.band_masks(keys["age"].to_numpy())
    veg = R[:, 0] * 10.90645 >= 1.0
    for lab in ("mid", "old"):
        mm = masks[lab] & veg
        print(f"test {lab}: height-blind year-40 height / R: slope {np.sum(out[mm,0]*R[mm,0])/np.sum(R[mm,0]**2):.3f}, mean change {np.mean(out[mm,0]-R[mm,0]):+.3f}")
    F = R.copy(); mm = (masks["mid"] | masks["old"]) & veg
    d = out[mm, 0] - R[mm, 0]
    F[mm, 0] = R[mm, 0] + np.clip(d, -0.3 * R[mm, 0], 0.3 * R[mm, 0])
    shift.write_predictions(F, io.ROOT / "results" / f"hblindprobe_midold_height_{args.tag}.csv")
    print(f"wrote results/hblindprobe_midold_height_{args.tag}.csv (change clipped to +-30%); nothing submitted")


if __name__ == "__main__":
    main()
