#!/usr/bin/env python
"""Direct GBDT from the slot-0 state and forcing summaries to the late (acclimated) rate of change.

Target: (u[40] - u[25]) * 40/15, the training stand's change over its last 15 years scaled to 40 - the dynamics after
the spin-up/run transient has died out - while the inputs are the slot-0 state, seed age, profile and the 40-year
forcing summaries of `direct_gbdt.py` (the test's states look like slot 0-10, its dynamics like late slots).

    <lgb venv>/bin/python scripts/accrate_gbdt.py --tag accrate1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cgb import cache, holdout, io, latent, shift  # noqa: E402
import direct_gbdt as dg  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="accrate1")
    ap.add_argument("--rounds", type=int, default=1200)
    ap.add_argument("--late", type=int, default=25, help="start slot of the late window")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all-slots", default="", help="comma list of window starts s (15-year windows s..s+15, start slot as a "
                    "feature); the competition is predicted at s = --late")
    args = ap.parse_args()
    stats = io.load_stats(); y_mean = stats["y_mean"].astype(np.float64); agef = latent.age_features(stats)
    split = holdout.make_split(seed=42)
    ann = cache.open_x_annual(); sea = cache.open_x_season()
    feat = np.concatenate([dg.site_features(np.asarray(ann[lo:lo + 4000]), np.asarray(sea[lo:lo + 4000])) for lo in range(0, ann.shape[0], 4000)])
    traj = cache.open_traj()
    u0 = np.transpose(np.asarray(traj[:, :, 0, :4], dtype=np.float64) / y_mean[:4], (1, 0, 2))
    X = dg.rows(feat, u0, agef)
    n = u0.shape[0]; site_of_row = np.tile(np.arange(n), 15)
    if args.all_slots:
        starts = [int(v) for v in args.all_slots.split(",")]
        Xs, ts = [], []
        for s0 in starts:
            Xs.append(np.concatenate([X, np.full((X.shape[0], 1), s0, np.float32)], 1))
            ts.append(np.concatenate([(np.asarray(traj[j, :, s0 + 15, :2], dtype=np.float64) - np.asarray(traj[j, :, s0, :2], dtype=np.float64)) / y_mean[:2] * (40 / 15) for j in range(15)]))
        X = np.concatenate(Xs); tgt = np.concatenate(ts); del Xs, ts
        site_of_row = np.tile(site_of_row, len(starts))
        slot_of_row = np.repeat(starts, 15 * n)
    else:
        lw = 40 - args.late
        tgt = np.concatenate([(np.asarray(traj[j, :, 40, :2], dtype=np.float64) - np.asarray(traj[j, :, args.late, :2], dtype=np.float64)) / y_mean[:2] * (40 / lw) for j in range(15)])
        slot_of_row = np.full(len(tgt), args.late)
    fit = np.isin(site_of_row, split.fit)
    params = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=args.seed)
    boosters = [lgb.train(params, lgb.Dataset(X[fit], label=tgt[fit, k]), num_boost_round=args.rounds) for k in range(2)]
    ages = np.array(io.SEED_AGES); age_of_row = np.tile(np.repeat(ages, n), len(tgt) // (15 * n))
    ho = ~fit & (slot_of_row == args.late)
    pred = np.stack([b.predict(X[ho], num_threads=8) for b in boosters], 1)
    for b, m in (("young", age_of_row[ho] <= 30), ("old", age_of_row[ho] >= 150)):
        r2 = [1 - np.mean((pred[m, k] - tgt[ho][m, k]) ** 2) / np.var(tgt[ho][m, k]) for k in range(2)]
        print(f"holdout {b}: R2 of the late-rate target h {r2[0]:.3f} a {r2[1]:.3f}; mean target a {tgt[ho][m, 1].mean():+.4f}", flush=True)
    keys = shift.template_keys(); out = np.zeros((len(keys), 2))
    for s in io.SCENARIOS:
        ann_t = np.load(cache.CACHE_DIR / f"test_{s}_annual.npy"); sea_t = np.load(cache.CACHE_DIR / f"test_{s}_season.npy")
        y0 = io.open_test(s)[1].astype(np.float64)[:, :, :4]
        u0t = dg.DECODE * (y0 - dg.FLOORS) / y_mean[:4]
        Xt = dg.rows(dg.site_features(ann_t, sea_t), u0t, agef)
        u0c = Xt[:, -19:-15]
        if args.all_slots:
            Xt = np.concatenate([Xt, np.full((Xt.shape[0], 1), args.late, np.float32)], 1)
        ch = np.stack([b.predict(Xt, num_threads=8) for b in boosters], 1)
        p = np.clip(u0c[:, :2] + ch, 0, None); p[u0c[:, 0] * y_mean[0] < 1.0] = 0.0
        p = p.reshape(15, -1, 2).transpose(1, 0, 2)
        m = keys["scenario"].to_numpy() == s
        out[m] = p[keys["site"].to_numpy()[m], np.searchsorted(ages, keys["age"].to_numpy()[m])]
    shift.write_predictions(out, ROOT / "results" / f"direct_{args.tag}_m114.csv")
    print(f"wrote results/direct_{args.tag}_m114.csv", flush=True)


if __name__ == "__main__":
    main()
