#!/usr/bin/env python
"""State-only direct GBDT: the forty-year change predicted from the initial state and the seed age alone.

No forcing, no site climate, no soil: features are the decoded initial pools (height, agb, soil, lai),
their ratios, and the age block. On the training holdout this captures the spin-up relaxation part of
the change (the shuffled-forcing latent explained ~55% of the old-band variance). Its old-band pattern
is a candidate direction for the board, where every climate-driven model shares one pattern that
transfers at ~50%.

    <lgb venv>/bin/python scripts/state_only_gbdt.py --tag state1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgb import cache, holdout, io, latent, shift  # noqa: E402

DECODE = np.array([1.14114, 1.14114, 1.14114, 1.0])


PROFILE_SKIP: tuple[int, ...] = ()  # seed-age indices left out of the profile (the competition's age-1 states are anomalous)


def feats(u0: np.ndarray, agef: np.ndarray, with_profile: bool) -> np.ndarray:
    """``u0`` (n, 15, 4) scaled decoded pools -> (15*n, F) age-major."""
    n = u0.shape[0]
    rows = []
    keep = [j for j in range(15) if j not in PROFILE_SKIP]
    prof = u0[:, keep, :].reshape(n, -1) if with_profile else np.zeros((n, 0))
    for j in range(15):
        s = u0[:, j, :]
        ratios = np.stack([s[:, 1] / (s[:, 0] + 0.05), s[:, 3] / (s[:, 1] + 0.05), s[:, 2] / (s[:, 1] + 0.05)], axis=1)
        rows.append(np.concatenate([s, ratios, np.repeat(agef[j][None], n, axis=0), prof], axis=1))
    return np.concatenate(rows, axis=0).astype(np.float32)


def anom_summ(a: np.ndarray) -> np.ndarray:
    """(n, 32): per monthly driver 0..7 the 40-year interannual sd, trend, max - mean, mean - min (precip in log1p)."""
    out = []
    yrs = np.arange(a.shape[1]) - (a.shape[1] - 1) / 2
    for j in range(8):
        x = np.asarray(a[:, :, j], dtype=np.float64)
        if j == 1:
            x = np.log1p(np.clip(x, 0, None))
        mu = x.mean(1)
        out += [x.std(1), (x * yrs).sum(1) / (yrs ** 2).sum(), x.max(1) - mu, mu - x.min(1)]
    return np.stack(out, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="state1")
    ap.add_argument("--profile", action="store_true", help="add the site's full seed-age profile of initial states (60 features)")
    ap.add_argument("--rounds", type=int, default=1200)
    ap.add_argument("--skip-age1", action="store_true", help="leave the age-1 states out of the profile")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--fluxes", action="store_true", help="append the stand's own year-0 gpp, npp, rh (scaled) to its features")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--anom", action="store_true",
                    help="append site-relative 40-year weather anomaly summaries (sd, trend, max-mean, mean-min of the 8 monthly drivers)")
    args = ap.parse_args()
    global PROFILE_SKIP
    PROFILE_SKIP = (0,) if args.skip_age1 else ()
    stats = io.load_stats(); y_mean = stats["y_mean"].astype(np.float64); agef = latent.age_features(stats)
    split = holdout.make_split(seed=42)
    traj = cache.open_traj()
    u0 = np.transpose(np.asarray(traj[:, :, 0, :4], dtype=np.float64) / y_mean[:4], (1, 0, 2))
    y40 = np.transpose(np.asarray(traj[:, :, 40, :2], dtype=np.float64) / y_mean[:2], (1, 0, 2))
    X = feats(u0, agef, args.profile)
    if args.fluxes:
        f0 = np.transpose(np.asarray(traj[:, :, 0, 4:7], dtype=np.float64) / y_mean[4:7], (1, 0, 2))  # (n, 15, 3)
        X = np.concatenate([X, np.concatenate([f0[:, j, :] for j in range(15)], axis=0).astype(np.float32)], axis=1)
    if args.anom:
        ann = cache.open_x_annual()
        A = np.concatenate([anom_summ(np.asarray(ann[lo:lo + 6000])) for lo in range(0, ann.shape[0], 6000)])
        X = np.concatenate([X, np.tile(A, (15, 1)).astype(np.float32)], axis=1)
    y = np.concatenate([(y40 - u0[:, :, :2])[:, j, :] for j in range(15)], axis=0)
    site_of_row = np.tile(np.arange(u0.shape[0]), 15)
    fit = np.isin(site_of_row, split.fit)
    params = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=args.threads, verbose=-1, seed=args.seed)
    boosters = [lgb.train(params, lgb.Dataset(X[fit], label=y[fit, k]), num_boost_round=args.rounds) for k in range(2)]
    ages = np.array(io.SEED_AGES)
    for fold, sites in (("public", split.public), ("private", split.private)):
        m = np.isin(site_of_row, sites)
        ch = np.stack([b.predict(X[m], num_threads=args.threads) for b in boosters], axis=1)
        p = np.clip(X[m][:, :2] + ch, 0, None); p[X[m][:, 0] * y_mean[0] < 1.0] = 0.0
        t = X[m][:, :2] + y[m]
        se = (p - t) ** 2; sp = (X[m][:, :2] - t) ** 2
        age_row = np.repeat(ages, len(sites))
        old = age_row >= 150
        print(f"  {fold}: score {se.mean():.4f} (old {se[old].mean():.4f}) | persistence {sp.mean():.4f} (old {sp[old].mean():.4f}) | old explained {1 - se[old].mean()/sp[old].mean():.3f}", flush=True)
    template = np.loadtxt(io.CACHE / "sample_submission.csv", delimiter=",", skiprows=1, dtype=str, usecols=0)
    ids, hs, as_ = [], [], []
    for scenario in io.SCENARIOS:
        y0 = io.open_test(scenario)[1].astype(np.float64)[:, :, :4]
        u0t = DECODE * (y0 - shift.FLOORS[:4]) / y_mean[:4]
        Xt = feats(u0t, agef, args.profile)
        if args.anom:
            At = anom_summ(np.load(cache.CACHE_DIR / f"test_{scenario}_annual.npy"))
            Xt = np.concatenate([Xt, np.tile(At, (15, 1)).astype(np.float32)], axis=1)
        if args.fluxes:
            yf = io.open_test(scenario)[1].astype(np.float64)[:, :, 4:7]
            f0t = 1.14114 * (yf - np.array([0.259026, 0.127815, 0.084656])) / y_mean[4:7]
            Xt = np.concatenate([Xt, np.concatenate([f0t[:, j, :] for j in range(15)], axis=0).astype(np.float32)], axis=1)
        ch = np.stack([b.predict(Xt, num_threads=args.threads) for b in boosters], axis=1)
        n = u0t.shape[0]
        p40 = np.clip(Xt[:, :2] + ch, 0, None); p40[Xt[:, 0] * y_mean[0] < 1.0] = 0.0
        p40 = p40.reshape(15, n, 2).transpose(1, 0, 2)
        for site in range(n):
            for age in io.SEED_AGES:
                ids.append(f"{scenario}_{site}_{age}")
        hs.append(p40[..., 0].reshape(-1)); as_.append(p40[..., 1].reshape(-1))
    assert np.array_equal(np.array(ids), template)
    pred = np.stack([np.concatenate(hs), np.concatenate(as_)], 1)
    out = io.ROOT / "results" / f"direct_{args.tag}_m114.csv"
    shift.write_predictions(pred, out); print(f"wrote {out}")


if __name__ == "__main__":
    main()
