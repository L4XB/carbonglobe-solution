#!/usr/bin/env python
"""Linear twin of ``state_late_gbdt.py``: ridge regression on state-only features over late windows.

Trees extrapolate constantly beyond the training range; the competition's mid/old stands sit outside it in
10-22% of rows (tall-thin tail, off-allometry pools). A ridge model on smooth transforms of the same
state features extrapolates linearly, so its change pattern on those rows differs from every GBDT's and
from the latent models'. Features per (site, age): the four scaled pools, their logs, the three ratios,
the seed-age one-hot, pools x age one-hot interactions, and the site's seed-age profile (ages 10-500).
Windows: 30 years from ``--starts30`` then 10 years from ``--starts10``, all four pools predicted.

    ./.venv/bin/python scripts/state_late_ridge.py --tag stateridge1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgb import cache, holdout, io, shift  # noqa: E402

DECODE = np.array([1.14114, 1.14114, 1.14114, 1.0])
POOLS = 4


def feats(u: np.ndarray) -> np.ndarray:
    """``u`` (n, 15, 4) scaled pools -> (15 n, F) age-major design."""
    n = u.shape[0]
    prof = u[:, 1:, :].reshape(n, -1)  # profile without age 1
    rows = []
    for j in range(15):
        s = u[:, j, :]
        lg = np.log(s + 0.05)
        ratios = np.stack([s[:, 1] / (s[:, 0] + 0.05), s[:, 3] / (s[:, 1] + 0.05), s[:, 2] / (s[:, 1] + 0.05)], axis=1)
        oh = np.zeros((n, 15)); oh[:, j] = 1.0
        inter = (s[:, :, None] * oh[:, None, :]).reshape(n, -1)
        rows.append(np.concatenate([s, lg, ratios, oh, inter, prof], axis=1))
    return np.concatenate(rows, axis=0)


def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    A = Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1])
    W = np.linalg.solve(A, Z.T @ (Y - Y.mean(0)))
    return W, mu, sd, Y.mean(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="stateridge1")
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--starts30", default="10")
    ap.add_argument("--starts10", default="20,25,30")
    args = ap.parse_args()
    y_mean = io.load_stats()["y_mean"].astype(np.float64)
    split = holdout.make_split(seed=42)
    traj = cache.open_traj()
    u_all = np.transpose(np.asarray(traj[:, :, :, :POOLS], dtype=np.float32) / y_mean[:POOLS].astype(np.float32), (1, 0, 2, 3))
    n_sites = u_all.shape[0]
    fit_sites = split.fit[::2]  # half the fit sites keep the normal equations cheap
    models = {}
    for horizon, starts in ((30, [int(s) for s in args.starts30.split(",")]), (10, [int(s) for s in args.starts10.split(",")])):
        Xs, Ys = [], []
        for s0 in starts:
            u = u_all[fit_sites][:, :, s0, :].astype(np.float64)
            Xs.append(feats(u))
            Ys.append(np.concatenate([u_all[fit_sites][:, j, s0 + horizon, :] - u_all[fit_sites][:, j, s0, :] for j in range(15)], axis=0))
        X = np.concatenate(Xs); Y = np.concatenate(Ys).astype(np.float64)
        models[horizon] = fit_ridge(X, Y, args.lam)
        print(f"horizon {horizon}: {X.shape[0]:,} rows, {X.shape[1]} features", flush=True)

    def step(u: np.ndarray, horizon: int) -> np.ndarray:
        W, mu, sd, ym = models[horizon]
        n = u.shape[0]
        X = feats(u)
        ch = ((X - mu) / sd) @ W + ym
        return np.clip(np.concatenate([u[:, j, :] for j in range(15)], axis=0) + ch, 0, None).reshape(15, n, POOLS).transpose(1, 0, 2)

    ages = np.array(io.SEED_AGES)
    sites = split.public
    u10 = u_all[sites][:, :, 10, :].astype(np.float64)
    p40 = step(u10, 30)[:, :, :2]
    true = u_all[sites][:, :, 40, :2]
    se = (p40 - true) ** 2; sp = (u10[:, :, :2] - true) ** 2
    old = ages >= 150
    print(f"  public from slot 10 (30-year model): score {se.mean():.4f} old {se[:, old].mean():.4f} | persistence {sp.mean():.4f} old {sp[:, old].mean():.4f} | old explained {1 - se[:, old].mean() / sp[:, old].mean():.3f}", flush=True)
    template = np.loadtxt(io.CACHE / "sample_submission.csv", delimiter=",", skiprows=1, dtype=str, usecols=0)
    ids, hs, as_ = [], [], []
    for scenario in io.SCENARIOS:
        y0 = io.open_test(scenario)[1].astype(np.float64)[:, :, :POOLS]
        u0t = DECODE * (y0 - shift.FLOORS[:POOLS]) / y_mean[:POOLS]
        p40 = step(step(u0t, 30), 10)[:, :, :2]
        p40[u0t[:, :, 0] * y_mean[0] < 1.0] = 0.0
        n = u0t.shape[0]
        for site in range(n):
            for age in io.SEED_AGES:
                ids.append(f"{scenario}_{site}_{age}")
        hs.append(p40[..., 0].reshape(-1)); as_.append(p40[..., 1].reshape(-1))
        for lab, sel in (("young", ages <= 30), ("mid", (ages >= 50) & (ages <= 100)), ("old", ages >= 150)):
            print(f"  {scenario} {lab}: mean change h {np.mean(p40[:, sel, 0] - u0t[:, sel, 0]):+.4f} a {np.mean(p40[:, sel, 1] - u0t[:, sel, 1]):+.4f}", flush=True)
    assert np.array_equal(np.array(ids), template)
    pred = np.stack([np.concatenate(hs), np.concatenate(as_)], 1)
    out = io.ROOT / "results" / f"direct_{args.tag}_m114.csv"
    shift.write_predictions(pred, out); print(f"wrote {out}; nothing submitted")


if __name__ == "__main__":
    main()
