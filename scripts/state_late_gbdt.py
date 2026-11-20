#!/usr/bin/env python
"""State-only windowed GBDT: the two best-transferring ideas of the board in one model.

The board says (``notes/board-solve.md``): the state-only GBDT (pools, ratios, age block and the site's
seed-age profile, no forcing and no climate) transfers best on old-band agb (lambda 0.83), and the
late-dynamics windowed GBDT (windows from slot 10, the test stands look acclimated) second (0.70).
Here the state-only features are fitted on late windows: a 30-year model on windows s0 -> s0+30 for
s0 in ``--starts30`` (default 10) and a 10-year model on ``--starts10`` (default 20, 25, 30), each
predicting all four pools so the second step has its inputs. The profile at the window start is the
site's states of all seed ages at that slot. Competition rollout: y0 --30y--> --10y--> year 40.

    <lightgbm venv>/bin/python scripts/state_late_gbdt.py --tag statelate1
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
import state_only_gbdt as so  # noqa: E402

DECODE = np.array([1.14114, 1.14114, 1.14114, 1.0])
POOLS = 4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="statelate1")
    ap.add_argument("--rounds", type=int, default=900)
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--starts30", default="10")
    ap.add_argument("--starts10", default="20,25,30")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--clim", action="store_true", help="append the site's seasonal ta/pr climatology and soil (31 features)")
    ap.add_argument("--slot-feature", type=float, default=-1.0,
                    help=">= 0: append the window's start slot as a feature (so all starts can be used) and predict the "
                         "competition with this slot value (e.g. 25 = post-transient dynamics)")
    ap.add_argument("--chain10", action="store_true",
                    help="no 30-year model: the competition rollout is four chained steps of the 10-year model (acclimated late "
                         "dynamics only; use late --starts10 such as 25,28,30)")
    ap.add_argument("--no-event", action="store_true",
                    help="drop (site, age) rows whose agb has a mortality event (single-year drop > 1 kgC and > 10 %%) inside the window")
    ap.add_argument("--anom", action="store_true",
                    help="add site-relative weather anomaly summaries of the window (sd, trend, max-mean, mean-min of the "
                         "8 monthly drivers' annual series; no absolute climate means)")
    args = ap.parse_args()
    so.PROFILE_SKIP = (0,)
    with_profile = not args.no_profile
    stats = io.load_stats(); y_mean = stats["y_mean"].astype(np.float64); agef = latent.age_features(stats)
    split = holdout.make_split(seed=42)
    traj = cache.open_traj()
    u_all = np.transpose(np.asarray(traj[:, :, :, :POOLS], dtype=np.float32) / y_mean[:POOLS].astype(np.float32), (1, 0, 2, 3))  # (n, 15, 41, 4)
    n_sites = u_all.shape[0]
    site_of_row = np.tile(np.arange(n_sites), 15)
    fit_rows = np.isin(site_of_row, split.fit)
    params = dict(objective="regression", learning_rate=args.lr, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=args.threads, verbose=-1, seed=args.seed)
    ann = cache.open_x_annual() if args.anom else None

    def anom(a: np.ndarray, lo: int, hi: int) -> np.ndarray:
        """(n, 32): per driver j in 0..7 the window's interannual sd, linear trend, max - mean and mean - min."""
        out = []
        for j in range(8):
            x = np.asarray(a[:, lo:hi, j], dtype=np.float64)
            if j == 1:
                x = np.log1p(np.clip(x, 0, None))
            n_y = hi - lo
            yrs = np.arange(n_y) - (n_y - 1) / 2
            mu = x.mean(1)
            out += [x.std(1), (x * yrs).sum(1) / (yrs ** 2).sum(), x.max(1) - mu, mu - x.min(1)]
        return np.stack(out, axis=1)

    SOIL = [129, 130, 131, 132, 133, 134, 135]
    clim_cur = {"X": None}
    CL_train = None
    if args.clim:
        a0 = cache.open_x_annual()
        CL_train = np.concatenate([latent.seasonal_block(cache.open_x_season())[:, :24], np.asarray(a0[:, 0, SOIL])], 1).astype(np.float32)

    def with_anom(X: np.ndarray, A: np.ndarray) -> np.ndarray:
        X = np.concatenate([X, np.tile(A, (15, 1))], axis=1) if A is not None else X
        return np.concatenate([X, np.tile(clim_cur["X"], (15, 1))], axis=1) if args.clim else X

    models = {}
    plan = [(10, [int(s) for s in args.starts10.split(",")])]
    if not args.chain10:
        plan.insert(0, (30, [int(s) for s in args.starts30.split(",")]))
    for horizon, starts in plan:
        Xs, ys = [], []
        clim_cur["X"] = CL_train
        for s0 in starts:
            u = u_all[:, :, s0, :].astype(np.float64)
            X = so.feats(u, agef, with_profile)
            if args.anom:
                X = with_anom(X, np.concatenate([anom(np.asarray(ann[lo:lo + 6000]), s0, s0 + horizon) for lo in range(0, n_sites, 6000)]))
            elif args.clim:
                X = with_anom(X, None)
            y = np.concatenate([(u_all[:, j, s0 + horizon, :] - u_all[:, j, s0, :]) for j in range(15)], axis=0)
            keep = fit_rows
            if args.no_event:
                a_tr = np.asarray(traj[:, :, s0:s0 + horizon + 1, 1], dtype=np.float64)  # (15, n, h+1) physical agb
                drop = a_tr[:, :, :-1] - a_tr[:, :, 1:]
                ev = ((drop > 1.0) & (drop / np.maximum(a_tr[:, :, :-1], 1e-6) > 0.10)).any(axis=2).reshape(-1)
                keep = fit_rows & ~ev
                print(f"  window {s0}+{horizon}: dropping {ev[fit_rows].mean():.3f} of the fit rows (events)", flush=True)
            if args.slot_feature >= 0:
                X = np.concatenate([X, np.full((X.shape[0], 1), s0, np.float32)], axis=1)
            Xs.append(X[keep]); ys.append(y[keep])
        X = np.concatenate(Xs); y = np.concatenate(ys); del Xs, ys
        print(f"horizon {horizon} (starts {starts}): {X.shape[0]:,} rows, {X.shape[1]} features", flush=True)
        models[horizon] = [lgb.train(params, lgb.Dataset(X, label=y[:, k]), num_boost_round=args.rounds) for k in range(POOLS)]
        del X, y

    pred_slot = {"v": args.slot_feature}

    def step(u: np.ndarray, horizon: int, A: np.ndarray | None = None) -> np.ndarray:
        n = u.shape[0]
        X = with_anom(so.feats(u, agef, with_profile), A)
        if args.slot_feature >= 0:
            X = np.concatenate([X, np.full((X.shape[0], 1), pred_slot["v"], np.float32)], axis=1)
        ch = np.stack([b.predict(X, num_threads=args.threads) for b in models[horizon]], axis=1)
        out = np.clip(X[:, :POOLS] + ch, 0, None).reshape(15, n, POOLS).transpose(1, 0, 2)
        return out

    def roll(u0: np.ndarray, a: np.ndarray | None = None) -> np.ndarray:
        if args.chain10:
            u = u0
            for k in range(4):
                u = step(u, 10, anom(a, 10 * k, 10 * k + 10) if args.anom else None)
            return u
        A30 = anom(a, 0, 30) if args.anom else None
        A10 = anom(a, 30, 40) if args.anom else None
        return step(step(u0, 30, A30), 10, A10)

    ages = np.array(io.SEED_AGES)
    for fold, sites in (("public", split.public), ("private", split.private)):
        clim_cur["X"] = CL_train[sites] if args.clim else None
        u0 = u_all[sites][:, :, 0, :].astype(np.float64)
        p40 = roll(u0, np.asarray(ann[sites]) if args.anom else None)[:, :, :2]
        p40[u0[:, :, 0] * y_mean[0] < 1.0] = 0.0
        true = u_all[sites][:, :, 40, :2]
        se = (p40 - true) ** 2; sp = (u0[:, :, :2] - true) ** 2
        old = ages >= 150
        print(f"  {fold} from slot 0 (30+10): score {se.mean():.4f} (old {se[:, old].mean():.4f}, young {se[:, ages <= 30].mean():.4f}) persistence {sp.mean():.4f}", flush=True)
    sites = split.public
    clim_cur["X"] = CL_train[sites] if args.clim else None
    u10 = u_all[sites][:, :, 10, :].astype(np.float64)
    if args.chain10:
        u = u10
        for k in range(3):
            u = step(u, 10, anom(np.asarray(ann[sites]), 10 + 10 * k, 20 + 10 * k) if args.anom else None)
        p40 = u[:, :, :2]
    else:
        p40 = step(u10, 30, anom(np.asarray(ann[sites]), 10, 40) if args.anom else None)[:, :, :2]
    true = u_all[sites][:, :, 40, :2]
    se = (p40 - true) ** 2; sp = (u10[:, :, :2] - true) ** 2
    old = ages >= 150
    print(f"  public from slot 10 (30-year model): score {se.mean():.4f} old {se[:, old].mean():.4f} | persistence {sp.mean():.4f} old {sp[:, old].mean():.4f} | "
          f"old explained {1 - se[:, old].mean() / sp[:, old].mean():.3f} | mean old agb change pred {np.mean(p40[:, old, 1] - u10[:, old, 1]):+.4f} true {np.mean(true[:, old, 1] - u10[:, old, 1]):+.4f}", flush=True)
    template = np.loadtxt(io.CACHE / "sample_submission.csv", delimiter=",", skiprows=1, dtype=str, usecols=0)
    ids, hs, as_ = [], [], []
    for scenario in io.SCENARIOS:
        y0 = io.open_test(scenario)[1].astype(np.float64)[:, :, :POOLS]
        u0t = DECODE * (y0 - shift.FLOORS[:POOLS]) / y_mean[:POOLS]
        a_t = np.load(cache.CACHE_DIR / f"test_{scenario}_annual.npy") if (args.anom or args.clim) else None
        if args.clim:
            clim_cur["X"] = np.concatenate([latent.seasonal_block(np.load(cache.CACHE_DIR / f"test_{scenario}_season.npy"))[:, :24],
                                            np.asarray(a_t[:, 0, SOIL])], 1).astype(np.float32)
        p40 = roll(u0t, a_t)[:, :, :2]
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
