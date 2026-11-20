#!/usr/bin/env python
"""Windowed direct GBDT: a 30-year model plus a 10-year model, trained on windows that start at
several year slots, rolled as two steps over the competition's forty years.

Why: with one training calendar every site's early losses (the 1982-83 El Nino in years 1-3 of
a slot-0 window) are memorised as site properties, and on the board the old band transfers at a
third of its change energy (``notes/board-solve.md``). Windows starting at different slots put
the same droughts at different positions, so a loss has to be attributed to the forcing inside
the window. The 30-year model is fitted on windows s0 -> s0+30 for s0 in {0, 5, 10} (its
forcing summaries are computed inside the window), the 10-year model on s0 in {0, 5, ..., 30}.
Competition rollout: state(y0) --30y--> state30 --10y--> year 40. Both models predict all four
pools (height, agb, soil, lai) so the second step has its inputs.

    <lightgbm venv>/bin/python scripts/direct_gbdt_windows.py --tag gw1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cgb import cache, holdout, io, latent  # noqa: E402
from cgb import mortality as mort  # noqa: E402
from direct_gbdt import DECODE, FLOORS, HOURLY, MONTHLY  # noqa: E402


STRESS_INDEX = "spi12"  # set by --stress-index; spi12 = precipitation-only 12-month deficit, wb12 = P - PET


def window_stress(which: str, lo: int, hi: int, sites: np.ndarray | None = None) -> np.ndarray:
    """Per-site maximum of a drought index (larger = more stress) inside forcing years [lo, hi)."""
    from mort_trigger import build_indices

    monthly = mort.open_monthly(which)
    out = []
    idx = np.arange(monthly.shape[0]) if sites is None else sites
    for start in range(0, len(idx), 4000):
        m = np.asarray(monthly[idx[start:start + 4000]])
        ix = build_indices(m)
        x = ix[STRESS_INDEX][:, lo:hi]
        out.append(np.nanmax(np.where(np.isfinite(x), x, -np.inf), axis=1))
    return np.concatenate(out)


def density_ratio_weights(s_train: np.ndarray, s_test: np.ndarray, n_bins: int = 25, clip: float = 5.0) -> np.ndarray:
    """Importance weights p_test(s)/p_train(s) from histograms on the pooled quantile grid, mean 1, clipped."""
    edges = np.quantile(np.concatenate([s_train, s_test]), np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    h_tr = np.histogram(s_train, edges)[0] / len(s_train) + 1e-6
    h_te = np.histogram(s_test, edges)[0] / len(s_test) + 1e-6
    ratio = np.clip(h_te / h_tr, 1.0 / clip, clip)
    w = ratio[np.clip(np.searchsorted(edges, s_train, side="right") - 1, 0, n_bins - 1)]
    return w / w.mean()

POOLS = 4  # height, agb, soil, lai


def window_summaries(ann: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Per-site summaries of the 12 annual series inside forcing years [lo, hi)."""
    series = [np.asarray(ann[:, lo:hi, j], dtype=np.float64) for j in MONTHLY]
    import direct_gbdt
    for start in (HOURLY.values() if direct_gbdt.USE_HOURLY else ()):
        idx = start + 4 * np.arange(24)
        series.append(np.asarray(ann[:, lo:hi, idx], dtype=np.float64).mean(axis=2))
    n_y = hi - lo
    years = np.arange(n_y) - (n_y - 1) / 2
    out = []
    for s in series:
        out += [s.mean(1), s.std(1), (s * years).sum(1) / (years**2).sum(), s[:, : n_y // 2].mean(1), s[:, n_y // 2 :].mean(1), s.min(1), s.max(1)]
    return np.stack(out, axis=1).astype(np.float32)


def site_static(ann: np.ndarray, sea: np.ndarray) -> np.ndarray:
    block = latent.seasonal_block(sea)[:, :24]
    static = np.asarray(ann[:, 0, [129, 130, 131, 132, 133, 134, 135]], dtype=np.float32)
    return np.concatenate([block, static], axis=1)


def rows(site_feat: np.ndarray, win_feat: np.ndarray, u: np.ndarray, agef: np.ndarray) -> np.ndarray:
    """``u`` is ``(n, 15, 4)`` scaled pools at the window start; returns ``(n*15, F)`` age-major."""
    n = site_feat.shape[0]
    base = np.concatenate([site_feat, win_feat], axis=1)
    parts = [np.concatenate([base, u[:, j, :], np.repeat(agef[j][None], n, axis=0)], axis=1) for j in range(15)]
    return np.concatenate(parts, axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="gw1")
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--stress-index", default="spi12", help="drought index for --stress-weight: spi12 (precipitation) or wb12 (P - PET)")
    ap.add_argument("--no-hourly", action="store_true", help="drop the four daily-mean hourly groups from the window summaries")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--starts30", default="0,5,10", help="window start slots of the 30-year model")
    ap.add_argument("--starts10", default="0,5,10,15,20,25,30", help="window start slots of the 10-year model")
    ap.add_argument("--stress-weight", action="store_true",
                    help="weight every training window by the density ratio test/train of its maximum 12-month "
                         "water-balance deficit (the competition forcing has ten times fewer precipitation droughts)")
    args = ap.parse_args()
    global STRESS_INDEX
    STRESS_INDEX = args.stress_index
    import direct_gbdt
    direct_gbdt.USE_HOURLY = not args.no_hourly
    stats = io.load_stats()
    y_mean = stats["y_mean"].astype(np.float64)
    agef = latent.age_features(stats)
    split = holdout.make_split(seed=42)
    ann = cache.open_x_annual()
    sea = cache.open_x_season()
    traj = cache.open_traj()
    n_sites = ann.shape[0]
    print("site features ...", flush=True)
    sfeat = np.concatenate([site_static(np.asarray(ann[lo:lo + 4000]), np.asarray(sea[lo:lo + 4000])) for lo in range(0, n_sites, 4000)])
    u_all = np.transpose(np.asarray(traj[:, :, :, :POOLS], dtype=np.float32) / y_mean[:POOLS].astype(np.float32), (1, 0, 2, 3))  # (n, 15, 41, 4)
    site_of_row = np.tile(np.arange(n_sites), 15)
    fit_rows = np.isin(site_of_row, split.fit)
    params = dict(objective="regression", learning_rate=args.lr, num_leaves=args.leaves, min_data_in_leaf=100, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=args.threads, verbose=-1, seed=args.seed)
    models = {}
    for horizon, starts in ((30, tuple(int(x) for x in args.starts30.split(","))), (10, tuple(int(x) for x in args.starts10.split(",")))):
        Xs, ys, ws = [], [], []
        if args.stress_weight:
            s_test = np.concatenate([window_stress(sc, 0, horizon) for sc in io.SCENARIOS])
        for s0 in starts:
            wf = np.concatenate([window_summaries(np.asarray(ann[lo:lo + 4000]), s0, s0 + horizon) for lo in range(0, n_sites, 4000)])
            X = rows(sfeat, wf, u_all[:, :, s0, :], agef)
            y = np.concatenate([u_all[:, j, s0 + horizon, :] - u_all[:, j, s0, :] for j in range(15)], axis=0)
            Xs.append(X[fit_rows]); ys.append(y[fit_rows])
            if args.stress_weight:
                s_tr = window_stress("train", s0, s0 + horizon)
                w_site = density_ratio_weights(s_tr, s_test)
                print(f"  horizon {horizon} s0 {s0}: stress quantiles train {np.quantile(s_tr, [.5, .9, .99]).round(1)} test {np.quantile(s_test, [.5, .9, .99]).round(1)}; weight quantiles {np.quantile(w_site, [.1, .5, .9]).round(2)}", flush=True)
                ws.append(np.tile(w_site, 15)[fit_rows])
        X = np.concatenate(Xs); y = np.concatenate(ys)
        w = np.concatenate(ws) if args.stress_weight else None
        print(f"horizon {horizon}: {X.shape[0]:,} fit rows, {X.shape[1]} features", flush=True)
        models[horizon] = []
        for k in range(POOLS):
            b = lgb.train(params, lgb.Dataset(X, label=y[:, k], weight=w), num_boost_round=args.rounds)
            models[horizon].append(b)
            print(f"  pool {k} trained", flush=True)
        del X, y, Xs, ys
    out_dir = io.ROOT / "work" / "models" / f"direct_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for h, bs in models.items():
        for k, b in enumerate(bs):
            b.save_model(str(out_dir / f"h{h}_pool{k}.txt"))

    def roll(ann_p: np.ndarray, sfeat_p: np.ndarray, u0: np.ndarray, s_start: int = 0) -> np.ndarray:
        """Two-step rollout from ``u0`` (n, 15, 4) over forcing years [s_start, s_start+40)."""
        n = ann_p.shape[0]
        wf = window_summaries(ann_p, s_start, s_start + 30)
        X = rows(sfeat_p, wf, u0, agef)
        ch = np.stack([b.predict(X, num_threads=args.threads) for b in models[30]], axis=1)
        u30 = np.clip(X[:, -19:-15] + ch, 0, None).reshape(15, n, POOLS).transpose(1, 0, 2)
        wf = window_summaries(ann_p, s_start + 30, s_start + 40)
        X = rows(sfeat_p, wf, u30, agef)
        ch = np.stack([b.predict(X, num_threads=args.threads) for b in models[10]], axis=1)
        u40 = np.clip(X[:, -19:-15] + ch, 0, None).reshape(15, n, POOLS).transpose(1, 0, 2)
        return u40

    res = {}
    ages = np.array(io.SEED_AGES)
    for fold, sites in (("public", split.public), ("private", split.private)):
        a_p = np.asarray(ann[sites]); u0 = u_all[sites][:, :, 0, :]
        p40 = roll(a_p, sfeat[sites], u0)[:, :, :2]
        bare = u0[:, :, 0] * y_mean[0] < 1.0
        p40[bare] = 0.0
        true = u_all[sites][:, :, 40, :2]
        se = (p40 - true) ** 2; sp = (u0[:, :, :2] - true) ** 2
        res[fold] = {"score": float(se.mean()), "h": float(se[..., 0].mean()), "a": float(se[..., 1].mean()), "old": float(se[:, ages >= 150].mean()), "persistence": float(sp.mean())}
        print(f"  {fold}: score {se.mean():.4f} (h {se[...,0].mean():.4f} a {se[...,1].mean():.4f}; old band {se[:, ages >= 150].mean():.4f} young {se[:, ages <= 30].mean():.4f}) persistence {sp.mean():.4f}", flush=True)
    sites = split.public
    a_p = np.asarray(ann[sites]); u10 = u_all[sites][:, :, 10, :]
    wf = window_summaries(a_p, 10, 40)
    X = rows(sfeat[sites], wf, u10, agef)
    ch = np.stack([b.predict(X, num_threads=args.threads) for b in models[30]], axis=1)
    p40 = np.clip(X[:, -19:-15] + ch, 0, None).reshape(15, len(sites), POOLS).transpose(1, 0, 2)[:, :, :2]
    true = u_all[sites][:, :, 40, :2]
    se = (p40 - true) ** 2; sp = (u10[:, :, :2] - true) ** 2
    print(f"  from slot 10 (30-year model): score {se.mean():.4f} old band {se[:, ages >= 150].mean():.4f} | persistence {sp.mean():.4f} old {sp[:, ages >= 150].mean():.4f} | "
          f"old-band explained {1 - se[:, ages >= 150].mean() / sp[:, ages >= 150].mean():.3f} | mean pred old agb change {np.mean(p40[:, ages >= 150, 1] - u10[:, ages >= 150, 1]):+.4f} true {np.mean(true[:, ages >= 150, 1] - u10[:, ages >= 150, 1]):+.4f}", flush=True)
    template = np.loadtxt(io.CACHE / "sample_submission.csv", delimiter=",", skiprows=1, dtype=str, usecols=0)
    ids, hs, as_ = [], [], []
    for scenario in io.SCENARIOS:
        ann_t = np.load(cache.CACHE_DIR / f"test_{scenario}_annual.npy")
        sea_t = np.load(cache.CACHE_DIR / f"test_{scenario}_season.npy")
        y0 = io.open_test(scenario)[1].astype(np.float64)[:, :, :POOLS]
        u0t = (DECODE * (y0 - FLOORS) / y_mean[:POOLS]).astype(np.float32)
        p40 = roll(ann_t, site_static(ann_t, sea_t), u0t)[:, :, :2]
        p40[u0t[:, :, 0] * y_mean[0] < 1.0] = 0.0
        n = ann_t.shape[0]
        for site in range(n):
            for age in io.SEED_AGES:
                ids.append(f"{scenario}_{site}_{age}")
        hs.append(p40[..., 0].reshape(-1)); as_.append(p40[..., 1].reshape(-1))
        print(f"  {scenario}: mean change h {np.mean(p40[..., 0] - u0t[..., 0]):+.4f} a {np.mean(p40[..., 1] - u0t[..., 1]):+.4f}", flush=True)
    ids = np.array(ids); h = np.concatenate(hs); a = np.concatenate(as_)
    assert np.array_equal(ids, template)
    out = io.ROOT / "results" / f"direct_{args.tag}_m114.csv"
    with open(out, "w") as fh:
        fh.write("id,height,agb\n")
        for i in range(len(ids)):
            fh.write(f"{ids[i]},{h[i]:.6f},{a[i]:.6f}\n")
    with open(out_dir / "results.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {out}; nothing submitted")


if __name__ == "__main__":
    main()
