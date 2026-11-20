#!/usr/bin/env python
"""Direct LightGBM on shift-robust features: a second, independent model for the board blend.

No CO2 drivers (a clock in the training forcing), no hour-of-day drivers (the competition's
hourly pipeline differs); the forcing enters as per-site 40-year summaries of the eight
monthly drivers and the four daily means of the hourly groups (mean, interannual sd, linear
trend, four decade means), plus the seasonal climatology of ta_m/pr, the seven soil drivers,
the decoded initial state (height, agb, soil, lai), the seed age and the age triplet. Target:
the forty-year change of height and agb in scaled units (persistence is the zero function).
Fitted on the fit sites of ``cgb.holdout.make_split(seed=42)``, scored on both holdout folds,
rolled over the competition inputs with the same decode as ``predict_latent.py``.

    <lightgbm venv>/bin/python scripts/direct_gbdt.py --tag gbdt1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, holdout, io, shift  # noqa: E402
try:
    from cgb import latent  # noqa: E402
except ImportError:  # an interpreter with LightGBM but no torch
    from cgb import latent_np as latent  # noqa: E402
from cgb import mortality as mort  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

MONTHLY = list(range(0, 8))  # ta_m, pr, tsl1..6
HOURLY = {"hus": 33, "ta_h": 34, "rsds": 35, "wind": 36}
DECODE = np.array([1.14114, 1.14114, 1.14114, 1.0])  # height, agb, soil, lai
FLOORS = shift.FLOORS[:4]


USE_HOURLY = True  # set False by --no-hourly: only the eight monthly drivers enter the forcing summaries


def forcing_summaries(ann: np.ndarray) -> np.ndarray:
    """``(n, 12*7)`` per-site summaries of 12 annual series (8 monthly drivers + 4 daily-mean hourly groups)."""
    n = ann.shape[0]
    series = [np.asarray(ann[:, :, j], dtype=np.float64) for j in MONTHLY]
    for start in (HOURLY.values() if USE_HOURLY else ()):
        idx = start + 4 * np.arange(24)
        series.append(np.asarray(ann[:, :, idx], dtype=np.float64).mean(axis=2))
    n_y = series[0].shape[1]  # 40 in training; a shorter window gives quarter means instead of decade means
    years = np.arange(n_y) - (n_y - 1) / 2
    q = n_y // 4
    out = []
    for s in series:  # (n, n_y)
        out += [s.mean(1), s.std(1), (s * years).sum(1) / (years**2).sum()]
        out += [s[:, d * q:(d + 1) * q].mean(1) for d in range(4)]
    return np.stack(out, axis=1).astype(np.float32)


NORMALS_ONLY = False  # set True by --normals-only: no window-specific forcing summaries (the no-window baseline B0)


def site_features(ann: np.ndarray, sea: np.ndarray) -> np.ndarray:
    block = latent.seasonal_block(sea)[:, :24]  # ta_m and pr climatology only
    static = np.asarray(ann[:, 0, [129, 130, 131, 132, 133, 134, 135]], dtype=np.float32)
    if NORMALS_ONLY:
        return np.concatenate([block, static], axis=1)
    return np.concatenate([forcing_summaries(ann), block, static], axis=1)


POSITIONAL = ("_max_y", "_argmax", "_y40")


SUBANNUAL_KEYS = ("spi1", "spi3", "spi6", "dryrun", "vpd3")  # sub-annual timing indices the detector left out


def drought_thresholds() -> dict[str, tuple[float, float]]:
    """q90/q99 of each drought index over (a quarter of) the fit sites, as ``mort_detector.py`` does."""
    import mort_detector
    from mort_trigger import build_indices

    if SUBANNUAL:
        mort_detector.SEQ_KEYS = tuple(mort_detector.SEQ_KEYS) + tuple(k for k in SUBANNUAL_KEYS if k not in mort_detector.SEQ_KEYS)
    split = holdout.make_split(seed=42)
    ix = build_indices(np.asarray(mort.open_monthly("train")[split.fit[::4]]))
    return {k: (float(np.nanquantile(ix[k], 0.9)), float(np.nanquantile(ix[k], 0.99))) for k in mort_detector.SEQ_KEYS}


SUBANNUAL = False


def drought_site_features(monthly: np.ndarray, thr: dict[str, tuple[float, float]]) -> np.ndarray:
    """Position-free drought summaries (maxima, exceedance counts and sums of six indices) plus
    the eight climate descriptors of ``mort_detector.py``; the per-band maxima and the year of
    the maximum are dropped because the competition forcing's extremes sit in other years."""
    from mort_detector import climatology, seq_features
    from mort_trigger import build_indices

    out = []
    for lo in range(0, monthly.shape[0], 4000):
        m = np.asarray(monthly[lo:lo + 4000])
        ix = build_indices(m)
        ix.pop("_pet_over_p", None)
        seq, names = seq_features(ix, thr)
        keep = [i for i, n in enumerate(names) if not any(p in n for p in POSITIONAL)]
        out.append(np.concatenate([seq[:, keep], climatology(m)], axis=1).astype(np.float32))
    return np.concatenate(out, axis=0)


def rows(site_feat: np.ndarray, u0: np.ndarray, agef: np.ndarray) -> np.ndarray:
    """Stack all 15 seed ages: ``u0`` is ``(n, 15, 4)`` scaled decoded state; returns ``(n*15, F)`` age-major."""
    n = site_feat.shape[0]
    parts = []
    for j in range(15):
        parts.append(np.concatenate([site_feat, u0[:, j, :], np.repeat(agef[j][None], n, axis=0)], axis=1))
    return np.concatenate(parts, axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="gbdt1")
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drought", action="store_true", help="append the position-free drought-index summaries (cgb.mortality caches)")
    ap.add_argument("--no-hourly", action="store_true", help="drop the four daily-mean hourly groups from the forcing summaries")
    ap.add_argument("--skip-age1", action="store_true", help="with --age-profile: leave the (anomalous) age-1 states out")
    ap.add_argument("--allo", action="store_true",
                    help="add the allometric excess of the initial state (h0 - h_allo(agb0, soil0, lai0, age, climate) and agb0 - a_allo(h0, ...)), "
                         "cross-fitted over two site folds for the training rows and fitted on all training sites for the test rows")
    ap.add_argument("--linear-tree", action="store_true", help="LightGBM linear_tree=True (leaves with linear models: extrapolates the excess)")
    ap.add_argument("--fluxes", action="store_true",
                    help="add the year-0 fluxes (gpp, npp, rh; decoded x1.141 above the bare-row plateau) of the row and of the site's profile")
    ap.add_argument("--subannual", action="store_true", help="with --drought: add the sub-annual timing indices (spi1, spi3, spi6, dryrun, vpd3)")
    ap.add_argument("--zone", default="", choices=["", "trop", "extra"],
                    help="fit only on training sites whose 40-year mean air temperature is above (trop) or at/below (extra) 293 K")
    ap.add_argument("--no-event-train", action="store_true",
                    help="fit only on (site, age) rows whose 40-year agb trajectory has no mortality event (single-year drop "
                         "> 1 kgC and > 10 %%): the background dynamics without collapses")
    ap.add_argument("--max-transient", type=float, default=0.0,
                    help="fit only on rows whose climate shows little training-world transient: |g40(clim, age) / "
                         "g0(clim, age + 40) - 1| below this for agb, with g0/g40 climate-only relations of the slot-0 and "
                         "year-40 states (the test world shows no relaxation outside the collapse cluster)")
    ap.add_argument("--normals-only", action="store_true",
                    help="site features = seasonal T/P climatology + soil only: the training world's expected outcome without "
                         "the window's interannual anomalies, trends and decade means (baseline B0 for the window-effect split)")
    ap.add_argument("--age-profile", action="store_true",
                    help="append the site's initial state at all 15 seed ages (height, agb, soil, lai; 60 features): the site's "
                         "spin-up history under its own forcing, which the competition provides for its own climate")
    args = ap.parse_args()
    global USE_HOURLY, SUBANNUAL, NORMALS_ONLY
    USE_HOURLY = not args.no_hourly
    NORMALS_ONLY = args.normals_only
    SUBANNUAL = args.subannual
    stats = io.load_stats()
    y_mean = stats["y_mean"].astype(np.float64)
    agef = latent.age_features(stats)
    split = holdout.make_split(seed=42)
    ann = cache.open_x_annual()
    sea = cache.open_x_season()
    print("building training features ...", flush=True)
    feat = np.concatenate([site_features(np.asarray(ann[lo:lo + 4000]), np.asarray(sea[lo:lo + 4000])) for lo in range(0, ann.shape[0], 4000)])
    thr = None
    if args.drought:
        thr = drought_thresholds()
        feat = np.concatenate([feat, drought_site_features(mort.open_monthly("train"), thr)], axis=1)
        print(f"drought features appended: {feat.shape[1]} site features", flush=True)
    traj = cache.open_traj()
    u0 = np.transpose(np.asarray(traj[:, :, 0, :4], dtype=np.float64) / y_mean[:4], (1, 0, 2))  # (n, 15, 4)
    if args.fluxes:
        fl0 = np.transpose(np.asarray(traj[:, :, 0, 4:7], dtype=np.float64) / y_mean[4:7], (1, 0, 2))  # (n, 15, 3)
        feat = np.concatenate([feat, fl0.reshape(fl0.shape[0], -1).astype(np.float32)], axis=1)
        print(f"flux profile appended: {feat.shape[1]} site features", flush=True)
    if args.age_profile:
        keep = [j for j in range(15) if not (args.skip_age1 and j == 0)]
        feat = np.concatenate([feat, u0[:, keep, :].reshape(u0.shape[0], -1).astype(np.float32)], axis=1)
        print(f"age profile appended: {feat.shape[1]} site features", flush=True)
    y40 = np.transpose(np.asarray(traj[:, :, 40, :2], dtype=np.float64) / y_mean[:2], (1, 0, 2))  # (n, 15, 2)
    target = (y40 - u0[:, :, :2])  # change
    X_all = rows(feat, u0, agef)
    if args.allo:
        import allometry_relax as ar
        sea_all = latent.seasonal_block(np.asarray(sea)); cl = np.concatenate([sea_all[:, :24], np.asarray(ann[:, 0, ar.SOIL])], 1).astype(np.float32)
        u0p = np.asarray(traj[:, :, 0, :4], dtype=np.float32)  # physical (15, n, 4)
        fold = (np.random.default_rng(3).random(u0p.shape[1]) < 0.5)
        exc = np.zeros((X_all.shape[0], 2), dtype=np.float32)
        allo_models = {}
        site_row = np.tile(np.arange(u0p.shape[1]), 15)
        for tcol in (0, 1):
            Xa = ar.rows(u0p, cl, tcol); ya = u0p[:, :, tcol].reshape(-1); vg = u0p[:, :, 0].reshape(-1) > 1.0
            pred = np.zeros_like(ya)
            for f in (False, True):
                tr = (fold[site_row] == f) & vg
                mdl = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=255, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=args.threads).fit(Xa[tr], ya[tr])
                te = fold[site_row] != f
                pred[te] = mdl.predict(Xa[te])
            allo_models[tcol] = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=255, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=args.threads).fit(Xa[vg], ya[vg])
            exc[:, tcol] = np.where(vg, (ya - pred) / y_mean[tcol], 0.0)
        X_all = np.concatenate([X_all, exc], axis=1)
        print(f"allometric excess appended (cross-fitted); quantiles h {np.quantile(exc[:, 0], [0.01, 0.5, 0.99]).round(3)}", flush=True)
    site_of_row = np.tile(np.arange(ann.shape[0]), 15)
    t_all = np.concatenate([target[:, j, :] for j in range(15)], axis=0)
    fit_rows = np.isin(site_of_row, split.fit)
    if args.zone:
        t_site = np.asarray(ann[:, :, 0]).mean(axis=1)
        keep = (t_site > 293.0) if args.zone == "trop" else (t_site <= 293.0)
        fit_rows = fit_rows & keep[site_of_row]
        print(f"zone {args.zone}: fitting on {keep.mean():.3f} of the sites", flush=True)
    if args.max_transient > 0:
        # climate-only relations (site features + age) of the slot-0 and year-40 agb, fitted on the fit sites
        n_s = ann.shape[0]
        Xc = np.concatenate([np.concatenate([feat, np.repeat(agef[j][None], n_s, 0)], 1) for j in range(15)]).astype(np.float32)
        a0 = np.concatenate([u0[:, j, 1] for j in range(15)]); a40 = np.concatenate([y40[:, j, 1] for j in range(15)])
        prm = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.7,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=args.threads, verbose=-1, seed=42)
        g0 = lgb.train(prm, lgb.Dataset(Xc[fit_rows], label=a0[fit_rows]), 600).predict(Xc, num_threads=args.threads).reshape(15, n_s)
        g40 = lgb.train(prm, lgb.Dataset(Xc[fit_rows], label=a40[fit_rows]), 600).predict(Xc, num_threads=args.threads).reshape(15, n_s)
        ages = np.asarray(io.SEED_AGES, dtype=float)
        g0_shift = np.stack([np.interp(np.minimum(ages + 40, 500.0), ages, g0[:, s]) for s in range(n_s)], 1)   # (15, n)
        trans = g40 / np.maximum(g0_shift, 0.05) - 1
        tr_row = trans.reshape(-1)
        keep_tr = (np.abs(tr_row) < args.max_transient) | (np.maximum(g0_shift, 0).reshape(-1) < 0.05)
        print(f"low-transient training: keeping {keep_tr[fit_rows].mean():.3f} of the fit rows (|transient| < {args.max_transient})", flush=True)
        fit_rows = fit_rows & keep_tr
    if args.no_event_train:
        a_tr = np.asarray(traj[:, :, :, 1], dtype=np.float64)  # (15, n, 41) physical agb
        drop = a_tr[:, :, :-1] - a_tr[:, :, 1:]
        ev = ((drop > 1.0) & (drop / np.maximum(a_tr[:, :, :-1], 1e-6) > 0.10)).any(axis=2)  # (15, n)
        ev_row = ev.reshape(-1)  # age-major like X_all
        print(f"no-event training: excluding {ev_row[fit_rows].mean():.3f} of the fit rows", flush=True)
        fit_rows = fit_rows & ~ev_row
    preds = {}
    boosters = []
    for k, name in enumerate(("height", "agb")):
        ds = lgb.Dataset(X_all[fit_rows], label=t_all[fit_rows, k])
        params = dict(objective="regression", learning_rate=args.lr, num_leaves=args.leaves, min_data_in_leaf=100, feature_fraction=0.7,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=args.threads, verbose=-1, seed=args.seed, linear_tree=args.linear_tree)
        b = lgb.train(params, ds, num_boost_round=args.rounds)
        boosters.append(b)
        preds[name] = b.predict(X_all, num_threads=args.threads)
        print(f"  {name}: trained {args.rounds} rounds", flush=True)
    pred_change = np.stack([preds["height"], preds["agb"]], axis=1)
    out_dir = io.ROOT / "work" / "models" / f"direct_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for fold, sites in (("public", split.public), ("private", split.private)):
        m = np.isin(site_of_row, sites)
        off = 2 if args.allo else 0
        u0cols = X_all[m][:, X_all.shape[1] - 19 - off: X_all.shape[1] - 15 - off]
        p40 = u0cols[:, :2] + pred_change[m]  # u0 h, agb sit right before the 15 age features
        p40 = np.clip(p40, 0, None)
        bare = u0cols[:, 0] * y_mean[0] < 1.0
        p40[bare] = 0.0
        true = (t_all[m] + u0cols[:, :2])
        se = (p40 - true) ** 2
        pers = (u0cols[:, :2] - true) ** 2
        res[fold] = {"score": float(se.mean()), "h": float(se[:, 0].mean()), "a": float(se[:, 1].mean()), "persistence": float(pers.mean())}
        print(f"  {fold}: score {se.mean():.4f} (h {se[:,0].mean():.4f} a {se[:,1].mean():.4f}) persistence {pers.mean():.4f}", flush=True)
    for b, name in zip(boosters, ("height", "agb")):
        b.save_model(str(out_dir / f"{name}.txt"))
    # competition
    template = np.loadtxt(io.CACHE / "sample_submission.csv", delimiter=",", skiprows=1, dtype=str, usecols=0)
    ids, hs, as_ = [], [], []
    for scenario in io.SCENARIOS:
        ann_t = np.load(cache.CACHE_DIR / f"test_{scenario}_annual.npy")
        sea_t = np.load(cache.CACHE_DIR / f"test_{scenario}_season.npy")
        y0all = io.open_test(scenario)[1].astype(np.float64)
        y0 = y0all[:, :, :4]
        raw0 = DECODE * (y0 - FLOORS)
        u0t = raw0 / y_mean[:4]
        flt = 1.14114 * (y0all[:, :, 4:7] - shift.FLOORS[4:7]) / y_mean[4:7]
        sf = site_features(ann_t, sea_t)
        if args.drought:
            sf = np.concatenate([sf, drought_site_features(mort.open_monthly(scenario), thr)], axis=1)
        if args.age_profile:
            keep = [j for j in range(15) if not (args.skip_age1 and j == 0)]
            sf = np.concatenate([sf, u0t[:, keep, :].reshape(u0t.shape[0], -1).astype(np.float32)], axis=1)
        if args.fluxes:
            sf = np.concatenate([sf, flt.reshape(flt.shape[0], -1).astype(np.float32)], axis=1)
        Xt = rows(sf, u0t, agef)
        u0cols_t = Xt[:, -19:-15]
        if args.allo:
            clt = np.concatenate([latent.seasonal_block(sea_t)[:, :24], np.asarray(ann_t[:, 0, ar.SOIL])], 1).astype(np.float32)
            u0tp = np.transpose(u0t * y_mean[:4], (1, 0, 2)).astype(np.float32)
            ext = np.zeros((Xt.shape[0], 2), dtype=np.float32)
            vgt = u0tp[:, :, 0].reshape(-1) > 1.0
            for tcol in (0, 1):
                pa = allo_models[tcol].predict(ar.rows(u0tp, clt, tcol))
                ext[:, tcol] = np.where(vgt, (u0tp[:, :, tcol].reshape(-1) - pa) / y_mean[tcol], 0.0)
            Xt = np.concatenate([Xt, ext], axis=1)
            print(f"  {scenario}: test allometric excess quantiles h {np.quantile(ext[:, 0], [0.01, 0.5, 0.99]).round(3)}", flush=True)
        ch = np.stack([b.predict(Xt, num_threads=args.threads) for b in boosters], axis=1)
        n = ann_t.shape[0]
        p40 = np.clip(u0cols_t[:, :2] + ch, 0, None)
        p40[u0cols_t[:, 0] * y_mean[0] < 1.0] = 0.0
        p40 = p40.reshape(15, n, 2).transpose(1, 0, 2)  # (n, 15, 2)
        for site in range(n):
            for age in io.SEED_AGES:
                ids.append(f"{scenario}_{site}_{age}")
        hs.append(p40[..., 0].reshape(-1)); as_.append(p40[..., 1].reshape(-1))
        print(f"  {scenario}: mean change h {np.mean(ch[:, 0]):+.4f} a {np.mean(ch[:, 1]):+.4f}", flush=True)
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
