"""Does the model's holdout error depend on how far a site's initial state sits from
what its climate predicts -- and does the test's distribution of that distance
reproduce the board?

``gap_adversarial.py`` measured that decoded y0 is predictable from climate + soil
with residual sd 0.19 (height) on the holdout and 0.39 on the competition sites. If
the model's error grows with that residual, a holdout reweighted to the test's
residual distribution is a holdout that sees the shift. Here: fit the same regressor
on the fit sites, score every holdout site's residual at age 100, bin the saved
year-40 predictions of ``latent_v1`` by it, and importance-weight.

    <scratchpad>/venvs/gap/bin/python scripts/gap_residual_bins.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, holdout, io, latent, shift  # noqa: E402


def main() -> None:
    split = holdout.make_split(seed=42)
    stats = io.load_stats()
    y_mean = stats["y_mean"].astype(np.float64)
    xa = cache.open_x_annual()
    traj = cache.open_traj()
    soil_idx = [129, 130, 131, 132, 133, 134]  # driver 135 left out: a constant on the test

    def feats(sites: np.ndarray) -> np.ndarray:
        season = latent.seasonal_block(cache.open_x_season()[sites]).astype(np.float64)
        static = np.asarray(xa[sites, 0][:, soil_idx], dtype=np.float64)
        return np.concatenate([season, static], axis=1)

    def y0_scaled(sites: np.ndarray) -> np.ndarray:
        return np.transpose(np.asarray(traj[:, sites, 0, :], dtype=np.float64), (1, 0, 2)) / y_mean  # (n, 15, 7)

    fit_sites = split.fit[::2]
    X_fit, u_fit = feats(fit_sites), y0_scaled(fit_sites)
    a100 = io.SEED_AGES.index(100)
    models = {}
    for k, var in ((0, "height"), (1, "agb")):
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, verbose=-1, random_state=42)
        m.fit(X_fit, u_fit[:, a100, k])
        models[var] = m

    # residuals on the two holdout folds and the competition sites
    floors = shift.measured_floors()
    res = {}
    for fold in ("public", "private"):
        sites = getattr(split, fold)
        X, u = feats(sites), y0_scaled(sites)
        res[fold] = {var: u[:, a100, k] - models[var].predict(X) for k, var in ((0, "height"), (1, "agb"))}
    for scen in io.SCENARIOS:
        sea_t = np.load(cache.CACHE_DIR / f"test_{scen}_season.npy")
        ann_t = np.load(cache.CACHE_DIR / f"test_{scen}_annual.npy", mmap_mode="r")
        X = np.concatenate([latent.seasonal_block(sea_t).astype(np.float64), np.asarray(ann_t[:, 0][:, soil_idx], dtype=np.float64)], axis=1)
        y0 = io.open_test(scen)[1].astype(np.float64)
        u = 1.14114 * (y0 - floors) / y_mean
        res[scen] = {var: u[:, a100, k] - models[var].predict(X) for k, var in ((0, "height"), (1, "agb"))}

    # the model's year-40 error per holdout site
    out = {}
    for fold in ("public", "private"):
        sites = getattr(split, fold)
        pred = np.load(io.ROOT / "work" / "models" / "latent_v1" / f"pred_{fold}.npy")  # (15, n, 5, 2) years 5,10,20,30,40
        pred40 = np.transpose(pred[:, :, -1, :], (1, 0, 2)).astype(np.float64)          # (n, 15, 2)
        u = y0_scaled(sites)
        true40 = np.transpose(np.asarray(traj[:, sites, 40, :2], dtype=np.float64), (1, 0, 2)) / y_mean[:2]
        pers40 = u[:, :, :2]
        bare = (u[:, :, 0] * y_mean[0]) < 1.0
        pred40[bare] = 0.0
        err = ((pred40 - true40) ** 2).mean(axis=(1, 2))          # per site
        perr = ((pers40 - true40) ** 2).mean(axis=(1, 2))
        r = res[fold]["height"]
        print(f"\n== {fold} fold: model {err.mean():.4f}, persistence {perr.mean():.4f}; height residual sd {r.std():.3f}")
        edges = np.quantile(np.abs(np.concatenate([res["ssp126"]["height"], res[fold]["height"]])), [0, .5, .8, .9, .95, .99, 1])
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (np.abs(r) >= lo) & (np.abs(r) < hi) if hi < edges[-1] else (np.abs(r) >= lo)
            share_test = float(np.mean((np.abs(res["ssp126"]["height"]) >= lo) & ((np.abs(res["ssp126"]["height"]) < hi) if hi < edges[-1] else True)))
            rows.append({"lo": float(lo), "hi": float(hi), "n": int(sel.sum()), "share_holdout": float(sel.mean()), "share_test": share_test,
                         "model": float(err[sel].mean()) if sel.any() else None, "persistence": float(perr[sel].mean()) if sel.any() else None})
            print(f"   |resid_h| in [{lo:.3f}, {hi:.3f}): holdout n {sel.sum():5d} ({sel.mean():.3f}) test share {share_test:.3f} | "
                  f"model {rows[-1]['model']:.4f}  persistence {rows[-1]['persistence']:.4f}")
        # signed: does the model err differently on stands taller than their climate?
        for lo, hi, lab in ((-9, -0.3, "much shorter than climate"), (-0.3, -0.1, "shorter"), (-0.1, 0.1, "on the relation"), (0.1, 0.3, "taller"), (0.3, 9, "much taller than climate")):
            sel = (r >= lo) & (r < hi)
            st = float(np.mean((res["ssp126"]["height"] >= lo) & (res["ssp126"]["height"] < hi)))
            print(f"   signed {lab:26s}: holdout share {sel.mean():.3f} test share {st:.3f} | model {err[sel].mean():.4f} persistence {perr[sel].mean():.4f} | "
                  f"mean pred change h {np.mean(pred40[sel, :, 0] - pers40[sel, :, 0]):+.4f} true {np.mean(true40[sel, :, 0] - pers40[sel, :, 0]):+.4f}")
        # importance weights: density ratio of the test's residual over the holdout's, on (height, agb) residuals, via a classifier
        Z = np.column_stack([r, res[fold]["agb"]])
        Zt = np.column_stack([res["ssp126"]["height"], res["ssp126"]["agb"]])
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=50, verbose=-1, random_state=42)
        clf.fit(np.concatenate([Z, Zt]), np.r_[np.zeros(len(Z)), np.ones(len(Zt))])
        p = np.clip(clf.predict_proba(Z)[:, 1], 1e-3, 1 - 1e-3)
        w = p / (1 - p) * (len(Z) / len(Zt))
        w = np.minimum(w, np.quantile(w, 0.99))
        w /= w.mean()
        print(f"   reweighted to the test's residual distribution (weights capped at q99, ESS {w.sum()**2 / (w**2).sum():.0f} of {len(w)}): "
              f"model {np.average(err, weights=w):.4f}  persistence {np.average(perr, weights=w):.4f}")
        out[fold] = {"bins": rows, "reweighted_model": float(np.average(err, weights=w)), "reweighted_persistence": float(np.average(perr, weights=w)),
                     "model": float(err.mean()), "persistence": float(perr.mean())}
    with open(io.ROOT / "work" / "metrics" / "gap_residual_bins.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
