"""Adversarial validation and a state-climate consistency test, train against test.

1. A LightGBM classifier on the model's site-level inputs (seasonal block, static
   drivers, decoded y0 at every seed age) told to separate fit sites from competition
   sites; out-of-fold AUC and the features it uses. AUC near 0.5 means the test
   sites are exchangeable with the fit sites on these inputs; near 1.0 means the
   model can tell them apart, and the importances say from what.
2. The relation between the initial state and the climate: a LightGBM regressor
   fitted on the fit sites predicting decoded y0 height/agb at seed ages 100 and
   500 from the site's climate and soil. On the holdout sites its residual is the
   fit-site relation's noise; on the competition sites a wider or displaced residual
   says the scored stands do not sit where the training climate-to-stand relation
   puts them, i.e. the initial state was produced under a different climate.

Runs in the isolated scratchpad venv (lightgbm 4.7.0, scikit-learn 1.9.1):

    <scratchpad>/venvs/gap/bin/python scripts/gap_adversarial.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, holdout, io, latent, shift  # noqa: E402

SEASON_NAMES = [f"{g}_m{m:02d}" for g in ("ta_m", "pr", "hus", "ta_h", "rsds", "wind") for m in range(12)]
STATIC_NAMES = ["dst", "k_sat", "s_theta", "r_theta", "L", "n", "m", "sd"]


def site_features(season: np.ndarray, static: np.ndarray, u0: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """``season`` (n, 72), ``static`` (n, 8), ``u0`` (n, 15, 7) scaled -> (n, 72 + 8 + 60)."""
    cols = [season, static]
    names = SEASON_NAMES + STATIC_NAMES
    for j, age in enumerate(io.SEED_AGES):
        cols.append(u0[:, j, :4])
        names += [f"y0_{v}_a{age}" for v in ("h", "agb", "soil", "lai")]
    return np.concatenate(cols, axis=1).astype(np.float32), names


def main() -> None:
    split = holdout.make_split(seed=42)
    stats = io.load_stats()
    y_mean = stats["y_mean"].astype(np.float64)
    xa = cache.open_x_annual()
    traj = cache.open_traj()

    def train_side(sites: np.ndarray):
        season = latent.seasonal_block(cache.open_x_season()[sites]).astype(np.float64)
        static = np.asarray(xa[sites, 0][:, list(cache.STATIC_DRIVERS)], dtype=np.float64)
        u0 = np.transpose(np.asarray(traj[:, sites, 0, :], dtype=np.float64), (1, 0, 2)) / y_mean
        return season, static, u0

    floors = shift.measured_floors()
    test = {}
    for scen in io.SCENARIOS:
        sea_t = np.load(cache.CACHE_DIR / f"test_{scen}_season.npy")
        ann_t = np.load(cache.CACHE_DIR / f"test_{scen}_annual.npy", mmap_mode="r")
        season = latent.seasonal_block(sea_t).astype(np.float64)
        static = np.asarray(ann_t[:, 0][:, list(cache.STATIC_DRIVERS)], dtype=np.float64)
        y0 = io.open_test(scen)[1].astype(np.float64)
        u0 = 1.14114 * (y0 - floors) / y_mean
        test[scen] = (season, static, u0)

    fit_sites = split.fit[::4]        # 12k fit sites against 6171 test sites
    hold_sites = split.holdout
    s_fit, st_fit, u_fit = train_side(fit_sites)
    s_hold, st_hold, u_hold = train_side(hold_sites)
    X_fit, names = site_features(s_fit, st_fit, u_fit)
    X_hold, _ = site_features(s_hold, st_hold, u_hold)
    X_test, _ = site_features(*test["ssp126"])
    out = {}

    # ---- 1. adversarial validation ----------------------------------------------
    def adversarial(label: str, X_a: np.ndarray, X_b: np.ndarray, cols: list[int]) -> dict:
        X = np.concatenate([X_a[:, cols], X_b[:, cols]]); y = np.r_[np.zeros(len(X_a)), np.ones(len(X_b))]
        oof = np.zeros(len(y)); imp = np.zeros(len(cols))
        for tr, te in KFold(5, shuffle=True, random_state=42).split(X):
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, verbose=-1, random_state=42)
            m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]; imp += m.booster_.feature_importance("gain")
        auc = roc_auc_score(y, oof)
        top = np.argsort(-imp)[:12]
        print(f"\n[adversarial] {label}: AUC {auc:.4f}")
        print("   top gain features: " + ", ".join(f"{names[cols[i]]} ({imp[i] / imp.sum():.2f})" for i in top))
        return {"auc": float(auc), "top": [(names[cols[i]], float(imp[i] / imp.sum())) for i in top]}

    all_cols = list(range(X_fit.shape[1]))
    clim_cols = list(range(72)); static_cols = list(range(72, 80)); y0_cols = list(range(80, X_fit.shape[1]))
    out["adv_holdout_vs_fit_all"] = adversarial("holdout sites vs fit sites, all inputs (control)", X_fit, X_hold, all_cols)
    out["adv_test_vs_fit_all"] = adversarial("competition sites vs fit sites, all inputs", X_fit, X_test, all_cols)
    out["adv_test_vs_fit_climate"] = adversarial("competition vs fit, seasonal block only", X_fit, X_test, clim_cols)
    out["adv_test_vs_fit_static"] = adversarial("competition vs fit, static drivers only", X_fit, X_test, static_cols)
    out["adv_test_vs_fit_y0"] = adversarial("competition vs fit, decoded y0 only", X_fit, X_test, y0_cols)
    # y0 only, but with soil driver 'sd' removed and the climate block replaced by physically robust summaries
    robust = [names.index(f"ta_m_m{m:02d}") for m in range(12)] + [names.index(f"pr_m{m:02d}") for m in range(12)]
    out["adv_test_vs_fit_ta_pr"] = adversarial("competition vs fit, ta_m and pr climatology only", X_fit, X_test, robust)

    # ---- 2. state-climate consistency -------------------------------------------------
    print("\n[consistency] decoded y0 from climate + soil (fit on fit sites)")
    clim = clim_cols + [names.index(n) for n in ("k_sat", "s_theta", "r_theta", "L", "n", "m")]  # driver 'sd' left out
    cons = {}
    for age in (100, 500):
        for var, k in (("height", 0), ("agb", 1)):
            col = names.index(f"y0_{'h' if k == 0 else 'agb'}_a{age}")
            m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, verbose=-1, random_state=42)
            m.fit(X_fit[:, clim], X_fit[:, col])
            res = {}
            for label, X in (("holdout", X_hold), ("ssp126", X_test), ("ssp585", site_features(*test["ssp585"])[0])):
                pred = m.predict(X[:, clim]); truth = X[:, col]
                r = truth - pred
                veg = truth > 0.02
                res[label] = {"mean_resid": float(r.mean()), "sd_resid": float(r.std()), "rmse": float(np.sqrt((r ** 2).mean())),
                              "r2": float(1 - (r ** 2).mean() / truth.var()), "mean_truth": float(truth.mean()), "mean_pred": float(pred.mean()),
                              "ratio_truth_pred_veg": float(truth[veg].mean() / pred[veg].mean()),
                              "corr": float(np.corrcoef(pred, truth)[0, 1])}
                print(f"   age {age} {var:6s} {label:8s}: resid mean {r.mean():+.4f} sd {r.std():.4f} rmse {np.sqrt((r**2).mean()):.4f} "
                      f"R2 {res[label]['r2']:.3f} corr {res[label]['corr']:.3f} | truth mean {truth.mean():.3f} pred mean {pred.mean():.3f} "
                      f"ratio(veg) {res[label]['ratio_truth_pred_veg']:.3f}")
            cons[f"a{age}_{var}"] = res
    out["consistency"] = cons

    # the age structure of y0 itself (forcing-free): ratios between ages, train vs test
    print("\n[age structure] median y0 ratio between seed ages on vegetated sites (train fit / holdout / ssp126 / ssp585)")
    for var, k in (("height", 0), ("agb", 1)):
        for a_hi, a_lo in ((500, 100), (100, 30), (30, 10), (500, 200)):
            i_hi, i_lo = io.SEED_AGES.index(a_hi), io.SEED_AGES.index(a_lo)
            vals = []
            for u in (u_fit, u_hold, test["ssp126"][2], test["ssp585"][2]):
                ok = (u[:, i_lo, k] > 0.05) & (u[:, i_hi, k] > 0.05)
                vals.append(float(np.median(u[ok, i_hi, k] / u[ok, i_lo, k])))
            print(f"   {var:6s} y0[{a_hi}]/y0[{a_lo}]: " + "  ".join(f"{v:.3f}" for v in vals))

    with open(io.ROOT / "work" / "metrics" / "gap_adversarial.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("saved work/metrics/gap_adversarial.json")


if __name__ == "__main__":
    main()
