#!/usr/bin/env python
"""Off-equilibrium excesses of the initial state and their linear effect on the 40-year change.

For each of the seven initial-state variables, the training slot-0 relation predicts it from the other
pools (fluxes from the four pools), the seed age and the site climate. The excess is actual minus predicted
(cross-fitted on two site folds for training rows, full fit for the competition rows). On the training
holdout the year-40 residual of the gbdt2 model is regressed on the seven excesses (per band and column):
these coefficients say how the dynamics respond to an off-equilibrium state beyond what the trees capture,
and multiplied by the competition's (much larger) excesses they give linear correction directions that a
tree cannot extrapolate. Writes work/cache/excess_{train_holdout,test}.npz and probe files per band.

    <lgb venv>/bin/python scripts/excess_directions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cgb import cache, holdout, io, latent, shift  # noqa: E402
import direct_gbdt  # noqa: E402

AGES = np.array(io.SEED_AGES)
SOIL = [129, 130, 131, 132, 133, 134, 135]
NAMES = ["height", "agb", "soil", "lai", "gpp", "npp", "rh"]
DEC = np.array([1.14114, 1.14114, 1.14114, 1.0, 1.14114, 1.14114, 1.14114])


def feats(u: np.ndarray, cl: np.ndarray, tgt: int) -> np.ndarray:
    keep = [k for k in range(4) if k != tgt]
    return np.concatenate([np.concatenate([u[j][:, keep], np.full((u.shape[1], 1), AGES[j], np.float32), cl], 1) for j in range(15)]).astype(np.float32)


def lgbm() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=127, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=6)


def main() -> None:
    y_mean = io.load_stats()["y_mean"]; agef = latent.age_features(io.load_stats())
    split = holdout.make_split(seed=42)
    sea = cache.open_x_season(); ann = cache.open_x_annual()
    cl = np.concatenate([latent.seasonal_block(sea)[:, :24], np.asarray(ann[:, 0, SOIL])], 1).astype(np.float32)
    traj = cache.open_traj()
    u = np.asarray(traj[:, :, 0, :7], dtype=np.float32)
    n = u.shape[1]; site = np.tile(np.arange(n), 15); veg = u[:, :, 0].reshape(-1) > 1.0
    fold = np.random.default_rng(5).random(n) < 0.5
    exc_tr = np.zeros((15 * n, 7), np.float32); full = {}
    for tgt in range(7):
        X = feats(u, cl, tgt); y = u[:, :, tgt].reshape(-1)
        for f in (False, True):
            tr = (fold[site] == f) & veg; te = fold[site] != f
            exc_tr[te, tgt] = (y[te] - lgbm().fit(X[tr], y[tr]).predict(X[te])) / y_mean[tgt]
        full[tgt] = lgbm().fit(X[veg], y[veg])
        print(f"relation {NAMES[tgt]} fitted", flush=True)
    exc_tr[~veg] = 0.0
    # gbdt2 holdout residual at year 40
    ho = split.holdout
    feat = direct_gbdt.site_features(np.asarray(ann[ho]), np.asarray(sea[ho]))
    u0s = np.transpose(np.asarray(traj[:, ho, 0, :4], dtype=np.float64) / y_mean[:4], (1, 0, 2))
    Xg = direct_gbdt.rows(feat, u0s, agef)
    bg = [lgb.Booster(model_file=str(ROOT / "work" / "models" / "direct_gbdt2" / f"{k}.txt")) for k in ("height", "agb")]
    pg = np.clip(Xg[:, -19:-17] + np.stack([b.predict(Xg) for b in bg], 1), 0, None)
    t40 = np.concatenate([np.asarray(traj[j, ho, 40, :2], dtype=np.float64) / y_mean[:2] for j in range(15)])
    rows_ho = np.concatenate([j * n + ho for j in range(15)])
    E = exc_tr[rows_ho]; res = t40 - pg; age_ho = np.repeat(AGES, len(ho)); vh = veg[rows_ho]
    coef = {}
    for band, am in (("young", age_ho <= 30), ("mid", (age_ho >= 50) & (age_ho <= 100)), ("old", age_ho >= 150)):
        m = am & vh
        for k, col in ((0, "h"), (1, "a")):
            A = np.c_[E[m], np.ones(m.sum())]
            beta, *_ = np.linalg.lstsq(A, res[m, k], rcond=None)
            r2 = 1 - np.var(res[m, k] - A @ beta) / np.var(res[m, k])
            coef[(band, k)] = beta[:7]
            print(f"holdout {band:5s} {col}: gbdt2 residual on the 7 excesses: R2 {r2:.3f}; beta " + " ".join(f"{NAMES[i]} {beta[i]:+.3f}" for i in range(7)), flush=True)
    # competition excesses
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    keys = shift.template_keys(); sc = keys["scenario"].to_numpy(); st = keys["site"].to_numpy(); ag = keys["age"].to_numpy()
    exc_te = np.zeros((len(R), 7))
    for s in io.SCENARIOS:
        y0 = io.open_test(s)[1].astype(np.float64)
        dec = np.stack([DEC[k] * (y0[:, :, k] - shift.FLOORS[k]) for k in range(7)], 2)
        ut = np.transpose(dec, (1, 0, 2)).astype(np.float32)
        xt = np.load(ROOT / "work" / "cache" / f"test_{s}_annual.npy", mmap_mode="r")
        clt = np.concatenate([latent.seasonal_block(np.load(ROOT / "work" / "cache" / f"test_{s}_season.npy"))[:, :24], np.asarray(xt[:, 0, SOIL])], 1).astype(np.float32)
        m = sc == s
        for tgt in range(7):
            p = full[tgt].predict(feats(ut, clt, tgt)).reshape(15, -1).T
            e = (dec[:, :, tgt] - p) / y_mean[tgt]
            exc_te[m, tgt] = e[st[m], np.searchsorted(AGES, ag[m])]
    exc_te[R[:, 0] < 1.0 / y_mean[0]] = 0.0
    np.savez(ROOT / "work" / "cache" / "excess_test.npz", exc=exc_te)
    print("test excess sd per variable:", np.round(exc_te.std(0), 3), "| training holdout:", np.round(E[vh].std(0), 3))
    print("test excess correlations (old band):"); mo = shift.band_masks(ag)["old"] & (R[:, 0] > 0.1)
    print(np.round(np.corrcoef(exc_te[mo].T), 2))
    # correction directions per band and column: D = beta . excess (linear extrapolation)
    masks = shift.band_masks(ag)
    for band in ("young", "mid", "old"):
        for k, col in ((0, "height"), (1, "agb")):
            D = np.zeros_like(R); D[masks[band], k] = exc_te[masks[band]] @ coef[(band, k)]
            P = R.copy(); P[:, k] = np.clip(R[:, k] + D[:, k], 0, None)
            path = ROOT / "results" / f"excessprobe_{band}_{col}.csv"
            shift.write_predictions(P, path)
            print(f"{path.name}: E[D^2] pooled {np.mean(D**2):.5f}, mean D {D[masks[band], k].mean():+.4f}")


if __name__ == "__main__":
    main()
