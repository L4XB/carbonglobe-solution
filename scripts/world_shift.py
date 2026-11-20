#!/usr/bin/env python
"""Training-world vs test-world climate relation of the initial state.

Fits climate (forcing summaries, seasonal climatology, soil) + seed age -> decoded slot-0 state (height, agb) once on
the training fit sites and once on the competition's own y0 (both scenarios, site-blocked two-fold), and compares the
two relations on the competition climate: where the test world holds more than the training world for the same
climate, models trained on the training world see a stand above its capacity and predict a relaxation the test
world may not have. Writes work/world_shift.npz (train_rel, test_rel; template order, scaled physical units).

    <lightgbm venv>/bin/python scripts/world_shift.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cgb import cache, holdout, io, shift  # noqa: E402
try:
    from cgb import latent  # noqa: E402
except ImportError:
    from cgb import latent_np as latent  # noqa: E402
from direct_gbdt import site_features  # noqa: E402

PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=6, verbose=-1, seed=42)


def stack_ages(feat: np.ndarray, agef: np.ndarray) -> np.ndarray:
    n = feat.shape[0]
    return np.concatenate([np.concatenate([feat, np.repeat(agef[j][None], n, 0)], 1) for j in range(15)]).astype(np.float32)


def main() -> None:
    stats = io.load_stats(); y_mean = stats["y_mean"].astype(np.float64); agef = latent.age_features(stats)
    split = holdout.make_split(seed=42)
    ann = cache.open_x_annual(); sea = cache.open_x_season()
    feat = np.concatenate([site_features(np.asarray(ann[lo:lo + 4000]), np.asarray(sea[lo:lo + 4000])) for lo in range(0, ann.shape[0], 4000)])
    traj = cache.open_traj()
    y0 = np.transpose(np.asarray(traj[:, :, 0, :2], dtype=np.float64) / y_mean[:2], (1, 0, 2))
    n = feat.shape[0]; X = stack_ages(feat, agef); Y = np.concatenate([y0[:, j, :] for j in range(15)])
    fit = np.isin(np.tile(np.arange(n), 15), split.fit)
    train_models = [lgb.train(PARAMS, lgb.Dataset(X[fit], label=Y[fit, k]), num_boost_round=800) for k in (0, 1)]
    # competition: climate features and decoded y0 per scenario
    keys = shift.template_keys()
    feats_t, y0_t, rows_t = [], [], []
    for scenario in io.SCENARIOS:
        ann_t = np.load(cache.CACHE_DIR / f"test_{scenario}_annual.npy"); sea_t = np.load(cache.CACHE_DIR / f"test_{scenario}_season.npy")
        sf = site_features(ann_t, sea_t); feats_t.append(sf)
        raw = np.asarray(io.open_test(scenario)[1], dtype=np.float64)
        dec = 1.14114 * np.clip(raw[:, :, :2] - shift.FLOORS[:2], 0, None) / y_mean[:2]   # (nt, 15, 2)
        y0_t.append(dec)
    nt = feats_t[0].shape[0]
    Xt = [stack_ages(f, agef) for f in feats_t]                        # per scenario (15*nt, F), age-major
    Yt = [np.concatenate([d[:, j, :] for j in range(15)]) for d in y0_t]
    site_row = np.tile(np.arange(nt), 15)
    fold = (np.random.default_rng(7).random(nt) < 0.5)[site_row]
    test_rel = [np.zeros((15 * nt, 2)) for _ in io.SCENARIOS]
    for f in (False, True):
        Xf = np.concatenate([Xt[s][fold == f] for s in range(2)]); Yf = np.concatenate([Yt[s][fold == f] for s in range(2)])
        for k in (0, 1):
            m = lgb.train(PARAMS, lgb.Dataset(Xf, label=Yf[:, k]), num_boost_round=800)
            for s in range(2):
                test_rel[s][fold != f, k] = m.predict(Xt[s][fold != f], num_threads=6)
    train_rel = [np.stack([b.predict(Xt[s], num_threads=6) for b in train_models], 1) for s in range(2)]
    out_tr = np.zeros((len(keys), 2)); out_te = np.zeros((len(keys), 2))
    for s, scenario in enumerate(io.SCENARIOS):
        rows = (keys["scenario"] == scenario).to_numpy()
        site = keys.loc[rows, "site"].to_numpy(); aidx = np.searchsorted(np.asarray(io.SEED_AGES), keys.loc[rows, "age"].to_numpy())
        tr = train_rel[s].reshape(15, nt, 2).transpose(1, 0, 2); te = test_rel[s].reshape(15, nt, 2).transpose(1, 0, 2)
        out_tr[rows] = tr[site, aidx]; out_te[rows] = te[site, aidx]
        r2 = [1 - np.mean((test_rel[s][:, k] - Yt[s][:, k]) ** 2) / np.var(Yt[s][:, k]) for k in (0, 1)]
        r2t = [1 - np.mean((train_rel[s][:, k] - Yt[s][:, k]) ** 2) / np.var(Yt[s][:, k]) for k in (0, 1)]
        print(f"{scenario}: test-world relation R2 on held-out test sites h {r2[0]:.3f} a {r2[1]:.3f}; training relation on test y0 h {r2t[0]:.3f} a {r2t[1]:.3f}", flush=True)
    np.savez(ROOT / "work" / "world_shift.npz", train_rel=np.clip(out_tr, 0, None), test_rel=np.clip(out_te, 0, None))
    print("wrote work/world_shift.npz")


if __name__ == "__main__":
    main()
