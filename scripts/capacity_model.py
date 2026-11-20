#!/usr/bin/env python
"""Climate-supported state: what the window forcing sustains after 40 years, without the initial state.

LightGBM from the per-site forcing summaries, seasonal climatology, soil and seed age (no y0) to the training stand's
year-40 height and agb (scaled units), fitted on the fit sites. On the competition forcing it gives, per row, the state
a stand of that age reaches in that climate - the target of a relaxation if the test stands are out of equilibrium
with their window (the collapse cluster: both scenarios collapse alike, so the cause is not the window's weather).
Writes work/capacity_g40.npy (template order, scaled physical units; multiply by 0.876 for the truth's scale).

    <lightgbm venv>/bin/python scripts/capacity_model.py [--no-hourly]
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
import direct_gbdt  # noqa: E402
from direct_gbdt import site_features  # noqa: E402


def main() -> None:
    no_hourly = "--no-hourly" in sys.argv[1:]
    direct_gbdt.USE_HOURLY = not no_hourly
    suffix = "_nohourly" if no_hourly else ""
    stats = io.load_stats()
    y_mean = stats["y_mean"].astype(np.float64)
    agef = latent.age_features(stats)                      # (15, k)
    split = holdout.make_split(seed=42)
    ann = cache.open_x_annual(); sea = cache.open_x_season()
    feat = np.concatenate([site_features(np.asarray(ann[lo:lo + 4000]), np.asarray(sea[lo:lo + 4000]))
                           for lo in range(0, ann.shape[0], 4000)])          # (n, F)
    traj = cache.open_traj()
    y40 = np.transpose(np.asarray(traj[:, :, 40, :2], dtype=np.float64) / y_mean[:2], (1, 0, 2))  # (n, 15, 2)
    n = feat.shape[0]
    X = np.concatenate([np.concatenate([feat, np.repeat(agef[j][None], n, 0)], 1) for j in range(15)]).astype(np.float32)
    site_row = np.tile(np.arange(n), 15)
    fit = np.isin(site_row, split.fit); hold = ~fit
    Y = np.concatenate([y40[:, j, :] for j in range(15)])
    boosters = []
    for k, nm in enumerate(("height", "agb")):
        b = lgb.train(dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.7,
                           bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=6, verbose=-1, seed=42),
                      lgb.Dataset(X[fit], label=Y[fit, k]), num_boost_round=800)
        boosters.append(b)
        p = b.predict(X[hold], num_threads=6)
        print(f"{nm}: holdout MSE {np.mean((p - Y[hold, k])**2):.4f}  var {np.var(Y[hold, k]):.4f}  R2 {1 - np.mean((p - Y[hold, k])**2)/np.var(Y[hold, k]):.3f}", flush=True)
    keys = shift.template_keys()
    out = np.zeros((len(keys), 2))
    variants = {"early": np.zeros((len(keys), 2)), "late": np.zeros((len(keys), 2))}
    n_series = 8 if no_hourly else 12  # 8 monthly drivers (+ 4 daily-mean hourly groups)
    for scenario in io.SCENARIOS:
        ann_t = np.load(cache.CACHE_DIR / f"test_{scenario}_annual.npy"); sea_t = np.load(cache.CACHE_DIR / f"test_{scenario}_season.npy")
        sf = site_features(ann_t, sea_t); nt = sf.shape[0]
        Xt = np.concatenate([np.concatenate([sf, np.repeat(agef[j][None], nt, 0)], 1) for j in range(15)]).astype(np.float32)
        P = np.stack([b.predict(Xt, num_threads=6) for b in boosters], 1).reshape(15, nt, 2).transpose(1, 0, 2)  # (nt, 15, 2)
        rows = (keys["scenario"] == scenario).to_numpy()
        site = keys.loc[rows, "site"].to_numpy(); aidx = np.searchsorted(np.asarray(io.SEED_AGES), keys.loc[rows, "age"].to_numpy())
        out[rows] = P[site, aidx]
        # stationary-climate variants: every series' mean and decade means set to its first (early) or last (late)
        # decade mean, trend 0 (forcing_summaries layout per series: mean, sd, trend, four decade means)
        for tag, dec in (("early", 0), ("late", 3)):
            sfv = sf.copy()
            for j in range(n_series):
                base = 7 * j
                d = sf[:, base + 3 + dec]
                sfv[:, base] = d; sfv[:, base + 2] = 0.0
                for q in range(4):
                    sfv[:, base + 3 + q] = d
            Xv = np.concatenate([np.concatenate([sfv, np.repeat(agef[j][None], nt, 0)], 1) for j in range(15)]).astype(np.float32)
            Pv = np.stack([b.predict(Xv, num_threads=6) for b in boosters], 1).reshape(15, nt, 2).transpose(1, 0, 2)
            variants[tag][rows] = Pv[site, aidx]
    out = np.clip(out, 0, None)
    np.save(ROOT / "work" / f"capacity_g40{suffix}.npy", out)
    for tag, v in variants.items():
        np.save(ROOT / "work" / f"capacity_g40{suffix}_{tag}.npy", np.clip(v, 0, None))
        print(f"wrote work/capacity_g40{suffix}_{tag}.npy", v.mean(0))
    print(f"wrote work/capacity_g40{suffix}.npy", out.shape, out.mean(0))


if __name__ == "__main__":
    main()
