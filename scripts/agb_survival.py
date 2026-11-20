#!/usr/bin/env python
"""P(a stand keeps at least half its agb over the 40 years): classifier on training stands, applied to the test.

Training rows: mid/old seed ages (50-500) with slot-0 height > 3.3 m; target alive = year-40 height > 30% of the
slot-0 height. Features: the row's decoded state (h, agb, soil, lai), seed age, the site's whole slot-0 height and agb
profile (the chronosequence: a bare older stand is the strongest die-off cue) and the forcing summaries of
`direct_gbdt.py`. Site-held-out AUC on the holdout split. Writes work/agb_survival_p.npy (template order, NaN where
the row is not a tall stand).

    <lightgbm venv>/bin/python scripts/height_survival.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cgb import cache, holdout, io, shift  # noqa: E402
from direct_gbdt import site_features, DECODE, FLOORS  # noqa: E402

AGES = np.asarray(io.SEED_AGES, dtype=float)


def rows_for(feat, prof_h, prof_a, state, ages_idx):
    """feat (n, F); prof (n, 15); state (n, 15, 4); returns X for the given seed-age indices, stacked age-major."""
    out = []
    for j in ages_idx:
        out.append(np.column_stack([feat, prof_h, prof_a, state[:, j, :], np.full(len(feat), AGES[j]),
                                    np.full(len(feat), np.log(AGES[j])), prof_h[:, min(j + 1, 14)], prof_h[:, min(j + 2, 14)]]))
    return np.concatenate(out).astype(np.float32)


def main() -> None:
    split = holdout.make_split(seed=42)
    ann = cache.open_x_annual(); sea = cache.open_x_season()
    feat = np.concatenate([site_features(np.asarray(ann[lo:lo + 4000]), np.asarray(sea[lo:lo + 4000])) for lo in range(0, ann.shape[0], 4000)])
    traj = cache.open_traj()                                                  # (15, n, 41, 7) physical
    st0 = np.transpose(np.asarray(traj[:, :, 0, :4], dtype=np.float64), (1, 0, 2))   # (n, 15, 4)
    h40 = np.asarray(traj[:, :, 40, 0], dtype=np.float64).T; a40 = np.asarray(traj[:, :, 40, 1], dtype=np.float64).T
    n = feat.shape[0]
    ages_idx = [j for j, a in enumerate(AGES) if a >= 50]
    X = rows_for(feat, st0[:, :, 0], st0[:, :, 1], st0, ages_idx)
    h0r = np.concatenate([st0[:, j, 0] for j in ages_idx]); a0r = np.concatenate([st0[:, j, 1] for j in ages_idx]); y = np.concatenate([a40[:, j] for j in ages_idx]) > 0.5 * a0r
    site_r = np.tile(np.arange(n), len(ages_idx)); tall = a0r > 1.0
    fit = tall & np.isin(site_r, split.fit); hold = tall & np.isin(site_r, split.holdout)
    prm = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.7,
               bagging_fraction=0.8, bagging_freq=1, num_threads=6, verbose=-1, seed=42)
    mdl = lgb.train(prm, lgb.Dataset(X[fit], label=y[fit].astype(float)), 600)
    p = mdl.predict(X[hold], num_threads=6)
    from sklearn.metrics import roc_auc_score
    print(f"training tall stands: {fit.sum()} fit rows (alive {y[fit].mean():.2f}); holdout AUC {roc_auc_score(y[hold], p):.3f}, "
          f"Brier {np.mean((p - y[hold]) ** 2):.3f} vs base {np.mean((y[hold].mean() - y[hold]) ** 2):.3f}", flush=True)
    keys = shift.template_keys(); out = np.full(len(keys), np.nan)
    for scenario in io.SCENARIOS:
        ann_t = np.load(cache.CACHE_DIR / f"test_{scenario}_annual.npy"); sea_t = np.load(cache.CACHE_DIR / f"test_{scenario}_season.npy")
        sf = site_features(ann_t, sea_t)
        raw = np.asarray(io.open_test(scenario)[1], dtype=np.float64)[:, :, :4]
        phys = DECODE * np.clip(raw - FLOORS, 0, None)                         # (nt, 15, 4) physical
        Xt = rows_for(sf, phys[:, :, 0], phys[:, :, 1], phys, ages_idx)
        pt = mdl.predict(Xt, num_threads=6).reshape(len(ages_idx), -1)        # (n_ages, nt)
        tall_t = np.stack([phys[:, j, 1] for j in ages_idx]) > 1.0
        rows = (keys["scenario"] == scenario).to_numpy()
        site = keys.loc[rows, "site"].to_numpy(); aidx = np.searchsorted(AGES, keys.loc[rows, "age"].to_numpy())
        lut = {j: k for k, j in enumerate(ages_idx)}
        vals = np.full(rows.sum(), np.nan)
        for i, (s, j) in enumerate(zip(site, aidx)):
            if j in lut and tall_t[lut[j], s]:
                vals[i] = pt[lut[j], s]
        out[rows] = vals
    np.save(ROOT / "work" / "agb_survival_p.npy", out)
    print(f"wrote work/agb_survival_p.npy: {np.isfinite(out).sum()} tall test rows, mean P(alive) {np.nanmean(out):.3f}")


if __name__ == "__main__":
    main()
