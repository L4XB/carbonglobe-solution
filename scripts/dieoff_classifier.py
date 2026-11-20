#!/usr/bin/env python
"""Die-off timing inside a chronosequence bracket.

A test row (site, seed age a) is "ambiguous" when its year-40 age a+40 falls strictly between two seed ages
a1 < a+40 < a2 of the site's own y0 profile with a forest at a1 (height > 3.3 m) and bare ground at a2
(height < 1.3 m): the stand dies somewhere in (a1, a2] and the row is either alive or dead at year 40.
The training set has the same brackets in its year-0 profiles and the answer in its trajectories. This
script fits P(alive at year 40) on the training rows (profile, bracket position, climate, soil) with a
site-held-out check, and writes the test probabilities to work/cache/dieoff_p_test.npy (aligned with the
submission template; NaN off the ambiguous rows).

    <lgb venv>/bin/python scripts/dieoff_classifier.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import cache, holdout, io, latent, shift  # noqa: E402

AGES = np.array(io.SEED_AGES)
SOIL = [129, 130, 131, 132, 133, 134, 135]
ALIVE, DEAD = 3.3, 1.3  # metres
M = 1.14114


def ambiguous(h0: np.ndarray) -> list[tuple[int, int, int, np.ndarray]]:
    """(seed-age index, j1, j2, site mask) for every bracketed age; h0 is (15, S) physical height at year 0."""
    out = []
    for ai, a in enumerate(AGES):
        t = a + 40
        j2 = int(np.searchsorted(AGES, t))
        if j2 >= 15 or AGES[j2] == t:
            continue
        j1 = j2 - 1
        out.append((ai, j1, j2, (h0[j1] > ALIVE) & (h0[j2] < DEAD)))
    return out


def features(prof: np.ndarray, ai: int, j1: int, j2: int, cl: np.ndarray) -> np.ndarray:
    """prof (S, 15, 3) physical height/agb/lai at year 0; cl (S, C) climate + soil."""
    a = AGES[ai]
    pos = (a + 40 - AGES[j1]) / (AGES[j2] - AGES[j1])
    S = prof.shape[0]
    h1 = prof[:, j1, 0]
    h1m = prof[:, max(j1 - 1, 0), 0]
    cols = [np.full((S, 1), a, np.float32), np.full((S, 1), pos, np.float32), np.full((S, 1), AGES[j1], np.float32),
            np.full((S, 1), AGES[j2] - AGES[j1], np.float32),
            prof[:, ai, :], prof[:, j1, :], prof[:, j2, :], (h1 - h1m)[:, None],
            prof[:, :, 0], prof[:, :, 1], cl]
    return np.concatenate(cols, 1).astype(np.float32)


def main() -> None:
    traj = cache.open_traj()
    h0 = np.asarray(traj[:, :, 0, 0], dtype=np.float32)
    S = h0.shape[1]
    prof = np.transpose(np.asarray(traj[:, :, 0, [0, 1, 3]], dtype=np.float32), (1, 0, 2))  # (S, 15, 3)
    sea = cache.open_x_season(); ann = cache.open_x_annual()
    cl = np.concatenate([latent.seasonal_block(sea)[:, :24], np.asarray(ann[:, 0, SOIL])], 1).astype(np.float32)
    h40 = np.asarray(traj[:, :, 40, 0], dtype=np.float32)
    X, y, site, age = [], [], [], []
    for ai, j1, j2, m in ambiguous(h0):
        idx = np.where(m)[0]
        lab = h40[ai, idx]
        keep = (lab > ALIVE) | (lab < DEAD)
        idx = idx[keep]
        X.append(features(prof[idx], ai, j1, j2, cl[idx])); y.append((h40[ai, idx] > ALIVE).astype(np.float32))
        site.append(idx); age.append(np.full(len(idx), AGES[ai]))
    X = np.concatenate(X); y = np.concatenate(y); site = np.concatenate(site); age = np.concatenate(age)
    split = holdout.make_split(seed=42)
    te = np.isin(site, split.holdout)
    print(f"training ambiguous rows {len(y)} (alive {y.mean():.2f}); holdout {te.sum()}", flush=True)
    params = dict(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=40, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=6)
    clf = lgb.LGBMClassifier(**params).fit(X[~te], y[~te])
    p = clf.predict_proba(X[te])[:, 1]
    from sklearn.metrics import roc_auc_score
    # baselines: uniform-position prior and the per-age training base rate
    pos = X[te, 1]
    base_age = np.array([y[~te][age[~te] == a].mean() for a in age[te]])
    for name, q in (("uniform position", 1 - pos), ("per-age base rate", base_age), ("classifier", p)):
        print(f"  {name:18s} Brier {np.mean((q - y[te]) ** 2):.4f}  AUC {roc_auc_score(y[te], q):.3f}")
    # cost in height units: squared error x h1^2 relative to the hedge
    h1 = X[te, 4 + 3]  # prof[:, j1, 0]
    for name, q in (("uniform position", 1 - pos), ("per-age base rate", base_age), ("classifier", p)):
        print(f"  {name:18s} E[(q - alive)^2 h1^2] {np.mean((q - y[te]) ** 2 * h1 ** 2):.2f} m^2")
    clf_all = lgb.LGBMClassifier(**params).fit(X, y)
    # test
    keys = shift.template_keys()
    sc = keys["scenario"].to_numpy(); st = keys["site"].to_numpy(); ag = keys["age"].to_numpy()
    p_test = np.full(len(keys), np.nan)
    for s in io.SCENARIOS:
        y0 = io.open_test(s)[1].astype(np.float64)  # (S, 15, 7) raw
        dec = np.stack([M * (y0[:, :, k] - shift.FLOORS[k]) for k in (0, 1, 3)], 2).astype(np.float32)
        dec[:, :, 2] = (y0[:, :, 3] - shift.FLOORS[3])  # lai is not scaled
        xt = np.load(ROOT / "work" / "cache" / f"test_{s}_annual.npy", mmap_mode="r")
        clt = np.concatenate([latent.seasonal_block(np.load(ROOT / "work" / "cache" / f"test_{s}_season.npy"))[:, :24],
                              np.asarray(xt[:, 0, SOIL])], 1).astype(np.float32)
        h0t = dec[:, :, 0].T
        for ai, j1, j2, m in ambiguous(h0t):
            idx = np.where(m)[0]
            if len(idx) == 0:
                continue
            q = clf_all.predict_proba(features(dec[idx], ai, j1, j2, clt[idx]))[:, 1]
            rows = np.where((sc == s) & (ag == AGES[ai]))[0]
            pos_in = np.searchsorted(st[rows], idx)
            p_test[rows[pos_in]] = q
    np.save(ROOT / "work" / "cache" / "dieoff_p_test.npy", p_test)
    ok = ~np.isnan(p_test)
    print(f"test ambiguous rows {ok.sum()}, mean P(alive) {np.nanmean(p_test):.3f}")
    for a in AGES:
        m = ok & (ag == a)
        if m.sum():
            print(f"  age {a:3d}: n {m.sum():5d}  mean P(alive) {p_test[m].mean():.2f}")


if __name__ == "__main__":
    main()
