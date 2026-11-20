#!/usr/bin/env python
"""Anchor + delta: the site's own chronosequence 40 years ahead plus a training-learned window effect.

ED's 15 seed-age states are one chronosequence under the spin-up forcing (ED v3.0 protocol: bare ground under a
climatology with constant CO2). The year-40 state of a stand of age a is then y0(a + 40) of its own profile (the
anchor) plus the effect of the window forcing differing from the spin-up forcing (interannual weather, trend, CO2).
That second term is learned on the training runs, where both are known: target = Y40(a) - A_train(a + 40) (the
training run's departure from the training chronosequence), features = forcing summaries, seasonal climatology, soil,
the state at a, the profile at the neighbouring seed ages and the seed age. On the test: T = 0.876 x (A_test(a + 40) +
predicted departure). The anchor is never refitted (a fitted anchor transfers badly, Agnuxo1 0.094 -> 0.137).

    <lightgbm venv>/bin/python scripts/anchor_delta.py --tag anchordelta1 [--gap 40]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cgb import cache, holdout, io, shift  # noqa: E402
from direct_gbdt import site_features, DECODE, FLOORS  # noqa: E402

AGES = np.asarray(io.SEED_AGES, dtype=float)


def anchor(prof: np.ndarray, gap: float) -> np.ndarray:
    """prof (n, 15) -> (n, 15) profile linearly interpolated at age + gap (flat beyond 500)."""
    return np.stack([np.interp(AGES + gap, AGES, prof[i]) for i in range(prof.shape[0])])


def features(sf: np.ndarray, st: np.ndarray, j: int) -> np.ndarray:
    """sf (n, F) site features; st (n, 15, 4) physical state; j seed-age index."""
    n = sf.shape[0]
    nb = [st[:, max(j - 1, 0), :2], st[:, min(j + 1, 14), :2], st[:, min(j + 2, 14), :2]]
    return np.column_stack([sf, st[:, j, :], *nb, np.full(n, AGES[j]), np.full(n, np.log(AGES[j]))]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="anchordelta1")
    ap.add_argument("--gap", type=float, default=40.0)
    ap.add_argument("--rounds", type=int, default=800)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    split = holdout.make_split(seed=42)
    ann = cache.open_x_annual(); sea = cache.open_x_season()
    feat = np.concatenate([site_features(np.asarray(ann[lo:lo + 4000]), np.asarray(sea[lo:lo + 4000])) for lo in range(0, ann.shape[0], 4000)])
    traj = cache.open_traj()
    st0 = np.transpose(np.asarray(traj[:, :, 0, :4], dtype=np.float64), (1, 0, 2))      # (n, 15, 4) physical
    y40 = np.transpose(np.asarray(traj[:, :, 40, :2], dtype=np.float64), (1, 0, 2))     # (n, 15, 2)
    n = feat.shape[0]
    anc = np.stack([anchor(st0[:, :, v], args.gap) for v in (0, 1)], 2)                  # (n, 15, 2)
    X = np.concatenate([features(feat, st0, j) for j in range(15)])
    A = np.concatenate([anc[:, j, :] for j in range(15)]); Y = np.concatenate([y40[:, j, :] for j in range(15)])
    S0 = np.concatenate([st0[:, j, :2] for j in range(15)])
    site_r = np.tile(np.arange(n), 15)
    fit = np.isin(site_r, split.fit); hold = np.isin(site_r, split.holdout)
    prm = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.7,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=args.threads, verbose=-1, seed=42)
    div = np.array([10.90645, 3.39567])
    models = []
    for v, nm in ((0, "height"), (1, "agb")):
        m = lgb.train(prm, lgb.Dataset(X[fit], label=(Y[fit, v] - A[fit, v])), args.rounds)
        models.append(m)
        p = A[hold, v] + m.predict(X[hold], num_threads=args.threads)
        p = np.clip(p, 0, None)
        e_ad = np.mean(((p - Y[hold, v]) / div[v]) ** 2); e_a = np.mean(((A[hold, v] - Y[hold, v]) / div[v]) ** 2)
        e_p = np.mean(((S0[hold, v] - Y[hold, v]) / div[v]) ** 2)
        print(f"{nm}: holdout MSE (scaled) anchor+delta {e_ad:.4f} | anchor only {e_a:.4f} | persistence {e_p:.4f}", flush=True)
    keys = shift.template_keys(); out = np.zeros((len(keys), 2))
    for scenario in io.SCENARIOS:
        ann_t = np.load(cache.CACHE_DIR / f"test_{scenario}_annual.npy"); sea_t = np.load(cache.CACHE_DIR / f"test_{scenario}_season.npy")
        sf = site_features(ann_t, sea_t)
        raw = np.asarray(io.open_test(scenario)[1], dtype=np.float64)[:, :, :4]
        phys = DECODE * np.clip(raw - FLOORS, 0, None)                                        # (nt, 15, 4) physical
        anc_t = np.stack([anchor(phys[:, :, v], args.gap) for v in (0, 1)], 2)
        nt = phys.shape[0]
        Xt = np.concatenate([features(sf, phys, j) for j in range(15)])
        At = np.concatenate([anc_t[:, j, :] for j in range(15)])
        Pt = np.stack([np.clip(At[:, v] + models[v].predict(Xt, num_threads=args.threads), 0, None) for v in (0, 1)], 1)
        Pt[np.concatenate([phys[:, j, 0] for j in range(15)]) <= 0] = 0.0                     # bare stays bare
        Pt = Pt.reshape(15, nt, 2).transpose(1, 0, 2)
        rows = (keys["scenario"] == scenario).to_numpy()
        site = keys.loc[rows, "site"].to_numpy(); aidx = np.searchsorted(AGES, keys.loc[rows, "age"].to_numpy())
        out[rows] = 0.876 * Pt[site, aidx] / div
    path = ROOT / "results" / f"{args.tag}_m114.csv"
    shift.write_predictions(np.clip(out, 0, None), path)
    print(f"wrote {path.relative_to(ROOT)}; test means h {out[:, 0].mean():.3f} a {out[:, 1].mean():.3f}")


if __name__ == "__main__":
    main()
