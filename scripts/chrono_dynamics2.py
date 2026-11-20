#!/usr/bin/env python
"""Test-native dynamics v2: chrono_dynamics.py with the full forcing summaries of direct_gbdt.py as site context
and the states of the two next-younger seed ages (no leakage: every target is an older seed age).

Test-native dynamics: learn the chronosequence step from the test's own seed-age profiles.

Every test site carries 15 separate stands of different ages grown under the same (spin-up) forcing, so a pair of
seed ages (a, a + g) at one site is a sample of "what a stand of age a looks like g years later" under that site's
climate, with none of the training set's spin-up/run transient. A LightGBM fitted on such pairs from OTHER sites
(cross-fitted by site) gives the expected g-year change of a stand from its own state and the site's climate; the
year-40 prediction is 0.876 x (physical state + predicted change at g = --gap), the raw-scale truth of hypothesis H.

Unlike the age-shift predictor (the site's own next-older stand), this is smooth in the site's profile: it keeps
the expected aging trend and drops the site-specific jaggedness of the profile.

    python scripts/chrono_dynamics.py --gap 46 --tag chronodyn1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io, shift, submission  # noqa: E402

AGES = np.array(io.SEED_AGES, dtype=np.float64)


def load_states() -> np.ndarray:
    """(2, 6171, 15, 7) test y0 with height/agb decoded to physical units, other variables raw."""
    floors = shift.measured_floors()
    out = []
    for s in io.SCENARIOS:
        y0 = np.asarray(io.open_test(s)[1], dtype=np.float64)
        y = y0.copy()
        for v in (0, 1):
            y[:, :, v] = shift.M_HEIGHT * np.clip(y0[:, :, v] - floors[v], 0.0, None)
        out.append(y)
    return np.stack(out)


def site_context_full() -> np.ndarray:
    """(2, 6171, F) direct_gbdt.py site features (forcing summaries, seasonal climatology, soil) per scenario."""
    from cgb import cache
    sys.path.insert(0, str(ROOT / "scripts"))
    from direct_gbdt import site_features
    out = []
    for s in io.SCENARIOS:
        ann = np.load(cache.CACHE_DIR / f"test_{s}_annual.npy"); sea = np.load(cache.CACHE_DIR / f"test_{s}_season.npy")
        out.append(site_features(ann, sea).astype(np.float64))
    return np.stack(out)


def site_context() -> np.ndarray:
    """(2, 6171, k) climate context per scenario: early-window means and seasonality of T and P, plus statics."""
    from cgb import cache
    feats = []
    for s in io.SCENARIOS:
        x = io.open_test(s)[0]
        t = np.asarray(x[:, :10, :, 0], dtype=np.float64)   # (site, 10, 12) first decade
        p = np.asarray(x[:, :10, :, 1], dtype=np.float64)
        tm, pm = t.mean(axis=1), p.mean(axis=1)              # monthly climatology (site, 12)
        st = np.asarray(x[:, 0, 0, list(cache.STATIC_DRIVERS)], dtype=np.float64)
        tall = np.asarray(x[:, :, :, 0], dtype=np.float64).mean(axis=2)
        pall = np.asarray(x[:, :, :, 1], dtype=np.float64).mean(axis=2)
        feats.append(np.column_stack([tm.mean(1), tm.max(1) - tm.min(1), pm.mean(1), pm.min(1), pm.max(1),
                                      tall[:, 30:].mean(1) - tall[:, :10].mean(1),
                                      pall[:, 30:].mean(1) - pall[:, :10].mean(1), st]))
    return np.stack(feats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gap", type=float, default=46.0)
    ap.add_argument("--max-gap", type=float, default=100.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    import lightgbm as lgb

    Y = load_states()                       # (2, n, 15, 7)
    ctx = np.concatenate([site_context(), site_context_full()], axis=2)   # (2, n, k)
    n = Y.shape[1]
    rows, tgt, site_of = [], [], []
    for s in range(2):
        for i in range(15):
            for j in range(i + 1, 15):
                g = AGES[j] - AGES[i]
                if g < 10 or g > args.max_gap:
                    continue
                f = [Y[s, :, i, :], Y[s, :, max(i - 1, 0), :2], Y[s, :, max(i - 2, 0), :2], np.full((n, 1), AGES[i]),
                     np.full((n, 1), g), np.full((n, 1), s)]
                if not args.no_context:
                    f.append(ctx[s])
                rows.append(np.column_stack(f))
                tgt.append(Y[s, :, j, :2] - Y[s, :, i, :2])
                site_of.append(np.arange(n))
    X = np.concatenate(rows); T = np.concatenate(tgt); S = np.concatenate(site_of)
    print(f"{len(X)} pairs, {X.shape[1]} features")
    fold = np.random.default_rng(42).integers(0, args.folds, size=n)
    params = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=args.threads, verbose=-1, seed=42)
    pred = np.zeros((2, n, 15, 2))
    cv = {0: [], 1: []}
    for k in range(args.folds):
        tr = fold[S] != k
        held = np.where(fold == k)[0]
        for v in (0, 1):
            m = lgb.train(params, lgb.Dataset(X[tr], T[tr, v]), num_boost_round=600)
            te = ~tr
            cv[v].append((float(np.mean((m.predict(X[te]) - T[te, v]) ** 2)), float(np.mean(T[te, v] ** 2))))
            for s in range(2):
                for i in range(15):
                    f = [Y[s, held, i, :], Y[s, held, max(i - 1, 0), :2], Y[s, held, max(i - 2, 0), :2],
                         np.full((len(held), 1), AGES[i]), np.full((len(held), 1), args.gap), np.full((len(held), 1), s)]
                    if not args.no_context:
                        f.append(ctx[s, held])
                    pred[s, held, i, v] = m.predict(np.column_stack(f))
        print(f"fold {k} done", flush=True)
    for v, name in ((0, "height"), (1, "agb")):
        mse = np.mean([a for a, _ in cv[v]]); base = np.mean([b for _, b in cv[v]])
        print(f"{name}: held-out pair MSE {mse:.4f} vs persistence {base:.4f} (R2 of the change {1 - mse / base:.3f})")
    # year-40 prediction on the raw scale of hypothesis H, template order (scenario, site, age)
    phys = Y[:, :, :, :2] + pred
    phys = np.clip(phys, 0.0, None)
    phys[Y[:, :, :, :2] <= 0] = 0.0          # a bare stand stays bare
    div = submission.divisors()
    out = np.empty((2 * n * 15, 2))
    out[:, 0] = (0.876 * phys[..., 0] / div["height"]).reshape(-1)
    out[:, 1] = (0.876 * phys[..., 1] / div["agb"]).reshape(-1)
    path = ROOT / "results" / f"{args.tag}_g{int(args.gap)}.csv"
    shift.write_predictions(out, path)
    np.save(ROOT / "work" / f"{args.tag}_change.npy", pred)
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
