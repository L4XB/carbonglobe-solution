#!/usr/bin/env python
"""Collapse detector on late windows: drought-index summaries -> 30-year agb change of acclimated old stands.

The slot-0 detector (``mort_gbdt.py``, position-free) was the most useful old-agb model direction on the board, but it
learned the training's first-decade transient together with the drought response. Under hypothesis H the test's old
stands are acclimated (physically ~stable on average), so the drought response of stands that are already ten years
into their run is the relevant one. Training rows: old seed ages, vegetated at slot 10, window slot 10 -> 40;
features: position-free summaries of six drought indices over forcing years [10, 40), the detector's climate normals
and soil, the scaled pools at slot 10, the age index; target: the scaled agb change over the window. Competition rows:
the same summaries over years [10, 40) of the scenario forcing and the decoded y0; the predicted 30-year change is
scaled by 4/3 to the forty-year horizon.

    <lightgbm venv>/bin/python scripts/late_detector.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cgb import cache, holdout, io, shift  # noqa: E402
from cgb import mortality as mort  # noqa: E402
from mort_detector import M_ALL, SEQ_KEYS, climatology  # noqa: E402
from mort_trigger import build_indices  # noqa: E402

OLD = (150, 200, 250, 300, 350, 400, 450, 500)


def window_summaries(ix: dict[str, np.ndarray], thr: dict, lo: int, hi: int) -> np.ndarray:
    cols = []
    for key in SEQ_KEYS:
        x = ix[key][:, lo:hi]
        fin = np.isfinite(x)
        x = np.where(fin, x, np.nanmin(np.where(fin, x, np.nan), axis=1, keepdims=True))
        q90, q99 = thr[key]
        cols += [x.max(axis=1), (x > q90).sum(axis=1), (x > q99).sum(axis=1), np.maximum(x - q90, 0).sum(axis=1)]
    return np.stack(cols, axis=1)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tropics", action="store_true", help="fit and apply only on sites with a mean air temperature above 20 C")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    y_mean = io.load_stats()["y_mean"].astype(np.float64)
    split = holdout.make_split(seed=42)
    mt = mort.open_monthly("train")
    n = mt.shape[0]
    ix_fit = build_indices(np.asarray(mt[split.fit[::4]]))
    thr = {k: (float(np.nanquantile(ix_fit[k], 0.9)), float(np.nanquantile(ix_fit[k], 0.99))) for k in SEQ_KEYS}
    print("building training features ...", flush=True)
    feats = []
    xa = cache.open_x_annual()
    for lo in range(0, n, 6000):
        m = np.asarray(mt[lo:lo + 6000])
        ix = build_indices(m)
        static = np.asarray(xa[lo:lo + 6000, 0, list(cache.STATIC_DRIVERS)], dtype=np.float64)
        feats.append(np.concatenate([window_summaries(ix, thr, 10, 40), climatology(m), static], axis=1))
    sx = np.concatenate(feats)
    traj = cache.open_traj()
    old = [io.SEED_AGES.index(a) for a in OLD]
    st = np.asarray(traj[old], dtype=np.float64)  # (8, n, 41, 7)
    s10 = (st[:, :, 10, :4] / y_mean[:4]).reshape(8 * n, 4)
    y = ((st[:, :, 40, 1] - st[:, :, 10, 1]) / y_mean[1]).reshape(-1)
    veg = (st[:, :, 10, 1] > mort.BARE_AGB).reshape(-1)
    X = np.concatenate([np.repeat(sx[None], 8, axis=0).reshape(8 * n, -1), s10, np.repeat(np.arange(8), n)[:, None]], axis=1)
    site_row = np.tile(np.arange(n), 8)
    fit = np.isin(site_row, split.fit) & veg
    if args.tropics:
        tmean = np.asarray(cache.open_x_season()[:, 0, :], dtype=np.float64).mean(1) - 273.15
        trop = np.tile(tmean > 20.0, 8)
        fit &= trop
        print(f"tropics only: {fit.sum():,} fit rows", flush=True)
    params = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42, num_threads=8)
    b = lgb.train(params, lgb.Dataset(X[fit], label=y[fit]), num_boost_round=800)
    pub = np.isin(site_row, split.public) & veg & (trop if args.tropics else True)
    p = b.predict(X[pub])
    print(f"holdout public (old, slot 10 -> 40): MSE {np.mean((p - y[pub]) ** 2):.4f} vs persistence {np.mean(y[pub] ** 2):.4f}; explained {1 - np.mean((p - y[pub]) ** 2) / np.mean((y[pub] - y[pub].mean()) ** 2):.3f}", flush=True)
    frames = {s: io.open_test(s)[1].astype(np.float64) for s in io.SCENARIOS}
    floors = np.concatenate([fr.reshape(-1, 7) for fr in frames.values()]).min(axis=0)
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    keys = shift.template_keys()
    sc_k, site_k, age_k = keys["scenario"].to_numpy(), keys["site"].to_numpy(), keys["age"].to_numpy()
    F = R.copy()
    for sc in io.SCENARIOS:
        m_t = np.asarray(mort.open_monthly(sc))
        ix = build_indices(m_t)
        xt = np.load(cache.CACHE_DIR / f"test_{sc}_annual.npy", mmap_mode="r")
        static = np.asarray(xt[:, 0, list(cache.STATIC_DRIVERS)], dtype=np.float64)
        sxt = np.concatenate([window_summaries(ix, thr, 10, 40), climatology(m_t), static], axis=1)
        raw0 = M_ALL * (frames[sc] - floors)
        nt = raw0.shape[0]
        s0 = (np.transpose(raw0[:, old, :4], (1, 0, 2)) / y_mean[:4]).reshape(8 * nt, 4)
        Xt = np.concatenate([np.repeat(sxt[None], 8, axis=0).reshape(8 * nt, -1), s0, np.repeat(np.arange(8), nt)[:, None]], axis=1)
        vt = (np.transpose(raw0[:, old, 1]) > mort.BARE_AGB).reshape(-1)
        if args.tropics:
            tt = np.load(cache.CACHE_DIR / f"test_{sc}_season.npy")[:, 0, :].mean(1) - 273.15
            vt &= np.tile(tt > 20.0, 8)
        pt = np.zeros(len(vt)); pt[vt] = b.predict(Xt[vt]) * (40.0 / 30.0)
        pt = pt.reshape(8, nt)
        print(f"{sc}: predicted 40-year old agb change mean {pt[pt != 0].mean():+.4f} sd {pt[pt != 0].std():.4f}", flush=True)
        msk = sc_k == sc
        for i, a in enumerate(OLD):
            r = np.where(msk & (age_k == a))[0]
            F[r, 1] = np.clip(R[r, 1] + pt[i, site_k[r]], 0, None)
    out = ROOT / "results" / f"latedetprobe_old_agb{args.tag}.csv"
    shift.write_predictions(F, out)
    print(f"wrote {out}; nothing submitted")


if __name__ == "__main__":
    main()
