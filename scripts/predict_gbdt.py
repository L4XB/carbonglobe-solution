#!/usr/bin/env python
"""Predict-only path for a trained direct GBDT (``direct_gbdt.py`` boosters) with a chosen y0 decode.

    <lgb venv>/bin/python scripts/predict_gbdt.py --tag gbdt2 --m-agb 1.0 --out results/direct_gbdt2_agb100.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cgb import cache, io, shift  # noqa: E402
try:
    from cgb import latent  # noqa: E402
except ImportError:  # an interpreter with LightGBM but no torch (as direct_gbdt.py does)
    from cgb import latent_np as latent  # noqa: E402
from cgb import mortality as mort  # noqa: E402
import direct_gbdt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gbdt2")
    ap.add_argument("--m-height", type=float, default=1.14114)
    ap.add_argument("--m-agb", type=float, default=1.14114)
    ap.add_argument("--m-soil", type=float, default=1.14114)
    ap.add_argument("--m-lai", type=float, default=1.0)
    ap.add_argument("--drought", action="store_true", help="the checkpoint was trained with drought features")
    ap.add_argument("--no-hourly", action="store_true", help="the checkpoint was trained without the hourly groups")
    ap.add_argument("--h-denoise", type=float, default=0.0,
                    help="replace the decoded initial height of mid+old rows by h0 + k (h_allo - h0) with h_allo from "
                         "work/cache/allometry_test_scaled.npy (the half relaxation the board measured is k ~ 0.44)")
    ap.add_argument("--age1-clim", action="store_true",
                    help="replace the age-1 state by the climate-predicted training-like seedling (work/cache/y0_clim_age1_*.npy)")
    ap.add_argument("--cache-tag", default="", help="read work/cache/test_<s>_annual_<tag>.npy / _season_<tag>.npy (scripts/repair_test_cache.py)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    direct_gbdt.USE_HOURLY = not args.no_hourly
    stats = io.load_stats(); y_mean = stats["y_mean"].astype(np.float64); agef = latent.age_features(stats)
    d = io.ROOT / "work" / "models" / f"direct_{args.tag}"
    boosters = [lgb.Booster(model_file=str(d / f"{k}.txt")) for k in ("height", "agb")]
    thr = direct_gbdt.drought_thresholds() if args.drought else None
    decode = np.array([args.m_height, args.m_agb, args.m_soil, args.m_lai])
    template = np.loadtxt(io.CACHE / "sample_submission.csv", delimiter=",", skiprows=1, dtype=str, usecols=0)
    ids, hs, as_ = [], [], []
    for scenario in io.SCENARIOS:
        sfx = f"_{args.cache_tag}" if args.cache_tag else ""
        ann_t = np.load(cache.CACHE_DIR / f"test_{scenario}_annual{sfx}.npy"); sea_t = np.load(cache.CACHE_DIR / f"test_{scenario}_season{sfx}.npy")
        y0 = io.open_test(scenario)[1].astype(np.float64)[:, :, :4]
        raw0 = decode * (y0 - shift.FLOORS[:4]); u0t = raw0 / y_mean[:4]
        if args.h_denoise:
            A = np.load(io.ROOT / "work" / "cache" / "allometry_test_scaled.npy")
            keys = shift.template_keys(); mrow = (keys["scenario"].to_numpy() == scenario)
            site_r = keys["site"].to_numpy()[mrow]; age_r = np.searchsorted(np.array(io.SEED_AGES), keys["age"].to_numpy()[mrow])
            ah = np.zeros(u0t.shape[:2]); ah[site_r, age_r] = A[mrow, 0]
            mo = np.array(io.SEED_AGES) >= 50
            u0t[:, mo, 0] = u0t[:, mo, 0] + args.h_denoise * (ah[:, mo] - u0t[:, mo, 0])
        if args.age1_clim:
            u0t[:, 0, :] = np.load(io.ROOT / "work" / "cache" / f"y0_clim_age1_{scenario}.npy") / y_mean[:4]
        sf = direct_gbdt.site_features(ann_t, sea_t)
        if args.drought:
            sf = np.concatenate([sf, direct_gbdt.drought_site_features(mort.open_monthly(scenario), thr)], axis=1)
        Xt = direct_gbdt.rows(sf, u0t, agef)
        ch = np.stack([b.predict(Xt) for b in boosters], axis=1)
        n = ann_t.shape[0]
        p40 = np.clip(Xt[:, -19:-15][:, :2] + ch, 0, None)
        p40[Xt[:, -19] * y_mean[0] < 1.0] = 0.0
        p40 = p40.reshape(15, n, 2).transpose(1, 0, 2)
        for site in range(n):
            for age in io.SEED_AGES:
                ids.append(f"{scenario}_{site}_{age}")
        hs.append(p40[..., 0].reshape(-1)); as_.append(p40[..., 1].reshape(-1))
        print(f"{scenario}: mean change vs own y0 h {np.mean(p40[..., 0] - u0t[..., 0]):+.4f} a {np.mean(p40[..., 1] - u0t[..., 1]):+.4f}; agb mean {p40[..., 1].mean():.4f}", flush=True)
    ids = np.array(ids); assert np.array_equal(ids, template)
    pred = np.stack([np.concatenate(hs), np.concatenate(as_)], 1)
    shift.write_predictions(pred, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
