#!/usr/bin/env python
"""The forty-year forcing trend's effect on growth, from gbdt2, applied to the test-native age-shift level.

The age-shift predictor (the site's own seed-age curve at age + 40) carries the test's growth conditions of the
spin-up but not the window's climate change; gbdt2 carries forcing effects learned in training but transfers its
level badly. gbdt2 is run twice on the competition rows: with the real forcing summaries and with the trend
removed (trend features 0, the four decade means set to the 40-year mean). The ratio of the two predicted
changes is the trend effect; it rescales the age-shift change: T = R + (A - R) x ratio (ratio clipped to [0.5, 1.5]).

    <lightgbm venv>/bin/python scripts/trend_effect.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cgb import cache, io, latent, shift  # noqa: E402
import direct_gbdt  # noqa: E402

DECODE = np.array([1.14114, 1.14114, 1.14114, 1.0])


def main() -> None:
    direct_gbdt.USE_HOURLY = True
    stats = io.load_stats(); y_mean = stats["y_mean"].astype(np.float64); agef = latent.age_features(stats)
    d = io.ROOT / "work" / "models" / "direct_gbdt2"
    boosters = [lgb.Booster(model_file=str(d / f"{k}.txt")) for k in ("height", "agb")]
    keys = shift.template_keys()
    R = shift.read_predictions(io.ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    A = shift.read_predictions(io.ROOT / "results" / "ageshift_m114.csv").astype(np.float64)
    G = np.zeros_like(R); Gd = np.zeros_like(R)
    ages = np.array(io.SEED_AGES)
    for scenario in io.SCENARIOS:
        ann_t = np.load(cache.CACHE_DIR / f"test_{scenario}_annual.npy"); sea_t = np.load(cache.CACHE_DIR / f"test_{scenario}_season.npy")
        y0 = io.open_test(scenario)[1].astype(np.float64)[:, :, :4]
        u0t = DECODE * (y0 - shift.FLOORS[:4]) / y_mean[:4]
        sf = direct_gbdt.site_features(ann_t, sea_t)
        n_series = 12
        sfd = sf.copy()
        for s in range(n_series):
            base = s * 7
            sfd[:, base + 2] = 0.0                      # trend
            sfd[:, base + 3: base + 7] = sf[:, [base]]  # decade means -> the 40-year mean
        for out, feats in ((G, sf), (Gd, sfd)):
            Xt = direct_gbdt.rows(feats, u0t, agef)
            ch = np.stack([b.predict(Xt) for b in boosters], axis=1)  # (15 n, 2) age-major
            n = ann_t.shape[0]
            ch = ch.reshape(15, n, 2).transpose(1, 0, 2)
            m = keys["scenario"].to_numpy() == scenario
            site = keys["site"].to_numpy()[m]; ai = np.searchsorted(ages, keys["age"].to_numpy()[m])
            out[m] = R[m] + ch[site, ai]
    masks = shift.band_masks(keys["age"].to_numpy())
    for b in ("young", "mid"):
        m = masks[b] & (R[:, 0] * 10.90645 >= 1.0)
        for c in (0, 1):
            dg, dd = G[m, c] - R[m, c], Gd[m, c] - R[m, c]
            print(f"{b} {'ha'[c]}: gbdt2 change real forcing {dg.mean():+.4f}, detrended {dd.mean():+.4f}; trend effect on the change: mean ratio {np.mean(dg)/np.mean(dd):.3f}, sd of per-row difference {np.std(dg - dd):.4f}")
    young = masks["young"] & (R[:, 0] * 10.90645 >= 1.0)
    F = R.copy()
    for c in (0, 1):
        dg, dd = G[young, c] - R[young, c], Gd[young, c] - R[young, c]
        ratio = np.clip((dg + 0.02) / (dd + 0.02), 0.5, 1.5)
        F[young, c] = R[young, c] + (A[young, c] - R[young, c]) * ratio
    shift.write_predictions(np.clip(F, 0, None), io.ROOT / "results" / "bandprobe_ageshifttrend_young.csv")
    L = R.copy(); L[young] = np.clip(F[young] - A[young] + R[young], 0, None)
    shift.write_predictions(L, io.ROOT / "results" / "ageshifttrendleg_young.csv")
    print("wrote results/bandprobe_ageshifttrend_young.csv and the leg file; nothing submitted")


if __name__ == "__main__":
    main()
