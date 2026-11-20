#!/usr/bin/env python
"""Expected-weather predictions of a saved direct GBDT (default gbdt2) on the test.

Same bootstrap weather realizations as `predict_latent.py --shuffle-years k --bootstrap` (per site: linear trend of every
annual driver kept, residual years drawn with replacement; rng seed 2k + scenario index). Writes
work/shuffle/<tag>_boot_mean.npy (template order, scaled units) and work/shuffle/<tag>_actual.csv.

    <lgb venv>/bin/python scripts/gbdt_weather_realizations.py --tag gbdt2 --k 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cgb import cache, io, latent, shift  # noqa: E402
import direct_gbdt as dg  # noqa: E402


def bootstrap(ann: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    yrs = (np.arange(ann.shape[1]) - (ann.shape[1] - 1) / 2).astype(np.float64)
    a64 = ann.astype(np.float64)
    slope = np.einsum("t,ntd->nd", yrs, a64) / float((yrs ** 2).sum())
    trend = a64.mean(axis=1, keepdims=True) + yrs[None, :, None] * slope[:, None, :]
    idx = rng.integers(0, ann.shape[1], size=ann.shape[:2])
    return (trend + np.take_along_axis(a64 - trend, idx[:, :, None], axis=1)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="gbdt2")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--no-hourly", action="store_true")
    args = ap.parse_args()
    dg.USE_HOURLY = not args.no_hourly
    stats = io.load_stats(); y_mean = stats["y_mean"].astype(np.float64); agef = latent.age_features(stats)
    boosters = [lgb.Booster(model_file=str(ROOT / "work" / "models" / f"direct_{args.tag}" / f"{k}.txt")) for k in ("height", "agb")]
    keys = shift.template_keys()
    out_act = np.zeros((len(keys), 2)); out_mean = np.zeros((len(keys), 2))
    for s in io.SCENARIOS:
        ann_t = np.load(cache.CACHE_DIR / f"test_{s}_annual.npy"); sea_t = np.load(cache.CACHE_DIR / f"test_{s}_season.npy")
        y0 = io.open_test(s)[1].astype(np.float64)[:, :, :4]
        u0t = dg.DECODE * (y0 - dg.FLOORS) / y_mean[:4]

        def pred(ann: np.ndarray) -> np.ndarray:
            Xt = dg.rows(dg.site_features(ann, sea_t), u0t, agef)
            u0c = Xt[:, -19:-15]
            ch = np.stack([b.predict(Xt, num_threads=6) for b in boosters], axis=1)
            p = np.clip(u0c[:, :2] + ch, 0, None)
            p[u0c[:, 0] * y_mean[0] < 1.0] = 0.0
            return p.reshape(15, ann.shape[0], 2).transpose(1, 0, 2)  # (n, 15, 2)

        act = pred(ann_t)
        acc = np.zeros_like(act)
        for k in range(args.k):
            rng = np.random.default_rng(k * 2 + io.SCENARIOS.index(s))
            acc += pred(bootstrap(ann_t, rng)) / args.k
            print(f"{s}: realization {k} done", flush=True)
        m = keys["scenario"].to_numpy() == s
        st = keys["site"].to_numpy()[m]; ai = np.searchsorted(np.array(io.SEED_AGES), keys["age"].to_numpy()[m])
        out_act[m] = act[st, ai]; out_mean[m] = acc[st, ai]
    np.save(ROOT / "work" / "shuffle" / f"{args.tag}_boot_mean.npy", out_mean)
    shift.write_predictions(out_act, ROOT / "work" / "shuffle" / f"{args.tag}_actual.csv")
    ref = shift.read_predictions(ROOT / "results" / f"direct_{args.tag}_m114.csv").astype(np.float64)
    print(f"actual vs results/direct_{args.tag}_m114.csv max abs diff {np.abs(out_act - ref).max():.2e}; "
          f"E[W^2] pooled {np.mean((out_act - out_mean) ** 2):.5f}")


if __name__ == "__main__":
    main()
