#!/usr/bin/env python
"""Does a latent model's realization-specific weather response carry skill in-distribution?

On the public holdout fold, each checkpoint is rolled out (from slot 0) with the sites' actual forcing and with K
bootstrap weather realizations (per site: linear trend of every driver kept, residual years drawn with replacement,
one draw per year shared by all drivers). M = mean over the realizations, W = actual - M. If the model's
weather-specific predictions were right, the truth's deviation from M would load on W with lambda ~1.

    <torch venv>/bin/python scripts/weather_skill_holdout.py --tags ss20pze24cpu,noco2 --k 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgb import holdout, io, latent  # noqa: E402


def bootstrap_years(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    yrs = (np.arange(x.shape[1]) - (x.shape[1] - 1) / 2).astype(np.float64)
    a64 = x.astype(np.float64)
    slope = np.einsum("t,ntd->nd", yrs, a64) / float((yrs ** 2).sum())
    trend = a64.mean(axis=1, keepdims=True) + yrs[None, :, None] * slope[:, None, :]
    idx = rng.integers(0, x.shape[1], size=x.shape[:2])
    resid = np.take_along_axis(a64 - trend, idx[:, :, None], axis=1)
    return (trend + resid).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tags", default="ss20pze24cpu,noco2")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-sites", type=int, default=1500)
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    split = holdout.make_split(seed=42)
    arr = latent.load_arrays(split.fit)
    sites = np.sort(split.public)[: args.n_sites]
    ages = np.array(io.SEED_AGES)
    true40 = np.transpose(arr.u[:, sites, 40, :2], (1, 0, 2))
    pers = np.transpose(arr.u[:, sites, 0, :2], (1, 0, 2))
    bands = {"young": ages <= 30, "mid": (ages >= 50) & (ages <= 100), "old": ages >= 150}
    for tag in args.tags.split(","):
        ckpt = torch.load(io.ROOT / "work" / "models" / f"latent_{tag}" / "model.pt", map_location="cpu", weights_only=False)
        a = ckpt["args"]
        model = latent.LatentRollout(n_drv=136 + (latent.N_MONTHLY_INPUTS if a.get("monthly") else 0), d_enc=a.get("d_enc", 128),
                                     d_h=a.get("d_h", 256), n_layers=a.get("layers", 2), use_t=not a.get("no_t", False)).to(device)
        model.load_state_dict(ckpt["model"])
        input_set = a.get("input_set", "noco2" if a.get("drop_co2") else "full")
        x_year, season = arr.x_year.copy(), arr.season.copy()
        latent.apply_input_set(x_year, season, input_set)

        def run(xy: np.ndarray) -> np.ndarray:
            arr2 = latent.Arrays(xy, season, arr.static, arr.u, arr.agef, arr.y_mean, arr.lat, arr.season_mu, arr.season_sd)
            pred = latent.rollout_sites(model, arr2, sites, device, s0=0)
            return np.transpose(pred[:, :, -1, :2], (1, 0, 2)).astype(np.float64)

        p_act = run(x_year)
        rng = np.random.default_rng(7)
        reals = []
        for k in range(args.k):
            xb = x_year.copy()
            xb[sites] = bootstrap_years(x_year[sites], rng)
            reals.append(run(xb))
        P = np.stack(reals); M = P.mean(0)
        W = p_act - M
        print(f"{tag}: holdout {len(sites)} sites, K={args.k}: score actual {np.mean((p_act-true40)**2):.4f}, "
              f"expected-weather {np.mean((M-true40)**2):.4f}, one other realization {np.mean((P[0]-true40)**2):.4f}", flush=True)
        for b, m in bands.items():
            for c, cn in ((0, "h"), (1, "a")):
                w = W[:, m, c]; r = (true40 - M)[:, m, c]
                lam = float(np.sum(w * r) / np.sum(w * w))
                print(f"    {b:5s} {cn}: E[W^2] {np.mean(w**2):.5f}  lambda_W {lam:+.3f}  "
                      f"| E[(T-M)^2] {np.mean(r**2):.5f}  realization sd^2 {np.mean(P[:, :, m, c].var(0)):.5f}", flush=True)


if __name__ == "__main__":
    main()
