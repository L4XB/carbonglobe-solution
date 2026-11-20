#!/usr/bin/env python
"""How much does each model move when a holdout site's descriptor is replaced by the competition
pipeline's (the nearest competition site in ta_m/pr climatology: its hourly-group climatology,
its hourly-group annual summaries and its soil), against the same swap with the nearest other
holdout site (control)? The difference is the transfer cost of the descriptor shift for that
model; a model whose gap is small can be trusted more on the board.

    <torch+lgb venv>/bin/python scripts/descriptor_swap.py --latent noco2,clim --gbdt gbdt2,gbdt3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cgb import cache, holdout, io, latent  # noqa: E402

HOURLY_ALL = list(range(33, 129))
SOIL = [129, 130, 131, 132, 133, 134, 135]


def nearest(key_h: np.ndarray, key_t: np.ndarray, exclude_self: bool) -> np.ndarray:
    mu, sd = key_h.mean(0), key_h.std(0) + 1e-6
    zh, zt = (key_h - mu) / sd, (key_t - mu) / sd
    nn = np.empty(len(zh), dtype=int)
    for lo in range(0, len(zh), 500):
        hi = min(lo + 500, len(zh))
        dd = ((zh[lo:hi, None, :] - zt[None, :, :]) ** 2).sum(-1)
        if exclude_self:
            dd[np.arange(hi - lo), np.arange(lo, hi)] = np.inf
        nn[lo:hi] = dd.argmin(1)
    return nn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latent", default="noco2,clim")
    ap.add_argument("--gbdt", default="gbdt2,gbdt3")
    args = ap.parse_args()
    device = torch.device("cpu")
    stats = io.load_stats()
    x_mean, x_std = stats["x_mean"].astype(np.float32), stats["x_std"].astype(np.float32)
    x_std = np.where(x_std == 0, 1.0, x_std).astype(np.float32)
    y_mean = stats["y_mean"].astype(np.float64)
    agef = latent.age_features(stats)
    split = holdout.make_split(seed=42)
    sites = split.public
    ann = np.asarray(cache.open_x_annual()[sites], dtype=np.float32)
    sea = np.asarray(cache.open_x_season()[sites], dtype=np.float32)
    traj = cache.open_traj()
    u_all = np.asarray(traj[:, sites], dtype=np.float32) / y_mean.astype(np.float32)
    u0 = np.ascontiguousarray(np.transpose(u_all[:, :, 0, :], (1, 0, 2)))
    true40 = np.transpose(u_all[:, :, 40, :2], (1, 0, 2)).astype(np.float64)
    bare = (u0[:, :, 0] * y_mean[0]) < 1.0
    ann_t = np.load(cache.CACHE_DIR / "test_ssp126_annual.npy")
    sea_t = np.load(cache.CACHE_DIR / "test_ssp126_season.npy")
    key_h = np.concatenate([sea[:, 0, :] - 273.15, np.log1p(sea[:, 1, :])], axis=1)
    key_t = np.concatenate([sea_t[:, 0, :] - 273.15, np.log1p(sea_t[:, 1, :])], axis=1)
    nn_t = nearest(key_h, key_t, False)
    nn_h = nearest(key_h, key_h, True)

    def swapped(src_ann: np.ndarray, src_sea: np.ndarray, nn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = ann.copy()
        a[:, :, HOURLY_ALL] = src_ann[nn][:, :, HOURLY_ALL]
        a[:, :, SOIL] = src_ann[nn][:, :, SOIL]
        s = sea.copy()
        s[:, HOURLY_ALL, :] = src_sea[nn][:, HOURLY_ALL, :]
        s[:, SOIL, :] = src_sea[nn][:, SOIL, :]
        return a, s

    variants = {"own": (ann, sea), "competition twin": swapped(ann_t, sea_t, nn_t), "holdout twin (control)": swapped(ann, sea, nn_h)}

    models = {}
    for tag in [t for t in args.latent.split(",") if t]:
        ckpt = torch.load(io.ROOT / "work" / "models" / f"latent_{tag}" / "model.pt", map_location="cpu", weights_only=False)
        a = ckpt["args"]
        m = latent.LatentRollout(d_enc=a.get("d_enc", 128), d_h=a.get("d_h", 256), n_layers=a.get("layers", 2))
        m.load_state_dict(ckpt["model"])
        m.eval()
        models[f"latent_{tag}"] = (m, ckpt["season_mu"], ckpt["season_sd"], a.get("input_set", "noco2" if a.get("drop_co2") else "full"))

    @torch.no_grad()
    def predict_latent(entry, a_p, s_p) -> np.ndarray:
        m, mu, sd, input_set = entry
        x_year = ((a_p - x_mean) / x_std).astype(np.float32)
        static = x_year[:, 0, list(cache.STATIC_DRIVERS)].astype(np.float32)
        season = ((latent.seasonal_block(s_p) - mu) / sd).astype(np.float32)
        x_year, season = latent.apply_input_set(x_year.copy(), season.copy(), input_set)
        n = x_year.shape[0]
        out = np.zeros((n, 15, 2))
        for lo in range(0, n, 256):
            b = min(256, n - lo)
            xt = torch.from_numpy(x_year[lo:lo + b]).repeat(15, 1, 1)
            st = torch.from_numpy(static[lo:lo + b]).repeat(15, 1)
            se = torch.from_numpy(season[lo:lo + b]).repeat(15, 1)
            u0t = torch.from_numpy(np.ascontiguousarray(np.transpose(u0[lo:lo + b], (1, 0, 2)).reshape(15 * b, 7)))
            ag = torch.from_numpy(np.repeat(agef, b, axis=0))
            o = m(xt, se, st, u0t, ag)[:, -1, :2].numpy().reshape(15, b, 2)
            out[lo:lo + b] = np.transpose(o, (1, 0, 2))
        out[bare] = 0.0
        return out

    gbdts = {}
    if args.gbdt:
        import lightgbm as lgb  # imported lazily: torch + lightgbm in one process segfaulted on this Mac
        import direct_gbdt
        globals()["direct_gbdt"] = direct_gbdt
        for tag in args.gbdt.split(","):
            d = io.ROOT / "work" / "models" / f"direct_{tag}"
            gbdts[f"direct_{tag}"] = [lgb.Booster(model_file=str(d / f"{k}.txt")) for k in ("height", "agb")]
    thr = None
    monthly_h = np.asarray(cache.open_traj()[0:1, 0:1])  # placeholder to keep names
    from cgb import mortality as mort

    def predict_gbdt(boosters, a_p, s_p, tag: str) -> np.ndarray:
        direct_gbdt.USE_HOURLY = boosters[0].num_feature() != 106  # the no-hourly variant has 106 features
        feat = direct_gbdt.site_features(a_p, s_p)
        if boosters[0].num_feature() > feat.shape[1] + 4 + 15:
            nonlocal_thr = direct_gbdt.drought_thresholds()
            feat = np.concatenate([feat, direct_gbdt.drought_site_features(mort.open_monthly("train")[sites], nonlocal_thr)], axis=1)
        X = direct_gbdt.rows(feat, u0[:, :, :4].astype(np.float64), agef)
        ch = np.stack([b.predict(X) for b in boosters], axis=1)
        p = np.clip(X[:, -19:-15][:, :2] + ch, 0, None).reshape(15, len(sites), 2).transpose(1, 0, 2)
        p[bare] = 0.0
        return p

    for name, entry in list(models.items()) + list(gbdts.items()):
        preds = {}
        for vname, (a_p, s_p) in variants.items():
            preds[vname] = predict_latent(entry, a_p, s_p) if name.startswith("latent") else predict_gbdt(entry, a_p, s_p, name)
        base = preds["own"]
        line = f"{name:14s} own {np.mean((base - true40) ** 2):.4f}"
        for vname in ("competition twin", "holdout twin (control)"):
            p = preds[vname]
            line += f" | {vname}: score {np.mean((p - true40) ** 2):.4f} shift {np.mean((p - base) ** 2):.4f}"
        gap = np.mean((preds["competition twin"] - base) ** 2) - np.mean((preds["holdout twin (control)"] - base) ** 2)
        print(line + f" | excess shift of the competition pipeline {gap:+.4f}", flush=True)


if __name__ == "__main__":
    main()
