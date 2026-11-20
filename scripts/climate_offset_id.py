#!/usr/bin/env python
"""Identify release offsets of the test climate channels from the test initial states.

Training: the organisers' seed lookup gives y0 (slot 0) as a deterministic function of the site's climate, soil and
seed age; our ED validation found the 2011-2020 climatology (window years 30-39) reproduces it best. A LightGBM learns
y0 (height, agb; scaled by the global means) from that decade's monthly climatology of ta_m and pr (24 values), the
seven soil drivers and the seed age. Applied to the test (first window decade 2021-2030 as the closest available
climatology), the decoded test y0 is compared with the prediction after shifting the released temperature by dT
and transforming the released precipitation (raw | floor removed | 1.14114 x (raw - floor)). A clear optimum dT != 0
at temperature-limited (cold) sites would point to a temperature offset in the release (or a warmer lookup climate).
    python scripts/climate_offset_id.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import cache, holdout, io, shift  # noqa: E402

PR_FLOOR = 6.3517
SOIL = [129, 130, 131, 132, 133, 134, 135]


def decade_clim(x, years: slice, chunk: int = 1024) -> np.ndarray:
    """(n, 24): monthly means of ta_m (K) and pr over the given window years."""
    n = x.shape[0]
    out = np.empty((n, 24), np.float64)
    for lo in range(0, n, chunk):
        b = np.asarray(x[lo:lo + chunk, years, :, 0:2], dtype=np.float64)   # (c, y, 12, 2)
        m = b.mean(axis=1)                                                    # (c, 12, 2)
        out[lo:lo + chunk] = np.concatenate([m[:, :, 0], m[:, :, 1]], axis=1)
    return out


def main() -> None:
    y_mean = io.load_stats()["y_mean"].astype(np.float64)
    split = holdout.make_split(seed=42)
    xtr = io.open_train_x()
    print("training decade climatology (years 30-39) ...", flush=True)
    clim_tr = decade_clim(xtr, slice(30, 40))
    soil_tr = np.asarray(cache.open_x_annual()[:, 0, SOIL], dtype=np.float64)
    traj = cache.open_traj()
    ages = np.array(io.SEED_AGES, dtype=np.float64)
    n = clim_tr.shape[0]
    X = np.concatenate([np.concatenate([clim_tr, soil_tr, np.full((n, 1), a)], 1) for a in ages], 0).astype(np.float32)
    Y = np.concatenate([np.asarray(traj[j, :, 0, :2], dtype=np.float64) / y_mean[:2] for j in range(15)], 0)
    site_of_row = np.tile(np.arange(n), 15)
    fit = np.isin(site_of_row, split.fit)
    hold = np.isin(site_of_row, split.public)
    params = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=6, verbose=-1, seed=42)
    boosters = [lgb.train(params, lgb.Dataset(X[fit], label=Y[fit, k]), num_boost_round=600) for k in range(2)]
    ph = np.stack([b.predict(X[hold]) for b in boosters], 1)
    print(f"holdout y0 MSE height {np.mean((ph[:, 0] - Y[hold, 0]) ** 2):.4f} agb {np.mean((ph[:, 1] - Y[hold, 1]) ** 2):.4f} "
          f"(variance {Y[hold, 0].var():.3f} / {Y[hold, 1].var():.3f})", flush=True)
    # test: decoded y0 and the first-decade climatology of the released forcing
    order = [int(x) for x in (ROOT / "work/edv3/order_v4.txt").read_text().split()]
    grp = np.empty(6171, object)
    for i, s in enumerate(order):
        grp[s] = "cl" if i < 204 else "warm" if i < 2366 else "cold" if i < 5015 else "bare"
    res = {}
    for scen in io.SCENARIOS:
        xt, y0t = io.open_test(scen)
        clim = decade_clim(xt, slice(0, 10))
        soil = np.asarray(xt[:, 0, 0, SOIL], dtype=np.float64)
        y0 = np.asarray(y0t, dtype=np.float64)[:, :, :2]
        dec = 1.14114 * (y0 - shift.FLOORS[None, None, :2]) / y_mean[None, None, :2]      # (6171, 15, 2) scaled physical
        res[scen] = (clim, soil, dec)
    print("dT  precip            | MSE all (h, a) | mean pred-y0 (h, a) by group cold / warm / cl", flush=True)
    for pmode in ("raw", "nofloor", "x1.14"):
        for dT in (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0):
            err = {g: [] for g in ("cold", "warm", "cl", "all")}
            for scen, (clim, soil, dec) in res.items():
                c = clim.copy()
                c[:, :12] += dT
                if pmode == "nofloor":
                    c[:, 12:] = np.clip(c[:, 12:] - PR_FLOOR, 0, None)
                elif pmode == "x1.14":
                    c[:, 12:] = 1.14114 * np.clip(c[:, 12:] - PR_FLOOR, 0, None)
                Xt = np.concatenate([np.concatenate([c, soil, np.full((c.shape[0], 1), a)], 1) for a in ages], 0).astype(np.float32)
                p = np.stack([b.predict(Xt) for b in boosters], 1).reshape(15, -1, 2).transpose(1, 0, 2)   # (6171, 15, 2)
                d = p - dec
                for g in ("cold", "warm", "cl"):
                    err[g].append(d[grp == g].reshape(-1, 2))
                err["all"].append(d[np.isin(grp, ["cold", "warm", "cl"])].reshape(-1, 2))
            e = {g: np.concatenate(v, 0) for g, v in err.items()}
            print(f"{dT:+.0f}  {pmode:8s} | MSE {np.mean(e['all'] ** 2):.4f} ({np.mean(e['all'][:, 0] ** 2):.4f}, {np.mean(e['all'][:, 1] ** 2):.4f}) | "
                  + " / ".join(f"{g} h {e[g][:, 0].mean():+.3f} a {e[g][:, 1].mean():+.3f} mse {np.mean(e[g] ** 2):.3f}" for g in ("cold", "warm", "cl")), flush=True)


if __name__ == "__main__":
    main()
