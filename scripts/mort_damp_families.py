#!/usr/bin/env python
"""Uniform damping of the old-stand losses against a VPD-proportional one: what is it worth?

The board says the scored trajectories change less than the training ones, and
``mort_test_forcing.py`` / ``mort_latent_response.py`` locate the latent model's
excess loss under ssp585 on the VPD axis, where the forcing has extremes the
training set supports only thinly (high VPD without a precipitation deficit).
Two ways to pull the predicted losses back: a **uniform** factor on every old-age
loss, or a factor **proportional to the site's VPD excess**, so the rows that rest
on the least-supported mechanism are damped most.

On the holdout the model is calibrated (predicted collapse rate 0.304 vs 0.313),
so neither family *removes* error there -- both add it, and the optimum is factor
1. What the holdout can measure:

  1. the alignment of each family with the model's residual, i.e. the slope of the
     old-age agb error in the damping factor at f = 1: proportional to
     mean(residual x weight x predicted loss). A family that damps where the
     model already over-predicts loss has the smaller slope (or a negative one);
  2. the full error curve in f for both, to see the cost of a given amount of
     damping under each weighting if the truth behaved like the holdout;

and on the scored rows:

  3. whether the two corrections are different objects at all: the correlation of
     the per-row adjustments at the same average damping, and where each puts
     its correction (biggest stands, most-damped tenth).

Weights: ``w_i`` is a site summary normalised to mean 1 over the vegetated
old-age rows, the retained fraction of the predicted loss is ``1 - (1 - f) w_i``
clipped to [0, 1], so ``f`` is the average retained fraction in every family.
Three VPD summaries are tried; the uniform family is ``w = 1``.

    ./.venv/bin/python scripts/mort_damp_families.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, holdout, io, submission  # noqa: E402
from cgb import mortality as mort  # noqa: E402
from mort_trigger import build_indices  # noqa: E402

M_ALL = 1.14114
FACTORS = (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5)


def vpd_summaries(monthly: np.ndarray, thr: dict[str, float]) -> dict[str, np.ndarray]:
    """Per-site VPD-excess summaries; larger = more of the loss rests on the VPD axis."""
    ix = build_indices(monthly)
    fill = lambda x: np.where(np.isfinite(x), x, np.nanmin(x, axis=1, keepdims=True))  # noqa: E731
    vpd6, spi12 = fill(ix["vpd6"]), fill(ix["spi12"])
    return {
        "vpd6_excess": np.maximum(vpd6 - thr["vpd6_q90"], 0).sum(axis=1),
        "n_vpd_hi": (vpd6 > thr["vpd6_q97"]).sum(axis=1).astype(float),
        "n_vpd_hi_not_dry": ((vpd6 > thr["vpd6_q97"]) & (spi12 < thr["spi12_q90"])).sum(axis=1).astype(float),
    }


def damp(pred: np.ndarray, y0: np.ndarray, w: np.ndarray, f: float) -> np.ndarray:
    """Pull predicted losses towards y0: retained fraction ``1 - (1 - f) w``, growth rows untouched."""
    loss = np.maximum(y0 - pred, 0.0)
    keep = np.clip(1.0 - (1.0 - f) * w, 0.0, 1.0)
    return np.where(pred < y0, y0 - keep * loss, pred)


def main() -> None:
    div = cache.scoring_divisors()
    y_mean = io.load_stats()["y_mean"].astype(np.float64)
    old = [j for j, a in enumerate(io.SEED_AGES) if a >= 150]
    traj = cache.open_traj()
    split = holdout.make_split(seed=42)
    monthly = mort.open_monthly("train")
    ix_fit = build_indices(np.asarray(monthly[split.fit[::4]]))
    thr = {
        "vpd6_q90": float(np.nanquantile(ix_fit["vpd6"], 0.90)),
        "vpd6_q97": float(np.nanquantile(ix_fit["vpd6"], 0.97)),
        "spi12_q90": float(np.nanquantile(ix_fit["spi12"], 0.90)),
    }
    del ix_fit

    # ---- holdout ---------------------------------------------------------------
    folds = {}
    for fold, sites in (("public", split.public), ("private", split.private)):
        pred = np.load(io.ROOT / "work" / "models" / "latent_v1" / f"pred_{fold}.npy")[:, :, -1, 1].astype(np.float64)[old]
        st = np.asarray(traj[old][:, sites], dtype=np.float64)
        y0 = st[:, :, 0, 1] / div[1]
        y40 = st[:, :, 40, 1] / div[1]
        veg = st[:, :, 0, 1] > mort.BARE_AGB
        s = vpd_summaries(np.asarray(monthly[sites]), thr)
        fams = {"uniform": np.ones_like(y0)}
        for k, v in s.items():
            w = np.repeat(v[None], 8, axis=0)
            fams[k] = w / w[veg].mean()
        folds[fold] = dict(pred=pred, y0=y0, y40=y40, veg=veg, fams=fams)

    print("== holdout: old-age agb error (scaled MSE over vegetated old rows) under each damping family ==")
    print(f"  weights normalised to mean 1 over the rows; f = average retained fraction of the predicted loss")
    payload: dict[str, object] = {"thr": thr, "holdout": {}}
    for fold, d in folds.items():
        veg = d["veg"]
        r = (d["pred"] - d["y40"])[veg]
        loss = np.maximum(d["y0"] - d["pred"], 0.0)[veg]
        print(f"\n  {fold}: {veg.sum()} rows; latent error {np.mean(r ** 2):.5f}; mean predicted loss {loss.mean():.4f}; mean residual {r.mean():+.4f}")
        print(f"    {'family':20s} {'slope at f=1':>12s} " + " ".join(f"{f:>8.2f}" for f in FACTORS))
        payload["holdout"][fold] = {}
        for name, w in d["fams"].items():
            wv = w[veg]
            slope = float(2 * np.mean(r * wv * loss))  # d MSE / d(1-f) at f = 1
            curve = [float(np.mean((damp(d["pred"], d["y0"], w, f) - d["y40"])[veg] ** 2)) for f in FACTORS]
            payload["holdout"][fold][name] = {"slope": slope, "curve": curve}
            print(f"    {name:20s} {slope:+12.5f} " + " ".join(f"{c:8.5f}" for c in curve))
        # the sign question: where does the model over-predict loss on the holdout?
        for name in ("vpd6_excess", "n_vpd_hi_not_dry"):
            wv = d["fams"][name][veg]
            hi = wv > np.quantile(wv, 0.9)
            print(f"    rows in the top tenth of {name:18s}: residual mean {r[hi].mean():+.4f} (rest {r[~hi].mean():+.4f}); "
                  f"predicted loss {loss[hi].mean():.4f} vs true loss {np.maximum(d['y0'] - d['y40'], 0)[veg][hi].mean():.4f}")

    # ---- scored rows: are the two corrections different objects? -----------------
    print("\n== scored runs: per-row adjustment of a uniform vs a VPD-proportional damping at the same average f ==")
    frame = pd.read_csv(io.ROOT / "results" / "latent_v1_m114.csv", dtype={"id": str}, keep_default_na=False)
    keys = submission.parse_ids(frame["id"])
    frames = {s: io.open_test(s)[1].astype(np.float64) for s in io.SCENARIOS}
    floors = np.concatenate([fr.reshape(-1, 7) for fr in frames.values()]).min(axis=0)
    payload["test"] = {}
    for sc in io.SCENARIOS:
        raw0 = M_ALL * (frames[sc] - floors)
        y0 = np.transpose(raw0[:, old, 1]) / y_mean[1]
        agb0 = np.transpose(raw0[:, old, 1])
        veg = agb0 > mort.BARE_AGB
        sel = keys["scenario"].to_numpy() == sc
        k = keys.loc[sel]
        p = np.full((6171, 15), np.nan)
        p[k["site"].to_numpy(), [io.SEED_AGES.index(a) for a in k["age"].to_numpy()]] = frame.loc[sel, "agb"].to_numpy(dtype=np.float64)
        pred = np.transpose(p[:, old])
        s = vpd_summaries(np.asarray(mort.open_monthly(sc)), thr)
        fams = {"uniform": np.ones_like(y0)}
        for kk, v in s.items():
            w = np.repeat(v[None], 8, axis=0)
            fams[kk] = w / w[veg].mean()
        loss = np.maximum(y0 - pred, 0.0)
        big = agb0 > 12.0
        print(f"\n  {sc}: vegetated old rows {veg.sum()}, predicted total loss {loss[veg].sum():.1f} (scaled), rows with a predicted loss {(loss[veg] > 0).mean():.3f}")
        print(f"    share of sites with n_vpd_hi_not_dry = 0: {(s['n_vpd_hi_not_dry'] == 0).mean():.3f}; mean {s['n_vpd_hi_not_dry'].mean():.2f} years  (holdout public: "
              f"{(folds['public']['fams']['n_vpd_hi_not_dry'][0] == 0).mean():.3f}, mean {folds['public']['fams']['n_vpd_hi_not_dry'][0].mean() * 1:.2f} x normalisation)")
        f = 0.7
        adj_u = (damp(pred, y0, fams["uniform"], f) - pred)[veg]
        payload["test"][sc] = {}
        print(f"    {'family':20s} {'corr w/ uniform':>15s} {'total adj':>10s} {'on agb0>12':>10s} {'on top-10% adj rows':>19s} {'rows adjusted':>13s}")
        for name, w in fams.items():
            adj = (damp(pred, y0, w, f) - pred)[veg]
            corr = float(np.corrcoef(adj, adj_u)[0, 1])
            tot = float(adj.sum())
            on_big = float(adj[big[veg]].sum() / tot) if tot else float("nan")
            top = adj > np.quantile(adj, 0.9)
            on_top = float(adj[top].sum() / tot) if tot else float("nan")
            payload["test"][sc][name] = {"corr_with_uniform": corr, "total": tot, "share_big": on_big, "share_top10": on_top}
            print(f"    {name:20s} {corr:15.3f} {tot:10.1f} {on_big:10.3f} {on_top:19.3f} {(adj > 0).mean():13.3f}")
    (io.ROOT / "work" / "metrics" / "mort_damp_families.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {io.ROOT / 'work' / 'metrics' / 'mort_damp_families.json'}")


if __name__ == "__main__":
    main()
