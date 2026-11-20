#!/usr/bin/env python
"""The one place the collapses are still unmodelled: events in the last three forcing years.

``mort_residuals.py``: on rows whose largest event falls in forcing years 38-40 the
latent model keeps 24 % of the persistence loss (0.48 per row against 0.02-0.06
elsewhere), and those 0.5 % of old rows carry 13 % of its old-age agb error. A
drop in year 40 has no later years to be corrected by, so the model has to read
it off the last year's forcing alone. This script asks whether that residual is
a *bias* the late-band drought index can see -- and if so, what a cross-fitted
per-bin correction (fitted on one holdout fold, scored on the other) is worth.

    ./.venv/bin/python scripts/mort_late_events.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, holdout, io  # noqa: E402
from cgb import mortality as mort  # noqa: E402
from mort_trigger import build_indices  # noqa: E402

DROP_1KG = 1.0 / 3.395674394279066


def main() -> None:
    div = cache.scoring_divisors()
    old = [j for j, a in enumerate(io.SEED_AGES) if a >= 150]
    traj = cache.open_traj()
    split = holdout.make_split(seed=42)
    folds = {}
    for fold, sites in (("public", split.public), ("private", split.private)):
        pred = np.load(io.ROOT / "work" / "models" / "latent_v1" / f"pred_{fold}.npy")[:, :, -1, 1].astype(np.float64)[old]
        st = np.asarray(traj[old][:, sites], dtype=np.float64)
        y0 = st[:, :, 0, 1] / div[1]
        y40 = st[:, :, 40, 1] / div[1]
        veg = st[:, :, 0, 1] > mort.BARE_AGB
        ev = mort.events(st[:, :, :, 1].astype(np.float32))
        late_ev = ev[:, :, 37:].any(axis=2)
        ix = build_indices(np.asarray(mort.open_monthly("train")[sites]))
        fill = lambda x: np.where(np.isfinite(x), x, np.nanmin(x, axis=1, keepdims=True))  # noqa: E731
        late = {k: np.repeat(fill(ix[k])[:, 37:].max(axis=1)[None], 8, axis=0) for k in ("wb12", "spi12", "vpd6", "wb6")}
        folds[fold] = dict(pred=pred, y0=y0, y40=y40, veg=veg, late_ev=late_ev, late=late, agb0=st[:, :, 0, 1])

    print("== rows with an event in forcing years 38-40 (old ages, both folds) ==")
    for fold, f in folds.items():
        m = f["late_ev"] & f["veg"]
        r = f["pred"] - f["y40"]
        print(f"  {fold}: {m.sum()} rows; latent residual mean {r[m].mean():+.4f}, median {np.median(r[m]):+.4f}; persistence loss {((f['y0'] - f['y40'])[m] ** 2).mean():.4f}, latent loss {(r[m] ** 2).mean():.4f}")
        # what fraction of these did the latent see coming: predicted loss > 1 kgC
        print(f"          latent predicted a loss > 1 kgC on {((f['pred'] - f['y0'])[m] < -DROP_1KG).mean():.3f} of them; mean predicted change {((f['pred'] - f['y0'])[m]).mean():+.4f} vs true {((f['y40'] - f['y0'])[m]).mean():+.4f}")

    print("\n== latent residual (pred - truth) by the late-band drought index, vegetated old rows, both folds pooled ==")
    for key in ("wb12", "spi12", "vpd6"):
        x = np.concatenate([f["late"][key][f["veg"]] for f in folds.values()])
        r = np.concatenate([(f["pred"] - f["y40"])[f["veg"]] for f in folds.values()])
        e_lat = np.concatenate([((f["pred"] - f["y40"]) ** 2)[f["veg"]] for f in folds.values()])
        a0 = np.concatenate([f["agb0"][f["veg"]] for f in folds.values()])
        big = a0 > 8.0
        qs = np.quantile(x, [0.5, 0.8, 0.9, 0.95, 0.98, 0.99])
        edges = np.r_[-np.inf, qs, np.inf]
        print(f"  {key} (max over years 38-40):")
        print(f"    {'bin':>18s} {'rows':>7s} {'resid mean':>10s} {'latent mse':>10s} | {'agb0>8 rows':>11s} {'resid mean':>10s} {'latent mse':>10s}")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x > lo) & (x <= hi)
            mb = m & big
            print(f"    ({lo:8.2f},{hi:8.2f}] {m.sum():7d} {r[m].mean():+10.4f} {e_lat[m].mean():10.4f} | {mb.sum():11d} {r[mb].mean():+10.4f} {e_lat[mb].mean():10.4f}")

    # cross-fitted correction: on rows with late wb12 above a quantile, subtract the other fold's mean residual
    print("\n== cross-fitted additive correction on rows above a late-wb12 quantile (fit the shift on one fold, apply to the other) ==")
    for q in (0.90, 0.95, 0.98, 0.99):
        gains = []
        for fit_fold, eval_fold in (("public", "private"), ("private", "public")):
            ff, fe = folds[fit_fold], folds[eval_fold]
            thr = np.quantile(ff["late"]["wb12"][ff["veg"]], q)
            mf = ff["veg"] & (ff["late"]["wb12"] > thr)
            shift = (ff["pred"] - ff["y40"])[mf].mean()
            me = fe["veg"] & (fe["late"]["wb12"] > thr)
            before = ((fe["pred"] - fe["y40"]) ** 2)
            after = before.copy()
            after[me] = ((fe["pred"] - shift - fe["y40"]) ** 2)[me]
            gains.append((eval_fold, float(before.mean()), float(after.mean()), int(me.sum()), float(shift)))
        for ef, b, a, n, s in gains:
            print(f"  q{q:.2f}: eval {ef:7s} rows corrected {n:5d} shift {s:+.4f}  old-age agb mse {b:.5f} -> {a:.5f} ({(a - b) / b:+.2%})")


if __name__ == "__main__":
    main()
