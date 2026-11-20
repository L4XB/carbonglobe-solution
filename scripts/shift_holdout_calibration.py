#!/usr/bin/env python
"""What a per-band damping can and cannot fix, measured on the holdout.

The correction is ``y0 + lambda (pred - y0)``: it rescales the model's predicted
change ``D`` band by band. Regressing the true change ``R = T - y0`` on ``D`` within
a band splits the model's error into the part a scalar fixes (the slope) and the
part it cannot (the residual after the best slope, i.e. change happening at the
wrong sites or in the wrong proportion between them). Two readings come out:

1. **In-distribution calibration.** On the holdout the initial states are in
   distribution, so the slope should be 1. If it is, the model is calibrated where
   the holdout can see, and a board-derived ``lambda != 1`` is the shift itself.
2. **The ceiling of the correction under a simulated shift.** If the test population
   changed by ``kappa`` times the holdout's change in a band, the truth there is
   ``y0 + kappa R``. The best scalar on ``D`` recovers ``kappa`` only where ``D``
   points along ``R``; the residual term stays. Printed for a few ``kappa``.

Reads the sibling's saved holdout predictions (year-40 slice of
``work/models/<tag>/pred_public.npy``, ``(15, 3086, 5, 2)``); nothing is retrained.

    ./.venv/bin/python scripts/shift_holdout_calibration.py --tag latent_v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, holdout, io, shift  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="latent_v1")
    args = ap.parse_args()

    div = cache.scoring_divisors()
    split = holdout.make_split(seed=42)
    sites = split.public
    y0 = cache.state_block(0)[:, sites, :2] / div  # (15, n, 2)
    y40 = cache.state_block(40)[:, sites, :2] / div
    pred = np.load(io.ROOT / "work" / "models" / args.tag / "pred_public.npy")[:, :, 4, :].astype(np.float64)
    if pred.shape != y0.shape:
        raise AssertionError(f"{args.tag}: pred {pred.shape} against truth {y0.shape}")
    ages = np.repeat(np.asarray(io.SEED_AGES)[:, None], len(sites), axis=1)
    masks = shift.band_masks(ages.ravel())

    d = (pred - y0).reshape(-1, 2)
    r = (y40 - y0).reshape(-1, 2)
    e_model = ((pred - y40) ** 2).reshape(-1, 2)
    e_pers = ((y0 - y40) ** 2).reshape(-1, 2)
    print(f"== {args.tag}, public fold, {len(sites)} sites: model score {e_model.mean():.5f}, persistence {e_pers.mean():.5f} ==")
    print(f"\n== 1. in-distribution calibration of the predicted change, per band and column ==")
    print(f"  {'band':6s} {'col':7s} {'slope R~D':>10s} {'R^2':>6s} {'mse model':>10s} {'mse persist':>12s} "
          f"{'best-lambda mse':>16s} {'share of model err':>19s}")
    tot = e_model.sum()
    for band, m in masks.items():
        for k, col in enumerate(shift.COLUMNS):
            dk, rk = d[m, k], r[m, k]
            sl = shift.slope(dk, rk)
            resid = rk - sl * dk
            r2 = 1.0 - np.mean(resid**2) / np.mean(rk**2)
            print(f"  {band:6s} {col:7s} {sl:10.4f} {r2:6.3f} {e_model[m, k].mean():10.5f} {e_pers[m, k].mean():12.5f} "
                  f"{np.mean(resid**2):16.5f} {e_model[m, k].sum() / tot:19.1%}")

    print("\n== 2. ceiling of a per-band scalar under a simulated shift of the old band ==")
    print("  truth' = y0 + kappa (T - y0) on the old band; everything else unchanged")
    m = masks["old"]
    print(f"  {'kappa':>6s} {'model as is':>12s} {'lambda*':>8s} {'corrected':>10s} {'persistence':>12s} {'unfixable':>10s}")
    for kappa in (1.0, 0.8, 0.6, 0.4):
        r_shift = kappa * r[m]
        e_as_is = np.mean((d[m] - r_shift) ** 2)
        lam = shift.slope(d[m], r_shift)
        e_corr = np.mean((lam * d[m] - r_shift) ** 2)
        e_pers_b = np.mean(r_shift**2)
        print(f"  {kappa:6.2f} {e_as_is:12.5f} {lam:8.4f} {e_corr:10.5f} {e_pers_b:12.5f} {e_corr / e_as_is:10.1%}")
    print("  'unfixable' is the corrected error as a share of the uncorrected one: the part that is not a scale.")

    print("\n== 3. the same, if the shift is not a uniform scale but a different set of sites moving ==")
    print("  truth' on the old band = y0 + (T - y0) with the change permuted across a random half of the sites")
    rng = np.random.default_rng(42)
    idx = np.flatnonzero(m)
    half = rng.choice(idx, size=len(idx) // 2, replace=False)
    r_perm = r.copy()
    r_perm[half] = r[rng.permutation(half)]
    e_as_is = np.mean((d[m] - r_perm[m]) ** 2)
    lam = shift.slope(d[m], r_perm[m])
    e_corr = np.mean((lam * d[m] - r_perm[m]) ** 2)
    print(f"  model as is {e_as_is:.5f}   lambda* {lam:.4f}   corrected {e_corr:.5f}   persistence {np.mean(r_perm[m] ** 2):.5f}")
    print("  a scalar cannot move change between sites; when the shift is where, not how much, lambda barely helps.")


if __name__ == "__main__":
    main()
