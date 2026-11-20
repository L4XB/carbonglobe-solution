#!/usr/bin/env python
"""Is the detector keyed on driver values or on where a year sits on the axis?

The training forcing is 1982-2021 (``mort_calendar.py``) and its collapses pile
into the El Nino years; a model that learned "year 2 is bad" would transfer to
nothing under the SSP runs. Test: roll the forcing of 600 holdout sites by k
years (a circular shift of the 40-year axis, month order intact), rebuild the
features and predict with the saved boosters. A value-keyed model moves only
through the windows that straddle the seam; a calendar-keyed one moves a lot.

Reported per variant: the correlation and the mean absolute change between the
unshifted and the shifted prediction, against the spread of the predictions.

Interpreter: the scratchpad venv of ``mort_gbdt.py``.

    <scratchpad>/venvs/cgb-mort/bin/python scripts/mort_shift_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, holdout, io  # noqa: E402
from cgb import mortality as mort  # noqa: E402
from mort_detector import site_features  # noqa: E402

OUT_DIR = io.ROOT / "work" / "models" / "mort_detector"


def main() -> None:
    y_mean = io.load_stats()["y_mean"].astype(np.float64)
    old = [j for j, a in enumerate(io.SEED_AGES) if a >= 150]
    split = holdout.make_split(seed=42)
    sites = split.public[:600]
    traj = cache.open_traj()
    xa = cache.open_x_annual()
    monthly = np.asarray(mort.open_monthly("train")[sites])  # (600, 40, 12, 9)
    static = np.asarray(xa[sites, 0], dtype=np.float64)[:, list(cache.STATIC_DRIVERS)]
    st = np.asarray(traj[old][:, sites], dtype=np.float64)
    n = len(sites)
    state0 = (st[:, :, 0, :4] / y_mean[:4]).reshape(8 * n, 4)
    veg = (st[:, :, 0, 1] > mort.BARE_AGB).reshape(-1)

    for vname in ("gbdt_full", "gbdt_position_free"):
        booster = lgb.Booster(model_file=str(OUT_DIR / f"{vname}.txt"))
        meta = json.loads((OUT_DIR / f"{vname}_features.json").read_text())
        thr = {k: tuple(v) for k, v in meta["thr"].items()}
        cols = meta["cols"]

        def predict(m: np.ndarray) -> np.ndarray:
            sx, _ = site_features(m, static, thr)
            x = np.concatenate([np.repeat(sx[None], 8, axis=0).reshape(8 * n, -1), state0, np.repeat(np.arange(8), n)[:, None]], axis=1)
            return booster.predict(x[veg][:, cols])

        base = predict(monthly)
        print(f"{vname}: {veg.sum()} vegetated rows, prediction sd {base.std():.4f}, mean {base.mean():+.4f}")
        for k in (1, 7, 13, 20):
            rolled = np.roll(monthly, k, axis=1)
            p = predict(rolled)
            print(f"   roll by {k:2d} years: corr {np.corrcoef(base, p)[0, 1]:.4f}, mean |delta| {np.abs(p - base).mean():.4f} "
                  f"({np.abs(p - base).mean() / base.std():.3f} of the sd), mean delta {np.mean(p - base):+.4f}")
        rng = np.random.default_rng(0)
        perm = rng.permutation(40)
        p = predict(monthly[:, perm])
        print(f"   random permutation of the 40 years: corr {np.corrcoef(base, p)[0, 1]:.4f}, mean |delta| {np.abs(p - base).mean():.4f} "
              f"({np.abs(p - base).mean() / base.std():.3f} of the sd)")


if __name__ == "__main__":
    main()
