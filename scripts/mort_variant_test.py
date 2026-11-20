#!/usr/bin/env python
"""Full or position-free detector: is the full model's extra 2-3% signal, and is it position-keyed?

Two measurements on the holdout, with the boosters ``mort_gbdt.py`` saved:

  1. A **paired site bootstrap** of the difference in old-age agb error between the
     two cross-fitted blends (latent + full, latent + position-free), on each fold.
     Sites are resampled whole; if the interval covers zero the 2-3% is not
     resolved and the position-free model gives up nothing measurable.
  2. The full model **scored on rolled forcing**: its features are rebuilt after the
     40-year axis is rolled by k years (values intact, positions scrambled) and the
     prediction is scored against the true year-40 change. The position-free
     model is unchanged by a roll by construction. If the full model on rolled
     forcing scores like the position-free one, everything the band and
     year-of-max features add rides on position; if it still scores better, part
     of it is value-keyed and survives the scrambling.

Interpreter: the scratchpad venv of ``mort_gbdt.py`` (lightgbm 4.7.0).

    <scratchpad>/venvs/cgb-mort/bin/python scripts/mort_variant_test.py
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
from mort_detector import DROP_1KG, site_features  # noqa: E402

OUT_DIR = io.ROOT / "work" / "models" / "mort_detector"
METRICS = io.ROOT / "work" / "metrics"


def main() -> None:
    div = cache.scoring_divisors()
    y_mean = io.load_stats()["y_mean"].astype(np.float64)
    old = [j for j, a in enumerate(io.SEED_AGES) if a >= 150]
    split = holdout.make_split(seed=42)
    traj = cache.open_traj()
    xa = cache.open_x_annual()
    monthly = mort.open_monthly("train")
    gb = json.loads((METRICS / "mort_gbdt.json").read_text())
    boosters = {}
    for v in ("gbdt_full", "gbdt_position_free"):
        meta = json.loads((OUT_DIR / f"{v}_features.json").read_text())
        boosters[v] = (lgb.Booster(model_file=str(OUT_DIR / f"{v}.txt")), meta["cols"], {k: tuple(t) for k, t in meta["thr"].items()})

    payload: dict[str, object] = {}
    for fold, sites, other in (("public", split.public, "private"), ("private", split.private, "public")):
        n = len(sites)
        m = np.asarray(monthly[sites])
        static = np.asarray(xa[sites, 0], dtype=np.float64)[:, list(cache.STATIC_DRIVERS)]
        st = np.asarray(traj[old][:, sites], dtype=np.float64)
        state0 = (st[:, :, 0, :4] / y_mean[:4]).reshape(8 * n, 4)
        veg = (st[:, :, 0, 1] > mort.BARE_AGB).reshape(-1)
        y0 = (st[:, :, 0, 1] / div[1]).reshape(-1)
        truth = ((st[:, :, 40, 1] - st[:, :, 0, 1]) / div[1]).reshape(-1)
        lat = np.load(io.ROOT / "work" / "models" / "latent_v1" / f"pred_{fold}.npy")[:, :, -1, 1].astype(np.float64)[old].reshape(-1) - y0
        site_of_row = np.tile(np.arange(n), 8)

        def predict(v: str, mm: np.ndarray) -> np.ndarray:
            booster, cols, thr = boosters[v]
            sx, _ = site_features(mm, static, thr)
            x = np.concatenate([np.repeat(sx[None], 8, axis=0).reshape(8 * n, -1), state0, np.repeat(np.arange(8), n)[:, None]], axis=1)
            p = np.zeros(8 * n)
            p[veg] = booster.predict(x[veg][:, cols])
            return p

        p_full = predict("gbdt_full", m)
        p_free = predict("gbdt_position_free", m)
        # blends with the weights fitted on the OTHER fold (as in mort_gbdt.py)
        w_full = np.array(gb["blend_weights"]["gbdt_full"][fold])  # stored under the eval fold
        w_free = np.array(gb["blend_weights"]["gbdt_position_free"][fold])
        b_full = w_full[0] * lat + w_full[1] * p_full
        b_free = w_free[0] * lat + w_free[1] * p_free
        e_full = (b_full - truth) ** 2
        e_free = (b_free - truth) ** 2
        e_lat = (lat - truth) ** 2
        print(f"== {fold}: {n} sites, {veg.sum()} vegetated old rows ==")
        print(f"  latent {e_lat.mean():.5f}   blend full {e_full.mean():.5f}   blend position-free {e_free.mean():.5f}   "
              f"difference {e_full.mean() - e_free.mean():+.5f} ({(e_full.mean() - e_free.mean()) / e_free.mean():+.2%})")
        # paired site bootstrap of the difference
        per_site_full = np.bincount(site_of_row, weights=e_full, minlength=n) / 8
        per_site_free = np.bincount(site_of_row, weights=e_free, minlength=n) / 8
        d = per_site_full - per_site_free
        rng = np.random.default_rng(7)
        draws = d[rng.integers(0, n, size=(4000, n))].mean(axis=1)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        print(f"  paired site bootstrap of (full - position-free): mean {d.mean():+.5f}, 95% [{lo:+.5f}, {hi:+.5f}], "
              f"P(full better) {(draws < 0).mean():.3f}, SE {draws.std(ddof=1):.5f}")
        payload[fold] = {"latent": float(e_lat.mean()), "blend_full": float(e_full.mean()), "blend_free": float(e_free.mean()),
                         "diff_mean": float(d.mean()), "diff_lo95": float(lo), "diff_hi95": float(hi), "p_full_better": float((draws < 0).mean())}
        # the full model on rolled forcing, scored against the truth
        print(f"  stand-alone scores: full {((p_full - truth) ** 2).mean():.5f}, position-free {((p_free - truth) ** 2).mean():.5f}")
        rolled = {}
        for k in (7, 13, 20):
            p_roll = predict("gbdt_full", np.roll(m, k, axis=1))
            b_roll = w_full[0] * lat + w_full[1] * p_roll
            rolled[k] = (float(((p_roll - truth) ** 2).mean()), float(((b_roll - truth) ** 2).mean()))
            print(f"    full on forcing rolled by {k:2d} years: stand-alone {rolled[k][0]:.5f}, blend {rolled[k][1]:.5f}  "
                  f"(position-free: {((p_free - truth) ** 2).mean():.5f} / {e_free.mean():.5f})")
        payload[fold]["rolled"] = rolled
        # where do the two stand-alone models disagree most? by event-year band of the row's largest event
        ev = mort.events(st[:, :, :, 1].astype(np.float32))
        has = ev.any(axis=2).reshape(-1)
        big = np.where(ev, st[:, :, :-1, 1] - st[:, :, 1:, 1], 0.0).argmax(axis=2).reshape(-1)
        print("  stand-alone error by the forcing year of the row's largest event (rows with an event):")
        for lo_y, hi_y in ((1, 3), (4, 10), (11, 20), (21, 30), (31, 37), (38, 40)):
            mm = has & (big >= lo_y - 1) & (big <= hi_y - 1) & veg
            print(f"    years {lo_y:2d}-{hi_y:2d}: rows {mm.sum():5d}  full {((p_full - truth) ** 2)[mm].mean():.4f}  position-free {((p_free - truth) ** 2)[mm].mean():.4f}  latent {e_lat[mm].mean():.4f}")
    (METRICS / "mort_variant_test.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {METRICS / 'mort_variant_test.json'}")


if __name__ == "__main__":
    main()
