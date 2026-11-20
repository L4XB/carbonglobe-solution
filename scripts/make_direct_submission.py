#!/usr/bin/env python
"""Fit the best measured baseline on all training sites and build a submission.

The rung is the ladder's winner (``scripts/baseline_ladder.py``): a per-seed-age
ridge that predicts the **year-40** state directly from the year-0 state and the
40-year forcing aggregates, with no autoregression. It scores 0.0716 on the
site-held-out offline holdout against 0.3115 for persistence, and
``scripts/rollout_drift.py`` reproduces 0.07164 from an independent code path, so
that number is not one script's arithmetic.

Three things stand between that model and a submission. All three are measured.

**The decode.** ``test_y0`` is not on the training set's footing. 8,185 of the 92,565
(site, age) rows carry one exact constant 7-vector, ``[-0.900947, -0.401412,
-0.532395, 0.117938, 0.259026, 0.127815, 0.084656]``; in training, 67,127 rows carry
the state of a bare site, and that state is the **zero vector**. A different sample of
grid cells cannot turn an exact zero into an exact -0.900947, so the additive offset is
real and its value is known. The slope is *not* identified -- the test sites are
different cells (no static-driver fingerprint of the 6,171 matches any of the 54,152),
so a contraction and a less productive sample look alike. This script uses
``physical = test_y0 - floor`` and measures what the alternatives would cost by handing
the offline holdout the same wrong input.

**The extrapolation.** Fitting on all 136 forcing means and all 136 forcing trends is
the best thing to do offline and a trap on the leaderboard. Training CO2 rises at
1.571 ppm/yr at every site with an across-site spread of 0.029, so the *coefficient on
the CO2 trend is fitted on noise*; the SSP scenarios then hand that feature a value
50-60 training standard deviations away. A filter on **test-side leverage** -- drop any
feature whose test value lies more than ``t`` training sd from the training mean --
uses no labels at all and is swept here against both the offline score and the
plausibility of the resulting predictions.

**The clip.** Year-40 height is hard-capped at 35.0 m in the process model and neither
column can go negative, so predictions are clipped into the training year-40 range.
That "can only help" is an argument, not a measurement, so it is measured.

Nothing here submits anything. It writes files and prints their paths.

    ./.venv/bin/python scripts/make_direct_submission.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, holdout, io, submission  # noqa: E402

WORK = io.ROOT / "work"
RIDGE_GRID = (1e-4, 1e-2, 1.0, 10.0, 100.0, 1e3)

#: The state a bare site carries in ``test_y0``. Measured, not chosen: the exact
#: 7-vector on 8,185 rows of ssp126 and 8,178 of ssp585, whose training twin is the
#: zero vector on 67,127 rows.
TEST_FLOOR = np.array(
    [-0.900947, -0.401412, -0.532395, 0.117938, 0.259026, 0.127815, 0.084656]
)

#: Slope of the same map, ``test = a * physical + floor``, i.e. the decode is
#: ``physical = (test - floor) / a``.
#:
#: Height's slope is pinned by a hard model constant: the 35.0002 m cap. 43 test rows
#: sit pinned at a shifted 30.671502 -- bit-identical across both scenarios, against
#: 295 training rows at the cap -- and 30.671502 / 35.000153 = 0.876325. The 1.5 m
#: seeding height (the age-1 floor, 107 training rows exactly on it) independently
#: gives 0.877420. The two agree to 0.12%, and at 1/0.876325 = 1.141 the decoded
#: height maximum *and* its 99.99th percentile both land on 35.000 exactly.
#:
#: Whether the same slope applies to the other six is not settled by a hard constant,
#: because none of them has a cap -- each has exactly one row at its maximum, i.e. a
#: sample maximum. ``scripts/theory_*.py`` argues one multiplier for all seven from the
#: bare-row flux plateau and the GPP-given-LAI relation, and that is the default here.
#: The dissenting measurement, recorded rather than buried: at a uniform 1.141 the
#: decoded gpp and npp 99.99th percentiles sit 8.8% *above* the training ones and lai
#: 18.3% above, while every other variable sits below -- which is what a slope of 1 on
#: those three would predict. It cannot be resolved offline: the test sites are
#: different cells, so a contraction and a different site sample look alike. A
#: leaderboard persistence probe settles it, and ``--slope-mode height-only`` rebuilds
#: both files under the alternative in one command.
SLOPE_MODES = {
    "uniform": np.full(7, 0.876325),
    "height-only": np.array([0.876325, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    "none": np.ones(7),
}

#: Leverage thresholds swept over. ``inf`` keeps every feature (the offline optimum).
THRESHOLDS = (np.inf, 6.0, 4.0, 3.0, 2.0, 0.0)


def quad(s: np.ndarray) -> np.ndarray:
    return np.concatenate([s, s**2, np.log1p(np.clip(s, 0.0, None))], axis=1)


class Std:
    def __init__(self, x: np.ndarray) -> None:
        self.mean = x.mean(axis=0)
        sd = x.std(axis=0)
        self.dead = sd <= 1e-12
        self.scale = np.where(self.dead, 1.0, sd)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = (x - self.mean) / self.scale
        out[..., self.dead] = 0.0
        return out


def ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n, p = x.shape
    d = np.empty((n, p + 1))
    d[:, :p] = x
    d[:, p] = 1.0
    pen = np.eye(p + 1) * lam
    pen[p, p] = 0.0
    return np.linalg.solve(d.T @ d + pen, d.T @ y)


def apply_beta(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return x @ beta[:-1] + beta[-1]


def scenario_climate(scenario: str) -> np.ndarray:
    """``(6171, 272)``: each test driver's 40-year mean, then its per-year trend."""
    x, _ = io.open_test(scenario)
    years = np.arange(40, dtype=np.float64)
    years -= years.mean()
    denom = float((years**2).sum())
    mean = np.empty((x.shape[0], 136))
    trend = np.empty((x.shape[0], 136))
    for lo in range(0, x.shape[0], 500):
        hi = min(lo + 500, x.shape[0])
        block = np.asarray(x[lo:hi], dtype=np.float64).mean(axis=2)
        mean[lo:hi] = block.mean(axis=1)
        trend[lo:hi] = np.tensordot(years, block, axes=(0, 1)) / denom
    return np.concatenate([mean, trend], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slope-mode",
        choices=sorted(SLOPE_MODES),
        default="uniform",
        help="which decode slope to apply to test_y0 (default: uniform 1.141 on all seven)",
    )
    args = parser.parse_args()
    test_slope = SLOPE_MODES[args.slope_mode]
    print(f"decode slope mode: {args.slope_mode} -> {np.round(1.0 / test_slope, 4)} as a multiplier\n")
    div = cache.scoring_divisors()
    ages = io.SEED_AGES
    n_ages = len(ages)

    y0_all = cache.state_block(0)
    y40_all = cache.state_block(40)[:, :, :2]
    target_all = y40_all / div
    clim = cache.climate_summary()
    cf_train = np.concatenate([clim["mean"], clim["trend"]], axis=1)  # (54152, 272)
    lo_y40 = y40_all.reshape(-1, 2).min(axis=0) / div
    hi_y40 = y40_all.reshape(-1, 2).max(axis=0) / div

    split = holdout.make_split(seed=42)
    hold, fit = split.holdout, split.fit
    pub_mask = np.isin(hold, split.public)

    print("== 1. how far outside the training forcing the two scenarios sit ==")
    mu, sd = cf_train.mean(axis=0), cf_train.std(axis=0)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    cf_test = {sc: scenario_climate(sc) for sc in io.SCENARIOS}
    lev = np.zeros(cf_train.shape[1])
    for sc, cf in cf_test.items():
        z = np.abs((cf - mu) / sd_safe)
        lev = np.maximum(lev, z.max(axis=0))
        print(
            f"  {sc}: median |z| {np.median(np.abs((cf - mu) / sd_safe)):.3f}, "
            f"max {z.max():.1f}, {(z > 3).mean():.2%} of feature cells beyond 3 sd"
        )
    worst = np.argsort(lev)[::-1][:6]
    print("  worst features (index < 136 is a 40-year mean, >= 136 a trend):")
    for d in worst:
        kind = "mean " if d < 136 else "trend"
        print(
            f"    driver {d % 136:3d} {kind}: leverage {lev[d]:6.1f} sd   "
            f"training across-site sd {sd[d]:.6g}"
        )
    print(
        "  the CO2 drivers (8..31) rise at the same rate at every training site, so their\n"
        "  trend has almost no across-site variance to fit a coefficient on -- and the SSP\n"
        "  pathways then move it tens of training sd. That is the extrapolation."
    )

    # ------------------------------------------------------------------ models
    def build(j: int, sites: np.ndarray, keep: np.ndarray, y0_source=None, cf_source=None):
        s = y0_all[j][sites] if y0_source is None else y0_source
        c = cf_train[sites][:, keep] if cf_source is None else cf_source[:, keep]
        return np.concatenate([quad(s), c], axis=1)

    def choose_lambda(keep: np.ndarray) -> float:
        rng = np.random.default_rng(1234)
        perm = rng.permutation(len(fit))
        cut = int(0.2 * len(fit))
        iv, ifit = fit[perm[:cut]], fit[perm[cut:]]
        best, best_mse = RIDGE_GRID[0], np.inf
        for lam in RIDGE_GRID:
            tot = 0.0
            for j in range(n_ages):
                d = build(j, ifit, keep)
                std = Std(d)
                beta = ridge(std(d), target_all[j][ifit], lam)
                dv = build(j, iv, keep)
                tot += float(((apply_beta(std(dv), beta) - target_all[j][iv]) ** 2).mean())
            if tot < best_mse:
                best_mse, best = tot, lam
        return best

    def holdout_score(keep: np.ndarray, lam: float, transform=None, clip: bool = True) -> float:
        err = np.empty((len(hold), n_ages, 2))
        for j in range(n_ages):
            d = build(j, fit, keep)
            std = Std(d)
            beta = ridge(std(d), target_all[j][fit], lam)
            s = y0_all[j][hold]
            if transform is not None:
                s = transform(s)
            dv = np.concatenate([quad(s), cf_train[hold][:, keep]], axis=1)
            pred = apply_beta(std(dv), beta)
            if clip:
                pred = np.clip(pred, lo_y40, hi_y40)
            err[:, j, :] = (pred - target_all[j][hold]) ** 2
        return float(err[pub_mask].mean())

    def predict_test(keep: np.ndarray, lam: float, offset: np.ndarray, clip: bool,
                     slope: np.ndarray | None = None):
        betas = []
        all_sites = np.arange(y0_all.shape[1])
        for j in range(n_ages):
            d = build(j, all_sites, keep)
            std = Std(d)
            betas.append((ridge(std(d), target_all[j][all_sites], lam), std))
        out = {}
        for sc in io.SCENARIOS:
            _, y0_test = io.open_test(sc)
            y0_test = np.asarray(y0_test, dtype=np.float64)
            p = np.empty((y0_test.shape[0], n_ages, 2))
            for j in range(n_ages):
                beta, std = betas[j]
                s_in = y0_test[:, j, :] - offset
                if slope is not None:
                    s_in = s_in / slope
                d = np.concatenate([quad(s_in), cf_test[sc][:, keep]], axis=1)
                v = apply_beta(std(d), beta)
                if clip:
                    v = np.clip(v, lo_y40, hi_y40)
                p[:, j, :] = v
            out[sc] = p
        return out

    # ------------------------------------------------- the leverage/accuracy trade
    print("\n== 2. the trade: offline accuracy against test-side extrapolation ==")
    print(
        "  a feature is kept only if every test site's value lies within t training sd of\n"
        "  the training mean. That rule reads inputs only -- no label, no leakage."
    )
    print(
        f"  {'t':>6s} {'features kept':>14s} {'offline public':>15s} {'test height mean':>17s} "
        f"{'% at 35m cap':>13s} {'test agb mean':>14s} {'ssp gap':>9s}"
    )
    train_h_mean = float(target_all[..., 0].mean())
    train_a_mean = float(target_all[..., 1].mean())
    sweep = {}
    for t in THRESHOLDS:
        keep = np.where(lev <= t)[0] if np.isfinite(t) else np.arange(cf_train.shape[1])
        if t == 0.0:
            keep = np.zeros(0, dtype=int)
        lam = choose_lambda(keep)
        off = holdout_score(keep, lam)
        p = predict_test(keep, lam, TEST_FLOOR, clip=True)
        h = np.concatenate([p[sc][..., 0].ravel() for sc in io.SCENARIOS])
        a = np.concatenate([p[sc][..., 1].ravel() for sc in io.SCENARIOS])
        gap = float(np.sqrt(((p["ssp585"] - p["ssp126"]) ** 2).mean()))
        sweep[str(t)] = {
            "keep": int(len(keep)),
            "lambda": lam,
            "offline": off,
            "height_mean": float(h.mean()),
            "at_cap": float((h >= hi_y40[0] - 1e-9).mean()),
            "agb_mean": float(a.mean()),
            "ssp_gap": gap,
        }
        label = "inf" if not np.isfinite(t) else f"{t:g}"
        print(
            f"  {label:>6s} {len(keep):14d} {off:15.5f} {h.mean():17.4f} "
            f"{(h >= hi_y40[0] - 1e-9).mean():13.1%} {a.mean():14.4f} {gap:9.4f}"
        )
    print(
        f"  for reference the training year-40 truth has height mean {train_h_mean:.4f} "
        f"and agb mean {train_a_mean:.4f}, with 0.0% of rows at the cap"
    )

    chosen_t = 3.0
    keep = np.where(lev <= chosen_t)[0]
    lam = choose_lambda(keep)
    print(
        f"\n  chosen: t = {chosen_t:g}, {len(keep)} of {cf_train.shape[1]} forcing features kept, "
        f"lambda {lam:g}\n"
        "  The unfiltered model is better offline and produces predictions that cannot be\n"
        "  right: a third of all rows pinned at the 35 m cap and a mean height twice the\n"
        "  training truth's. Offline cannot see that, because offline nothing extrapolates."
    )

    # ------------------------------------------------------------------ clip
    no_clip = holdout_score(keep, lam, clip=False)
    with_clip = holdout_score(keep, lam, clip=True)
    print(
        f"\n== 3. clipping into the training year-40 range ==\n"
        f"  off {no_clip:.5f} -> on {with_clip:.5f} ({(with_clip - no_clip) / no_clip:+.2%}); kept"
    )

    # ------------------------------------------------------------------ decode
    print("\n== 4. what a wrong decode of test_y0 would cost ==")
    print("  the offline holdout is handed the same wrong input; the model is unchanged")
    print(f"  {'input handed to the model':44s} {'public':>9s} {'vs correct':>11s}")
    ref = with_clip
    decode_costs = {}
    for label, fn in (
        ("correct physical state", lambda s: s),
        ("state + floor  (reading test_y0 raw)", lambda s: s + TEST_FLOOR),
        ("state x 0.92   (an 8% contraction)", lambda s: s * 0.92),
        ("state x 0.85   (a 15% contraction)", lambda s: s * 0.85),
        ("(state + floor) x 0.92", lambda s: (s + TEST_FLOOR) * 0.92),
        (
            "height x 0.876 only (missing the height slope)",
            lambda s: s * np.array([0.876325, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ),
        (
            "every column x 0.876 (slope on all seven)",
            lambda s: s * 0.876325,
        ),
    ):
        v = holdout_score(keep, lam, transform=fn)
        decode_costs[label] = v
        print(f"  {label:44s} {v:9.5f} {v / ref:10.2f}x")
    print(
        "  reading test_y0 raw is worse than answering all zeros (2.83). The offset is the\n"
        "  single most expensive thing to get wrong after the /mean scaling itself."
    )

    # ------------------------------------------------------------------ write
    print("\n== 5. submission files ==")
    template = submission.load_template()
    submission.check_divisors()
    key = submission.parse_ids(template["id"])
    site = key["site"].to_numpy()
    age_pos = np.array([ages.index(a) for a in key["age"].to_numpy()])
    is126 = key["scenario"].to_numpy() == "ssp126"

    print(
        "  the model predicts the PHYSICAL year-40 state. Which space the stored solution\n"
        "  is in is not knowable offline: test_y0 is an affine image of the physical state,\n"
        "  so the solution may be too. Both readings are written.\n"
        "    physical -- solution holds physical/mean; submit the prediction as is\n"
        "    encoded  -- solution holds (a*physical + floor)/mean; re-encode before dividing\n"
        "  A persistence probe on the leaderboard settles it in one slot; until it does,\n"
        "  note that under the encoded reading a wrong slope largely cancels, because the\n"
        "  same slope divides the input and multiplies the output."
    )
    written = {}
    p = predict_test(keep, lam, TEST_FLOOR, clip=True, slope=test_slope)
    for space in ("physical", "encoded"):
        q = {sc: v.copy() for sc, v in p.items()}
        if space == "encoded":
            for sc in q:
                q[sc] = q[sc] * test_slope[:2] + TEST_FLOOR[:2] / div
        values = np.empty((len(template), 2))
        values[is126] = q["ssp126"][site[is126], age_pos[is126]]
        values[~is126] = q["ssp585"][site[~is126], age_pos[~is126]]
        frame = submission.assemble(values, template, already_scaled=True)
        suffix = "" if args.slope_mode == "uniform" else f"_{args.slope_mode}"
        path = WORK / f"submission_direct_{space}{suffix}.csv"
        submission.write(frame, path)
        reread = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
        report = submission.validate(reread, template)
        print(f"\n  {path}  ({path.stat().st_size:,} bytes)")
        print(report.render())
        if not report.ok:
            raise SystemExit(f"{path} did not validate")
        written[space] = str(path)

    (WORK / "metrics" / "submission.json").write_text(
        json.dumps(
            {
                "threshold": chosen_t,
                "features_kept": int(len(keep)),
                "lambda": lam,
                "offline_public": with_clip,
                "sweep": sweep,
                "decode_costs": decode_costs,
                "test_floor": TEST_FLOOR.tolist(),
                "slope_mode": args.slope_mode,
                "test_slope": test_slope.tolist(),
                "files": written,
            },
            indent=2,
            default=float,
        )
    )
    print(f"\n  primary: {written['physical']}")
    print("  nothing was submitted; `kaggle competitions submit` was not run")


if __name__ == "__main__":
    main()
