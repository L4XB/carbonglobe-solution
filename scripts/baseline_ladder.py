#!/usr/bin/env python
"""What the cheap answers score, before anyone builds a model.

Every rung is fitted only on the 47,981 fit sites of ``cgb.holdout.make_split(42)``
and scored on the held-out 6,171 -- the competition's own site count -- split into
public-like and private-like folds. Scores are in the competition's scaled units
(height/10.90645, agb/3.39567), so they sit on the same axis as the leaderboard.

The ladder, cheapest first:

``zeros``            all zeros. "Completely wrong", the reference the trap is measured against.
``one``              constant 1.0 on both columns. The targets are divided by their global
                     means, so 1.0 is the know-nothing answer that needs no data at all.
``best_constant``    the fit set's own mean year-40 state, one number per column. The best
                     any age-blind, site-blind constant can do.
``age_mean``         the growth curve on seed age alone: fit-set mean year-40 state per age.
``persistence``      year 40 = year 0. No model, but it uses the site.
``age_affine``       per age, per variable, OLS on that variable's own year-0 value.
``age_state``        per age, OLS on the whole 7-variable year-0 state.
``age_state2``       the same plus squares and a log term -- is linearity the binding constraint?
``age_state_clim3``  + three 40-year forcing aggregates (temperature, precipitation, CO2).
``age_state_climfull`` + all 136 forcing means and all 136 forcing trends, ridged.

Each rung's gain over the one below it is reported with a **paired** site bootstrap,
because the fold-to-fold noise (~3% of the score) is larger than some of the gaps and
an unpaired comparison cannot resolve them.

    ./.venv/bin/python scripts/baseline_ladder.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, holdout, io  # noqa: E402

OUT = io.ROOT / "work" / "metrics"
RIDGE_GRID = (1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class Scored:
    name: str
    pred: np.ndarray  # (n_holdout_sites, n_ages, 2), scaled units


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge coefficients for standardised ``x`` with an unpenalised intercept."""
    n, p = x.shape
    design = np.empty((n, p + 1), dtype=np.float64)
    design[:, :p] = x
    design[:, p] = 1.0
    gram = design.T @ design
    penalty = np.eye(p + 1) * lam
    penalty[p, p] = 0.0
    return np.linalg.solve(gram + penalty, design.T @ y)


def ridge_apply(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return x @ beta[:-1] + beta[-1]


class Standardiser:
    """Centre and scale on the fit set; a zero-variance column becomes a zero column."""

    def __init__(self, x: np.ndarray) -> None:
        self.mean = x.mean(axis=0)
        sd = x.std(axis=0)
        self.scale = np.where(sd > 1e-12, sd, 1.0)
        self.dead = sd <= 1e-12

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = (x - self.mean) / self.scale
        out[:, self.dead] = 0.0
        return out


def fit_linear_rung(
    features: list[np.ndarray],
    target: np.ndarray,
    fit_sites: np.ndarray,
    eval_sites: np.ndarray,
    *,
    lam_grid: tuple[float, ...] = RIDGE_GRID,
    inner_fraction: float = 0.2,
    seed: int = 1234,
) -> tuple[np.ndarray, float]:
    """Fit one linear rung per seed age and return its holdout prediction.

    ``features`` is one ``(n_sites, p_age)`` design per seed age -- the design may
    differ by age because the year-0 state does. ``target`` is
    ``(n_ages, n_sites, 2)`` in scaled units. The ridge strength is chosen on an
    inner split of the **fit** sites, never on the holdout.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(fit_sites))
    cut = int(round(len(fit_sites) * inner_fraction))
    inner_val = fit_sites[perm[:cut]]
    inner_fit = fit_sites[perm[cut:]]

    scores = []
    for lam in lam_grid:
        total = 0.0
        for j, design in enumerate(features):
            std = Standardiser(design[inner_fit])
            beta = ridge_fit(std(design[inner_fit]), target[j][inner_fit], lam)
            resid = ridge_apply(std(design[inner_val]), beta) - target[j][inner_val]
            total += float((resid**2).mean())
        scores.append(total / len(features))
    lam = lam_grid[int(np.argmin(scores))]

    pred = np.empty((len(eval_sites), len(features), 2), dtype=np.float64)
    for j, design in enumerate(features):
        std = Standardiser(design[fit_sites])
        beta = ridge_fit(std(design[fit_sites]), target[j][fit_sites], lam)
        pred[:, j, :] = ridge_apply(std(design[eval_sites]), beta)
    return pred, lam


def squared_error(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """``(n_sites, n_ages, 2)`` squared residuals."""
    return (pred - truth) ** 2


def fold_score(err: np.ndarray) -> float:
    return float(err.mean())


def paired_bootstrap(
    err_a: np.ndarray, err_b: np.ndarray, *, n_boot: int = 2000, seed: int = 7
) -> dict[str, float]:
    """Interval on ``score(a) - score(b)`` resampling the **same** sites for both."""
    a = err_a.reshape(err_a.shape[0], -1).mean(axis=1)
    b = err_b.reshape(err_b.shape[0], -1).mean(axis=1)
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    draws = diff[idx].mean(axis=1)
    return {
        "delta": float(diff.mean()),
        "lo95": float(np.percentile(draws, 2.5)),
        "hi95": float(np.percentile(draws, 97.5)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    div = cache.scoring_divisors()
    ages = io.SEED_AGES
    n_ages = len(ages)

    print("== inputs ==")
    t0 = time.time()
    y0 = cache.state_block(0)  # (15, 54152, 7) physical
    y40 = cache.state_block(40)[:, :, :2]  # (15, 54152, 2) physical
    target = y40 / div  # scaled, the competition's units
    print(f"  y0 {y0.shape}  y40 {y40.shape}  loaded in {time.time() - t0:.1f}s")
    print(f"  divisors height {div[0]!r}  agb {div[1]!r}")

    clim = cache.climate_summary()
    tair = clim["mean"][:, cache.D_TAIR]
    precip = clim["mean"][:, cache.D_PRECIP]
    co2 = clim["mean"][:, cache.D_CO2]
    print(
        f"  driver {cache.D_TAIR} 40-yr mean over sites: {tair.mean():.2f} "
        f"[{tair.min():.2f}, {tair.max():.2f}] -- air temperature in K\n"
        f"  driver {cache.D_PRECIP}: {precip.mean():.2f} [{precip.min():.2f}, {precip.max():.2f}]\n"
        f"  driver {cache.D_CO2}: {co2.mean():.2f} [{co2.min():.2f}, {co2.max():.2f}], "
        f"across-site sd of its trend {clim['trend'][:, cache.D_CO2].std():.4f} ppm/yr -- CO2"
    )

    split = holdout.make_split(seed=42)
    print("\n== split ==")
    print("  " + split.describe())
    hold = split.holdout
    pub_mask = np.isin(hold, split.public)
    prv_mask = np.isin(hold, split.private)
    print(f"  holdout rows scored: {len(hold) * n_ages * 2:,} (sites x 15 ages x 2 columns)")

    truth = np.transpose(target[:, hold, :], (1, 0, 2))  # (n_hold, 15, 2)

    # ---------------------------------------------------------------- rungs
    rungs: list[Scored] = []

    rungs.append(Scored("zeros", np.zeros_like(truth)))
    rungs.append(Scored("one", np.ones_like(truth)))

    const = target[:, split.fit, :].reshape(-1, 2).mean(axis=0)
    print(f"\n  best_constant (fit-set mean, scaled): height {const[0]:.4f}  agb {const[1]:.4f}")
    rungs.append(Scored("best_constant", np.broadcast_to(const, truth.shape).copy()))

    age_mean = target[:, split.fit, :].mean(axis=1)  # (15, 2)
    rungs.append(Scored("age_mean", np.broadcast_to(age_mean, truth.shape).copy()))

    persist = np.transpose(y0[:, hold, :2] / div, (1, 0, 2))
    rungs.append(Scored("persistence", persist))

    tgt = np.transpose(target, (0, 1, 2))  # (15, 54152, 2) -- per-age design targets

    # age_affine: per age, per variable, on that variable's own year-0 value only.
    pred = np.empty_like(truth)
    for j in range(n_ages):
        for v in range(2):
            x = y0[j, :, v : v + 1]
            std = Standardiser(x[split.fit])
            beta = ridge_fit(std(x[split.fit]), tgt[j][split.fit, v : v + 1], 0.0)
            pred[:, j, v] = ridge_apply(std(x[hold]), beta)[:, 0]
    rungs.append(Scored("age_affine", pred))

    state = [y0[j] for j in range(n_ages)]
    pred, lam = fit_linear_rung(state, tgt, split.fit, hold)
    print(f"  age_state ridge lambda {lam:g}")
    rungs.append(Scored("age_state", pred))

    state2 = [
        np.concatenate([s, s**2, np.log1p(np.clip(s, 0.0, None))], axis=1) for s in state
    ]
    pred, lam = fit_linear_rung(state2, tgt, split.fit, hold)
    print(f"  age_state2 ridge lambda {lam:g}")
    rungs.append(Scored("age_state2", pred))

    clim3 = np.stack([tair, precip, co2], axis=1)
    state_c3 = [np.concatenate([s, clim3], axis=1) for s in state2]
    pred, lam = fit_linear_rung(state_c3, tgt, split.fit, hold)
    print(f"  age_state_clim3 ridge lambda {lam:g}")
    rungs.append(Scored("age_state_clim3", pred))

    climfull = np.concatenate([clim["mean"], clim["trend"]], axis=1)  # (54152, 272)
    state_cf = [np.concatenate([s, climfull], axis=1) for s in state2]
    pred, lam = fit_linear_rung(state_cf, tgt, split.fit, hold)
    print(f"  age_state_climfull ridge lambda {lam:g}  ({state_cf[0].shape[1]} features)")
    rungs.append(Scored("age_state_climfull", pred))

    # ---------------------------------------------------------------- scoring
    print("\n== the ladder, scaled units, same axis as the leaderboard ==")
    print(
        f"  {'rung':22s} {'public':>9s} {'private':>9s} {'95% CI (public)':>24s} "
        f"{'mse_h':>8s} {'mse_agb':>8s}"
    )
    results: dict[str, dict] = {}
    errs: dict[str, np.ndarray] = {}
    for rung in rungs:
        err = squared_error(rung.pred, truth)
        errs[rung.name] = err
        pub, prv = err[pub_mask], err[prv_mask]
        boot = holdout.site_bootstrap(pub.reshape(pub.shape[0], -1), n_boot=2000, seed=7)
        row = {
            "public": fold_score(pub),
            "private": fold_score(prv),
            "lo95": boot["lo95"],
            "hi95": boot["hi95"],
            "se": boot["se"],
            "mse_height": float(pub[..., 0].mean()),
            "mse_agb": float(pub[..., 1].mean()),
        }
        results[rung.name] = row
        print(
            f"  {rung.name:22s} {row['public']:9.5f} {row['private']:9.5f} "
            f"[{row['lo95']:8.5f}, {row['hi95']:8.5f}] {row['mse_height']:8.5f} {row['mse_agb']:8.5f}"
        )

    print("\n== each rung against the one below it, paired over the same sites ==")
    order = [r.name for r in rungs]
    for lower, upper in zip(order, order[1:]):
        d = paired_bootstrap(errs[upper][pub_mask], errs[lower][pub_mask])
        verdict = "resolved" if d["hi95"] < 0 else ("no gain" if d["lo95"] > 0 else "NOT resolved")
        rel = d["delta"] / results[lower]["public"]
        print(
            f"  {upper:22s} - {lower:22s} {d['delta']:+9.5f} "
            f"[{d['lo95']:+8.5f}, {d['hi95']:+8.5f}] {rel:+7.1%}  {verdict}"
        )

    print("\n== per seed age, public fold: score, and share of the total squared error ==")
    shown = ["persistence", "age_mean", "age_state", "age_state_clim3", "age_state_climfull"]
    header = "  age  " + "".join(f"{n[:16]:>18s}" for n in shown)
    print(header)
    per_age: dict[str, list[float]] = {n: [] for n in shown}
    for j, age in enumerate(ages):
        cells = []
        for n in shown:
            s = float(errs[n][pub_mask][:, j, :].mean())
            per_age[n].append(s)
            share = s / (errs[n][pub_mask].mean() * n_ages)
            cells.append(f"{s:9.4f} ({share:5.1%})")
        print(f"  {age:>4d} " + "".join(f"{c:>18s}" for c in cells))

    print("\n== height vs agb, public fold ==")
    for n in shown:
        e = errs[n][pub_mask]
        h, a = float(e[..., 0].mean()), float(e[..., 1].mean())
        print(f"  {n:22s} height {h:8.5f} ({h / (h + a):5.1%})   agb {a:8.5f} ({a / (h + a):5.1%})")

    payload = {
        "split_seed": 42,
        "fit_sites": int(len(split.fit)),
        "public_sites": int(len(split.public)),
        "private_sites": int(len(split.private)),
        "divisors": div.tolist(),
        "rungs": results,
        "per_age_public": {k: v for k, v in per_age.items()},
        "seed_ages": list(ages),
    }
    (OUT / "ladder.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT / 'ladder.json'}")


if __name__ == "__main__":
    main()
