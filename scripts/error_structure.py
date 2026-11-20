#!/usr/bin/env python
"""Where the score is actually decided: which seed ages, which column, which sites.

A mean squared error over 185,130 rows hides which rows it came from. This script
opens that up for the best offline model (the direct year-40 ridge whose per-site
residuals ``scripts/rollout_drift.py`` wrote out), and for persistence as the null,
on the same site-held-out fold.

Three decompositions, each of which changes what is worth working on next:

**by seed age** -- if one age band carries a quarter of the error, a model that is
only good at equilibrium forests is leaving that quarter on the table.

**by column** -- height and agb are pooled into one mean, so whichever column has the
larger scaled variance silently sets the agenda.

**by site** -- 6,171 sites each contribute their thirty rows. If the top 1% of sites
carry a tenth of the error, the leaderboard is largely a contest over a few hundred
hard places, and the site bootstrap's width is set by whether those sites are drawn.

    ./.venv/bin/python scripts/error_structure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, holdout, io  # noqa: E402

METRICS = io.ROOT / "work" / "metrics"


def lorenz(per_site: np.ndarray) -> dict[str, float]:
    """Share of total squared error carried by the worst 1 / 5 / 10 / 25 % of sites."""
    order = np.sort(per_site)[::-1]
    total = order.sum()
    out = {}
    for frac in (0.01, 0.05, 0.10, 0.25, 0.50):
        k = max(1, int(round(len(order) * frac)))
        out[f"top{int(frac * 100)}pct"] = float(order[:k].sum() / total)
    return out


def gini(per_site: np.ndarray) -> float:
    x = np.sort(per_site)
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main() -> None:
    div = cache.scoring_divisors()
    ages = io.SEED_AGES
    split = holdout.make_split(seed=42)
    hold = split.holdout
    pub_mask = np.isin(hold, split.public)

    # --- the two models, as per-(site, age, column) squared error -------------
    direct = np.stack(
        [np.load(METRICS / f"direct_err40_age{age:03d}.npy") for age in ages], axis=1
    )  # (6171, 15, 2)
    y0 = cache.state_block(0)[:, hold, :2] / div
    y40 = cache.state_block(40)[:, hold, :2] / div
    persist = np.transpose((y0 - y40) ** 2, (1, 0, 2))
    print("== models compared ==")
    print(f"  direct year-40 ridge   public {direct[pub_mask].mean():.5f}")
    print(f"  persistence            public {persist[pub_mask].mean():.5f}")
    print(f"  rows per fold: {pub_mask.sum():,} sites x {len(ages)} ages x 2 columns")

    models = {"direct": direct, "persistence": persist}
    payload: dict[str, dict] = {}

    # --- by seed age ---------------------------------------------------------
    print("\n== by seed age, public fold ==")
    print(
        f"  {'age':>4s} {'direct mse':>11s} {'share':>7s} {'cum':>7s} | "
        f"{'persist mse':>12s} {'share':>7s} {'cum':>7s} | {'ratio':>6s}"
    )
    shares = {}
    for name, err in models.items():
        e = err[pub_mask]
        shares[name] = np.array([e[:, j, :].sum() for j in range(len(ages))])
        shares[name] = shares[name] / shares[name].sum()
    cum_d = cum_p = 0.0
    for j, age in enumerate(ages):
        d = float(direct[pub_mask][:, j, :].mean())
        p = float(persist[pub_mask][:, j, :].mean())
        cum_d += shares["direct"][j]
        cum_p += shares["persistence"][j]
        print(
            f"  {age:4d} {d:11.5f} {shares['direct'][j]:7.1%} {cum_d:7.1%} | "
            f"{p:12.5f} {shares['persistence'][j]:7.1%} {cum_p:7.1%} | {p / d:6.2f}x"
        )
    young = slice(0, 4)  # ages 1..30
    old = slice(6, 15)  # ages 100..500
    print(
        f"\n  ages 1-30   carry {shares['direct'][young].sum():.1%} of the direct model's error "
        f"({shares['persistence'][young].sum():.1%} of persistence's)"
    )
    print(
        f"  ages 100-500 carry {shares['direct'][old].sum():.1%} of the direct model's error "
        f"({shares['persistence'][old].sum():.1%} of persistence's)"
    )
    payload["age_share"] = {k: v.tolist() for k, v in shares.items()}

    # --- by column -----------------------------------------------------------
    print("\n== by column, public fold ==")
    for name, err in models.items():
        e = err[pub_mask]
        h, a = float(e[..., 0].mean()), float(e[..., 1].mean())
        print(f"  {name:12s} height {h:8.5f} ({h / (h + a):5.1%})   agb {a:8.5f} ({a / (h + a):5.1%})")
    truth40 = np.transpose(y40, (1, 0, 2))[pub_mask]
    print(
        f"  scaled truth spread: height sd {truth40[..., 0].std():.4f}, "
        f"agb sd {truth40[..., 1].std():.4f} -- agb is the wider column, "
        f"so it sets more of the agenda"
    )

    # --- by site -------------------------------------------------------------
    print("\n== by site, public fold: concentration of the squared error ==")
    payload["site_concentration"] = {}
    for name, err in models.items():
        per_site = err[pub_mask].reshape(pub_mask.sum(), -1).sum(axis=1)
        share = lorenz(per_site)
        payload["site_concentration"][name] = {**share, "gini": gini(per_site)}
        print(
            f"  {name:12s} worst  1% of sites carry {share['top1pct']:6.1%}, "
            f"5% {share['top5pct']:6.1%}, 10% {share['top10pct']:6.1%}, "
            f"25% {share['top25pct']:6.1%}   gini {gini(per_site):.3f}"
        )
    print(
        "  (a uniform spread would give 1% / 5% / 10% / 25%; anything far above that "
        "means a few hundred sites decide the score)"
    )

    # --- what makes a hard site hard ----------------------------------------
    per_site = direct[pub_mask].reshape(pub_mask.sum(), -1).sum(axis=1)
    hold_pub = hold[pub_mask]
    growth = np.abs(
        np.transpose(y40, (1, 0, 2))[pub_mask] - np.transpose(y0, (1, 0, 2))[pub_mask]
    ).mean(axis=(1, 2))
    level = np.transpose(y40, (1, 0, 2))[pub_mask].mean(axis=(1, 2))
    clim = cache.climate_summary()
    tair = clim["mean"][hold_pub, cache.D_TAIR]
    precip = clim["mean"][hold_pub, cache.D_PRECIP]

    print("\n== what the hard sites have in common (Spearman, public fold) ==")

    def spearman(a: np.ndarray, b: np.ndarray) -> float:
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    for label, v in (
        ("mean |year40 - year0| (how much the site moves)", growth),
        ("mean year-40 level (how much biomass is there)", level),
        ("40-year mean air temperature", tair),
        ("40-year mean precipitation", precip),
    ):
        print(f"  site squared error vs {label:52s} {spearman(per_site, v):+.3f}")

    worst = np.argsort(per_site)[::-1][: max(1, len(per_site) // 100)]
    rest = np.argsort(per_site)[::-1][max(1, len(per_site) // 100) :]
    print(
        f"\n  the worst 1% ({len(worst)} sites): mean year-40 level {level[worst].mean():.2f} "
        f"vs {level[rest].mean():.2f} elsewhere; mean movement {growth[worst].mean():.2f} "
        f"vs {growth[rest].mean():.2f}; mean temperature {tair[worst].mean():.1f} K "
        f"vs {tair[rest].mean():.1f} K"
    )

    # --- how much of the fold noise is site draw -----------------------------
    boot = holdout.site_bootstrap(direct[pub_mask].reshape(pub_mask.sum(), -1), n_boot=4000, seed=11)
    print(
        f"\n== resolving power on this model ==\n"
        f"  direct model public {boot['score']:.5f}  se {boot['se']:.5f} "
        f"({boot['se'] / boot['score']:.2%} of the score)  95% [{boot['lo95']:.5f}, {boot['hi95']:.5f}]\n"
        f"  private {direct[~pub_mask].mean():.5f}  "
        f"fold gap {abs(direct[pub_mask].mean() - direct[~pub_mask].mean()) / direct[pub_mask].mean():.2%}"
    )

    (METRICS / "structure.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {METRICS / 'structure.json'}")


if __name__ == "__main__":
    main()
