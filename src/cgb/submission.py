"""Assemble, validate and score a CarbonGlobe submission.

Three things are easy to get wrong here and all three are cheap to assert, so
this module asserts them rather than trusting the pipeline that produced a frame.

**The scaling.** The competition scores ``height`` and ``agb`` *after each has been
divided by its global mean*. Those divisors are not free parameters: they are
``y_mean[0]`` and ``y_mean[1]`` of ``cache/data_stats.npz``, the training set's own
per-variable means, and the README quotes them rounded to 10.90645 and 3.39567.
:func:`check_divisors` proves the README's constants and the shipped statistics are
the same numbers before anything divides by them.

An unscaled submission is not merely worse, it is catastrophically worse, and by a
predictable amount: submitting raw ``height`` when the solution holds ``height/M``
inflates every residual by a factor of roughly ``M``, so the MSE contribution rises
by roughly ``M**2`` -- about 119x for height, about 11x for agb.
``scripts/prove_metric.py`` measures that rather than reasoning about it.

**The row contract.** Kaggle will not score a file whose row set is not the
template's. 185,130 rows, ``{scenario}_{site}_{age}``, in exactly the order
``sample_submission.csv`` uses. A prediction dropped in a join is a wrong answer at
full weight, and a duplicate is worse because whichever copy wins is arbitrary.

**The id parse.** ``ssp126_0_1`` splits into three fields on ``_`` -- but so does
``ssp126`` itself contain no underscore, which is the only reason a naive split
works. :func:`parse_ids` splits from the left exactly twice and checks every field
against the known scenario, site and age sets, so a renamed scenario breaks loudly
instead of silently reordering the file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .io import CACHE, SCENARIOS, SEED_AGES, load_stats

#: Columns of the submission file, in template order.
SUBMISSION_COLUMNS: list[str] = ["id", "height", "agb"]

#: Measured row count of ``sample_submission.csv``: 2 scenarios x 6171 sites x 15 ages.
EXPECTED_ROWS = 185_130
EXPECTED_SITES = 6171

#: The divisors the README states, to the precision the README states them.
README_DIVISORS = {"height": 10.90645, "agb": 3.39567}

#: Loosest bound that still catches a pipeline that forgot to scale. Scaled
#: height is height/10.9, and the largest training height is far below 120 m.
SCALED_LIMITS = (-5.0, 15.0)


def divisors() -> dict[str, float]:
    """Return the scoring divisors, taken from the shipped training statistics."""
    y_mean = load_stats()["y_mean"]
    return {"height": float(y_mean[0]), "agb": float(y_mean[1])}


def check_divisors() -> dict[str, tuple[float, float]]:
    """Assert the shipped ``y_mean`` agrees with the README's rounded constants."""
    shipped = divisors()
    out: dict[str, tuple[float, float]] = {}
    for name, stated in README_DIVISORS.items():
        got = shipped[name]
        if abs(got - stated) > 0.5 * 10 ** -5:
            raise AssertionError(
                f"{name}: README says {stated}, data_stats.npz says {got!r}; "
                "these must be the same number rounded"
            )
        out[name] = (got, stated)
    return out


def load_template(path: Path | str | None = None) -> pd.DataFrame:
    """Read ``sample_submission.csv`` with the id column kept as text."""
    path = Path(path) if path is not None else CACHE / "sample_submission.csv"
    frame = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
    if list(frame.columns) != SUBMISSION_COLUMNS:
        raise AssertionError(f"{path.name}: columns are {list(frame.columns)}")
    return frame


def parse_ids(ids: "pd.Series[str] | np.ndarray") -> pd.DataFrame:
    """Split ``{scenario}_{site}_{age}`` into typed columns, validating each field."""
    series = pd.Series(np.asarray(ids), dtype=str)
    parts = series.str.split("_", n=2, expand=True)
    if parts.shape[1] != 3:
        raise AssertionError(f"ids do not split into 3 fields: {series.iloc[0]!r}")
    scenario, site, age = parts[0], parts[1], parts[2]

    unknown = set(scenario.unique()) - set(SCENARIOS)
    if unknown:
        raise AssertionError(f"unknown scenario(s) in ids: {sorted(unknown)}")
    if not site.str.fullmatch(r"\d+").all():
        raise AssertionError("some site fields are not plain integers")
    if not age.str.fullmatch(r"\d+").all():
        raise AssertionError("some age fields are not plain integers")

    out = pd.DataFrame(
        {
            "scenario": scenario.to_numpy(),
            "site": site.astype(np.int64).to_numpy(),
            "age": age.astype(np.int64).to_numpy(),
        }
    )
    bad_age = set(out["age"].unique()) - set(SEED_AGES)
    if bad_age:
        raise AssertionError(f"unknown seed age(s) in ids: {sorted(bad_age)}")
    if out["site"].min() < 0 or out["site"].max() >= EXPECTED_SITES:
        raise AssertionError(
            f"site ids run {out['site'].min()}..{out['site'].max()}, expected 0..{EXPECTED_SITES - 1}"
        )
    return out


@dataclass
class Report:
    """Outcome of :func:`validate`. ``errors`` non-empty means do not submit."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [f"  {k}: {v}" for k, v in self.stats.items()]
        lines += [f"  WARNING  {t}" for t in self.warnings]
        lines += [f"  ERROR    {t}" for t in self.errors]
        lines.append(f"  verdict: {'OK' if self.ok else 'REJECTED'}")
        return "\n".join(lines)


def assemble(
    predictions: np.ndarray | pd.DataFrame,
    template: pd.DataFrame | None = None,
    *,
    already_scaled: bool = False,
) -> pd.DataFrame:
    """Build a submission frame in template order.

    ``predictions`` is either an ``(n, 2)`` array already in template row order, or
    a frame carrying ``scenario``/``site``/``age`` plus ``height``/``agb``, which is
    joined one-to-one onto the template so a missing or duplicated prediction raises
    instead of silently shifting the file.

    ``already_scaled=False`` (the default) means the caller hands over **physical**
    year-40 height and agb and this function divides. The default is the safe one:
    forgetting to scale is the documented way to score as if completely wrong, and
    a caller who has already scaled has to say so out loud.
    """
    template = load_template() if template is None else template
    div = divisors()

    if isinstance(predictions, pd.DataFrame):
        key = parse_ids(template["id"])
        need = ["scenario", "site", "age", "height", "agb"]
        missing = [c for c in need if c not in predictions.columns]
        if missing:
            raise KeyError(f"predictions are missing {missing}")
        merged = key.merge(
            predictions.loc[:, need], on=["scenario", "site", "age"], how="left", validate="one_to_one"
        )
        if len(merged) != len(template):
            raise AssertionError(f"the merge changed the row count, {len(template)} -> {len(merged)}")
        values = merged.loc[:, ["height", "agb"]].to_numpy(dtype=np.float64)
    else:
        values = np.asarray(predictions, dtype=np.float64)
        if values.shape != (len(template), 2):
            raise AssertionError(f"predictions are {values.shape}, expected {(len(template), 2)}")

    if not already_scaled:
        values = values / np.array([div["height"], div["agb"]], dtype=np.float64)

    return pd.DataFrame(
        {"id": template["id"].to_numpy(), "height": values[:, 0], "agb": values[:, 1]}
    )


def validate(submission: pd.DataFrame, template: pd.DataFrame | None = None) -> Report:
    """Check a submission against the template and against the scaling, without scoring."""
    template = load_template() if template is None else template
    report = Report()

    if list(submission.columns) != SUBMISSION_COLUMNS:
        report.errors.append(f"columns are {list(submission.columns)}, expected {SUBMISSION_COLUMNS}")
        return report

    report.stats["rows"] = f"{len(submission):,}"
    if len(template) != EXPECTED_ROWS:
        report.errors.append(f"the template itself has {len(template):,} rows, measured {EXPECTED_ROWS:,}")
    if len(submission) != len(template):
        report.errors.append(f"{len(submission):,} rows against {len(template):,} in the template")
        return report

    got = submission["id"].astype(str).to_numpy()
    want = template["id"].astype(str).to_numpy()
    if not np.array_equal(got, want):
        if set(got.tolist()) == set(want.tolist()):
            report.errors.append("the ids are the right set but in a different order")
        else:
            missing = sorted(set(want.tolist()) - set(got.tolist()))
            extra = sorted(set(got.tolist()) - set(want.tolist()))
            report.errors.append(
                f"id mismatch: {len(missing):,} missing, {len(extra):,} unknown"
                + (f", first missing {missing[0]}" if missing else "")
            )
        return report

    key = parse_ids(got)
    report.stats["scenarios"] = dict(key["scenario"].value_counts().sort_index())
    report.stats["sites"] = f"{key['site'].nunique():,}"
    report.stats["ages"] = f"{key['age'].nunique()}"
    dup = int(len(key) - len(key.drop_duplicates(["scenario", "site", "age"])))
    if dup:
        report.errors.append(f"{dup:,} duplicated (scenario, site, age) keys")

    for column in ("height", "agb"):
        values = pd.to_numeric(submission[column], errors="coerce").to_numpy(dtype=np.float64)
        n_nan = int(np.isnan(values).sum())
        n_inf = int(np.isinf(values).sum())
        if n_nan:
            report.errors.append(f"{column}: {n_nan:,} rows are unanswered (NaN or unparseable)")
        if n_inf:
            report.errors.append(f"{column}: {n_inf:,} values are infinite")
        finite = values[np.isfinite(values)]
        if finite.size:
            report.stats[column] = (
                f"min {finite.min():.4f}  mean {finite.mean():.4f}  max {finite.max():.4f}"
            )
            low, high = SCALED_LIMITS
            out_of_range = int(((finite < low) | (finite > high)).sum())
            if out_of_range:
                report.errors.append(
                    f"{column}: {out_of_range:,} values outside [{low}, {high}] -- "
                    f"this is what an UNSCALED submission looks like; divide by "
                    f"{divisors()[column]:.5f}"
                )
            if np.all(finite == 0.0):
                report.errors.append(f"{column}: every value is exactly zero, so nothing answered it")

    return report


def score(submission: pd.DataFrame, solution: pd.DataFrame) -> dict[str, float]:
    """Reproduce Kaggle's built-in MSE over the two pre-scaled columns.

    Both frames must already carry **scaled** values, which is what the solution
    file holds and what a submission is required to hold. The competition score is
    the mean of the squared errors over both columns pooled, which is identical to
    ``0.5 * (MSE_height + MSE_agb)`` when both columns have the same row count --
    both are returned so the identity is visible rather than assumed.
    """
    if not np.array_equal(submission["id"].astype(str).to_numpy(), solution["id"].astype(str).to_numpy()):
        raise AssertionError("submission and solution ids differ or are differently ordered")
    pred = submission.loc[:, ["height", "agb"]].to_numpy(dtype=np.float64)
    true = solution.loc[:, ["height", "agb"]].to_numpy(dtype=np.float64)
    err = pred - true
    mse_h = float(np.mean(err[:, 0] ** 2))
    mse_a = float(np.mean(err[:, 1] ** 2))
    pooled = float(np.mean(err**2))
    return {
        "score": pooled,
        "mse_height": mse_h,
        "mse_agb": mse_a,
        "half_sum": 0.5 * (mse_h + mse_a),
        "rrmse": float(np.sqrt(pooled)),
    }


def write(submission: pd.DataFrame, path: Path | str) -> Path:
    """Write the submission in template column order, no index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.loc[:, SUBMISSION_COLUMNS].to_csv(path, index=False, float_format="%.6f")
    return path
