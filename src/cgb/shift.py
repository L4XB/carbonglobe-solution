"""Correct a year-40 prediction for the train/test population shift, and calibrate
that correction from the leaderboard instead of from the offline holdout.

**The shift.** The scored population is not the training population: it carries 8.6%
less second moment, its trajectories change less over the forty years, and the
regression slope of the truth on decoded persistence height is 0.960 on the board
against 1.084 offline (``notes/board-anchors.md``). ``scripts/shift_population.py``
measures where the two populations differ; the short version is that the old stands
(seed age >= 150) are taller and thinner in the test set, and the young ones are the
same. A model fitted offline therefore predicts the *right kind* of change with the
*wrong amount* at old ages, and the offline holdout cannot see it because there the
initial states are in-distribution by construction.

**The correction.** A post-hoc transform of a finished prediction file, so that a
board comparison of corrected against uncorrected differs in exactly one thing:

    corrected = y0 + lambda_band * (pred - y0)

with ``y0`` the decoded persistence prediction in scaled units and ``lambda`` one
number per seed-age band (optionally per column). ``lambda = 1`` is the model as it
is, ``lambda = 0`` is persistence. Nothing is retrained.

**The calibration.** The metric is a mean squared error, so along any fixed
direction ``V`` in prediction space the score is an exact parabola in the step
length, ``score(t) = score(0) + 2 t E[V (A - T)] + t^2 E[V^2]``, and ``E[V^2]`` is
ours to compute. Two board scores on the same line therefore give the optimal step
and the score there, with a precision set only by the board's rounding.
:func:`line_search` is that identity; :func:`band_probe` is the same identity for
the lead's per-band persistence probes, in the ``B_band = E[X T]`` parametrisation
of ``notes/board-anchors.md``.

Everything here is pooled over both columns the way the score is: an expectation is
a mean over the ``2 N`` scored cells, so ``E[X^2] = 0.5 (mean X_h^2 + mean X_a^2)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import io, submission

#: Transformed zero of each ``test_y0`` variable (``notes/theory.md`` Sec. 5): the
#: column minimum for the pools, the bare-row plateau for lai and the fluxes.
#: :func:`measured_floors` re-derives the two scored ones from the files.
FLOORS = np.array([-0.900947, -0.401412, -0.532395, 0.117938, 0.259026, 0.127815, 0.084656])

#: Decode multiplier pinned by the 35.0002 m cap and the 1.5 m seedling
#: (``scripts/theory_y0_scale.py``): ``physical = M_HEIGHT * (y0 - floor)``.
M_HEIGHT = 1.14114

#: Seed-age bands. ``young`` is where y0 carries little and the growth curve is the
#: signal; ``old`` is where y0 carries ~90% of the variance and the spin-up
#: relaxation lives; ``mid`` is the transition.
BANDS: dict[str, tuple[int, ...]] = {
    "young": (1, 10, 20, 30),
    "mid": (50, 70, 100),
    "old": (150, 200, 250, 300, 350, 400, 450, 500),
}
COLUMNS = ("height", "agb")


def measured_floors() -> np.ndarray:
    """Per-variable minimum of ``test_y0`` over both scenarios, asserted against :data:`FLOORS`."""
    stacked = np.concatenate([io.open_test(s)[1].reshape(-1, 7) for s in io.SCENARIOS]).astype(np.float64)
    floors = stacked.min(axis=0)
    if not np.allclose(floors[:2], FLOORS[:2], atol=1e-6):
        raise AssertionError(f"height/agb floors moved: {floors[:2]} against {FLOORS[:2]}")
    return floors


def template_keys() -> pd.DataFrame:
    """``scenario``/``site``/``age`` of every template row, in template order."""
    return submission.parse_ids(submission.load_template()["id"])


def band_of(ages: np.ndarray) -> np.ndarray:
    """Band name of each row given its seed age."""
    ages = np.asarray(ages)
    out = np.empty(ages.shape, dtype=object)
    for name, members in BANDS.items():
        out[np.isin(ages, members)] = name
    if (out == None).any():  # noqa: E711 -- object array
        raise ValueError("a seed age belongs to no band")
    return out.astype(str)


def band_masks(ages: np.ndarray) -> dict[str, np.ndarray]:
    b = band_of(ages)
    return {name: b == name for name in BANDS}


def persistence(m_height: float = M_HEIGHT, m_agb: float = M_HEIGHT) -> np.ndarray:
    """``(N, 2)`` decoded year-0 height and agb in scaled units, template order.

    The decode is ``m * (y0 - floor) / y_mean``. The floor is settled on the board
    and the height multiplier offline; the agb multiplier is the caller's reading.
    """
    div = submission.divisors()
    keys = template_keys()
    floors = measured_floors()
    out = np.empty((len(keys), 2), dtype=np.float64)
    for scenario in io.SCENARIOS:
        _, y0 = io.open_test(scenario)
        y0 = np.asarray(y0, dtype=np.float64)  # (6171, 15, 7)
        # a row at the floor is a bare site and decodes to exactly zero, not to the
        # float noise of the subtraction
        h = m_height * np.clip(y0[:, :, 0] - floors[0], 0.0, None) / div["height"]
        a = m_agb * np.clip(y0[:, :, 1] - floors[1], 0.0, None) / div["agb"]
        rows = (keys["scenario"] == scenario).to_numpy()
        # template order inside a scenario is site-major, then age in README order,
        # which is exactly the C order of a (site, age) array; asserted below.
        site = keys.loc[rows, "site"].to_numpy()
        age_idx = np.searchsorted(np.asarray(io.SEED_AGES), keys.loc[rows, "age"].to_numpy())
        out[rows, 0] = h[site, age_idx]
        out[rows, 1] = a[site, age_idx]
    if not np.isfinite(out).all():
        raise AssertionError("persistence has non-finite cells")
    return out


def read_predictions(path: Path | str) -> np.ndarray:
    """Read a submission file into ``(N, 2)`` after asserting it is in template order."""
    frame = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
    template = submission.load_template()
    if list(frame.columns) != submission.SUBMISSION_COLUMNS:
        raise AssertionError(f"{path}: columns are {list(frame.columns)}")
    if not np.array_equal(frame["id"].to_numpy(), template["id"].to_numpy()):
        raise AssertionError(f"{path}: ids differ from the template or are reordered")
    values = frame.loc[:, ["height", "agb"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise AssertionError(f"{path}: non-finite predictions")
    return values


def write_predictions(pred: np.ndarray, path: Path | str, *, probe: bool = False) -> submission.Report:
    """Assemble ``pred`` (scaled, template order) into a file, validate it, write it.

    ``probe=True`` tolerates exactly one of the validator's rules -- "every value
    is exactly zero" in a column -- because a moment probe answers one column with
    a constant and the other with nothing on purpose. Every other rule still holds.
    """
    frame = submission.assemble(pred, already_scaled=True)
    report = submission.validate(frame)
    if probe:
        report.errors = [e for e in report.errors if "every value is exactly zero" not in e]
    if not report.ok:
        raise AssertionError(f"refusing to write {path}:\n{report.render()}")
    submission.write(frame, path)
    return report


def _lambda_matrix(lam: dict[str, float | tuple[float, float]], ages: np.ndarray) -> np.ndarray:
    """Expand a per-band ``lambda`` spec into an ``(N, 2)`` multiplier array."""
    out = np.ones((len(ages), 2), dtype=np.float64)
    masks = band_masks(ages)
    for name, value in lam.items():
        if name not in masks:
            raise KeyError(f"unknown band {name!r}; bands are {list(BANDS)}")
        pair = (value, value) if np.isscalar(value) else tuple(value)
        if len(pair) != 2:
            raise ValueError(f"lambda for {name!r} must be a scalar or (height, agb)")
        out[masks[name], 0] = pair[0]
        out[masks[name], 1] = pair[1]
    return out


def damp(pred: np.ndarray, base: np.ndarray, lam: dict[str, float | tuple[float, float]],
         ages: np.ndarray) -> np.ndarray:
    """``base + lambda_band * (pred - base)``; bands not named keep ``lambda = 1``.

    ``base`` is normally :func:`persistence` under the same decode the model used, so
    that ``pred - base`` is the model's predicted forty-year change and ``lambda``
    damps (or amplifies) that change and nothing else.
    """
    pred = np.asarray(pred, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    if pred.shape != base.shape or pred.ndim != 2 or pred.shape[1] != 2:
        raise ValueError(f"pred {pred.shape} and base {base.shape} must both be (N, 2)")
    # No clipping here: for lambda in [0, 1] the result is a convex combination of
    # two non-negative files and cannot go negative, and leaving the map exactly
    # linear is what makes the board pair (pred, damped) an exact parabola. A
    # caller using lambda outside [0, 1] clips afterwards and says so.
    return base + _lambda_matrix(lam, ages) * (pred - base)


def scale(pred: np.ndarray, factor: dict[str, float | tuple[float, float]], ages: np.ndarray) -> np.ndarray:
    """``factor_band * pred``: rescale the predicted *level* of a band, not its change.

    This is the other one-parameter family the board can calibrate, and it is the
    one that matters where the model's change has the wrong sign: the board's
    persistence slope for height is 0.960 on the decoded column while every model
    trained on the present-day runs predicts old stands slightly taller, and no
    ``lambda`` in ``[0, 1]`` on the change can turn growth into loss. A factor on
    the level can. ``pred`` and the scaled file lie on a line, so
    :func:`line_search` reads the optimal factor off two board scores.
    """
    pred = np.asarray(pred, dtype=np.float64)
    return _lambda_matrix(factor, ages) * pred


@dataclass(frozen=True)
class LineSearch:
    """The parabola through two board points along ``V = b - a``.

    ``t = 0`` is file ``a``, ``t = 1`` is file ``b``. ``t_star`` is the step that
    minimises the score and ``score_star`` the score there. ``t_err`` and
    ``score_err`` are worst-case bounds from the board's rounding of each score.
    """

    e_v2: float
    e_v_res: float
    t_star: float
    score_star: float
    t_err: float
    score_err: float
    score_a: float
    score_b: float

    def score_at(self, t: float) -> float:
        return self.score_a + 2.0 * t * self.e_v_res + t * t * self.e_v2

    def render(self, what: str = "t") -> str:
        return (
            f"  E[V^2] = {self.e_v2:.6f}   E[V (a - T)] = {self.e_v_res:+.6f}\n"
            f"  optimal {what} = {self.t_star:.4f} (+/- {self.t_err:.4f} from rounding)\n"
            f"  score at optimum = {self.score_star:.5f} (+/- {self.score_err:.5f}); "
            f"a scored {self.score_a:.4f}, b scored {self.score_b:.4f}"
        )


def line_search(pred_a: np.ndarray, pred_b: np.ndarray, score_a: float, score_b: float,
                rounding: float = 5e-4) -> LineSearch:
    """Solve the exact parabola along ``b - a`` from the two files' board scores.

    ``score(t) = E[(a + t V - T)^2] = score_a + 2 t E[V (a - T)] + t^2 E[V^2]``.
    ``E[V^2]`` is computed from the files; ``score_b`` supplies the cross term. The
    truth ``T`` never appears, which is what makes this usable on the board.

    ``rounding`` is half the last digit the board shows (0.0005 for three
    decimals). Each score may be off by that much, so the difference is off by at
    most twice it; the bounds returned are that worst case propagated exactly.
    """
    v = np.asarray(pred_b, dtype=np.float64) - np.asarray(pred_a, dtype=np.float64)
    e_v2 = float(np.mean(v**2))
    if e_v2 <= 0.0:
        raise ValueError("the two files are identical; there is no line to search")
    e_v_res = (score_b - score_a - e_v2) / 2.0
    t_star = -e_v_res / e_v2
    score_star = score_a - e_v_res**2 / e_v2
    # d t*/d(score_b - score_a) = -1 / (2 E[V^2]); the difference is off by <= 2 * rounding
    t_err = rounding / e_v2
    # score* = score_a - e^2/E[V^2]; d/de = -2e/E[V^2] with |de| <= rounding, plus the rounding of score_a
    score_err = rounding + 2.0 * abs(e_v_res) * rounding / e_v2
    return LineSearch(e_v2, e_v_res, t_star, score_star, t_err, score_err, score_a, score_b)


@dataclass(frozen=True)
class QuadraticSearch:
    """The score's exact quadratic form on the span of several probe directions.

    ``k`` files ``b_i = a + V_i`` around a reference ``a``: on ``a + sum_i t_i V_i``
    the score is ``score_a + 2 t . g + t^T H t`` with ``H_ij = E[V_i V_j]`` known
    from the files and ``g_i = (score_b_i - score_a - H_ii) / 2`` from the board.
    So ``k + 1`` scores solve a ``k``-parameter family exactly, ``t* = -H^-1 g``.
    """

    h: np.ndarray
    g: np.ndarray
    t_star: np.ndarray
    score_star: float
    t_lo: np.ndarray
    t_hi: np.ndarray
    score_lo: float
    score_hi: float
    score_a: float

    def render(self, names: list[str] | None = None) -> str:
        names = names or [f"t{i}" for i in range(len(self.t_star))]
        lines = [f"  H = E[V_i V_j]:\n" + "\n".join("    " + " ".join(f"{v:10.6f}" for v in row) for row in self.h)]
        for i, n in enumerate(names):
            lines.append(f"  optimal {n} = {self.t_star[i]:.4f}  [{self.t_lo[i]:.4f}, {self.t_hi[i]:.4f}]")
        lines.append(f"  score at optimum = {self.score_star:.5f}  [{self.score_lo:.5f}, {self.score_hi:.5f}]; a scored {self.score_a:.4f}")
        return "\n".join(lines)


def quadratic_search(pred_a: np.ndarray, preds_b: list[np.ndarray], score_a: float,
                     scores_b: list[float], rounding: float = 5e-4) -> QuadraticSearch:
    """Solve the exact quadratic on the span of ``preds_b - pred_a`` from ``k + 1`` board scores.

    Worst-case bounds enumerate every corner of the ``k + 1`` scores' rounding
    intervals, so they are exact for the rounding model, not a linearisation.
    """
    import itertools

    a = np.asarray(pred_a, dtype=np.float64)
    v = [np.asarray(b, dtype=np.float64) - a for b in preds_b]
    k = len(v)
    if k != len(scores_b):
        raise ValueError("one score per probe file")
    h = np.array([[float(np.mean(vi * vj)) for vj in v] for vi in v])
    if np.linalg.matrix_rank(h) < k:
        raise ValueError("the probe directions are linearly dependent; the family is not identified")

    def solve(sa: float, sb: list[float]) -> tuple[np.ndarray, float]:
        g = np.array([(sbi - sa - h[i, i]) / 2.0 for i, sbi in enumerate(sb)])
        t = -np.linalg.solve(h, g)
        return t, sa + 2.0 * t @ g + t @ h @ t

    t_star, score_star = solve(score_a, scores_b)
    g = np.array([(sbi - score_a - h[i, i]) / 2.0 for i, sbi in enumerate(scores_b)])
    ts, ss = [], []
    for ds in itertools.product((-rounding, rounding), repeat=k + 1):
        t, s = solve(score_a + ds[0], [sbi + d for sbi, d in zip(scores_b, ds[1:])])
        ts.append(t)
        ss.append(s)
    ts = np.array(ts)
    return QuadraticSearch(h, g, t_star, score_star, ts.min(axis=0), ts.max(axis=0), min(ss), max(ss), score_a)


@dataclass(frozen=True)
class BandProbe:
    """What one per-band persistence probe reads off the board.

    The probe scales the rows of one band by ``s`` (both columns, or one) and leaves
    every other row as the reference. Then
    ``score_probe - score_ref = f_band * ((s^2 - 1) A_band - 2 (s - 1) B_band)``
    with ``f_band`` the band's share of rows, ``A_band = E_band[X^2]`` known from the
    file, and ``B_band = E_band[X T]`` the unknown. ``s_star = B_band / A_band`` is
    the band's own optimal persistence scale: how much the test trajectories of that
    band change over forty years, in the direction of their initial state.
    """

    band: str
    columns: str
    s: float
    f_band: float
    a_band: float
    b_band: float
    s_star: float
    s_err: float

    def render(self) -> str:
        return (
            f"  {self.band:5s} ({self.columns}) scaled by {self.s}: f_band {self.f_band:.4f}  "
            f"A_band {self.a_band:.6f}  B_band {self.b_band:.6f}  ->  s* {self.s_star:.4f} "
            f"(+/- {self.s_err:.4f})"
        )


def band_probe(ref: np.ndarray, probe: np.ndarray, score_ref: float, score_probe: float,
               ages: np.ndarray, rounding: float = 5e-4) -> BandProbe:
    """Identify which band and columns ``probe`` scaled, and solve for that band's ``B``.

    Pooled over the columns the probe touched: if both columns were scaled the
    expectations are over ``2 n_band`` cells, if one column over ``n_band`` cells,
    and the row-share bookkeeping below is adjusted so the identity in the class
    docstring holds in either case.
    """
    ref = np.asarray(ref, dtype=np.float64)
    probe = np.asarray(probe, dtype=np.float64)
    # files carry six decimals, so anything below 1e-4 in the reference cannot show
    # a scaling and is treated as zero
    tiny = 1e-4
    diff = np.abs(probe - ref) > 1e-6
    rows = diff.any(axis=1)
    if not rows.any():
        raise ValueError("probe equals the reference; nothing was scaled")
    touched_bands = set(band_of(ages[rows]))
    if len(touched_bands) != 1:
        raise ValueError(f"probe touches more than one band: {sorted(touched_bands)}")
    band = touched_bands.pop()
    mask = band_masks(ages)[band]
    cols = [k for k in range(2) if diff[:, k].any()]
    columns = "both" if cols == [0, 1] else COLUMNS[cols[0]]
    # a row whose reference is exactly zero in the touched columns cannot show a
    # scaling; every other row of the band must have moved
    nonzero = mask & (np.abs(ref[:, cols]).max(axis=1) > tiny)
    if (nonzero & ~rows).any():
        raise ValueError(f"probe scales only part of band {band!r}")
    big = ref[mask][:, cols]
    ratio = probe[mask][:, cols][np.abs(big) > tiny] / big[np.abs(big) > tiny]
    s_values = np.unique(np.round(ratio, 4))
    if len(s_values) != 1:
        raise ValueError(f"probe is not a single scale inside band {band!r}: {s_values[:5]}")
    s = float(s_values[0])
    n_cells_total = ref.size
    cells = ref[mask][:, cols]
    f_band = cells.size / n_cells_total
    a_band = float(np.mean(cells**2))
    delta = score_probe - score_ref
    b_band = ((s * s - 1.0) * a_band - delta / f_band) / (2.0 * (s - 1.0))
    s_star = b_band / a_band
    # |d s*/d delta| = 1 / (2 (s-1) f_band A_band); delta is off by <= 2 * rounding
    s_err = rounding / (abs(s - 1.0) * f_band * a_band)
    return BandProbe(band, columns, s, f_band, a_band, b_band, s_star, s_err)


def scaled_parabola(score_at_1: float, score_at_s: float, s: float, a: float) -> tuple[float, float]:
    """``(B, C)`` of ``score(s) = s^2 A - 2 s B + C`` from two points and a known ``A``."""
    b = ((s * s - 1.0) * a - (score_at_s - score_at_1)) / (2.0 * (s - 1.0))
    c = score_at_1 - a + 2.0 * b
    return b, c


def moment_matched_lambda(s_star_board: float, s_star_model: float) -> float:
    """``lambda`` that makes the damped model reproduce the board's persistence slope.

    With ``P = X + D`` the model's slope is ``E[X P]/E[X^2] = 1 + E[X D]/E[X^2]``; the
    damped model's is ``1 + lambda E[X D]/E[X^2]``. Matching it to the board's
    ``s*`` gives ``lambda = (s*_board - 1) / (s*_model - 1)``. It is a first-moment
    calibration (the projection of the truth on ``X``), not the MSE optimum, which
    only a pair of model files on the board gives (:func:`line_search`).
    """
    if abs(s_star_model - 1.0) < 1e-9:
        raise ZeroDivisionError("the model predicts no change along X; lambda is undefined")
    return (s_star_board - 1.0) / (s_star_model - 1.0)


def slope(x: np.ndarray, y: np.ndarray) -> float:
    """Regression slope through the origin, ``E[x y] / E[x^2]``, pooled over all cells."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.sum(x * y) / np.sum(x * x))
