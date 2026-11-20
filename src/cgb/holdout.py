"""A site-held-out 40-step evaluation carved out of the present-day training set.

**Why this is possible at all.** ``glob_Y{age}.npy`` carries 41 year slots against
the forcing's 40. Slot 0 is the state before any forcing year is applied and slots
1..40 are the state after each one, so the training set already contains exactly
the object the competition asks for: an initial state, 40 years of forcing, and the
year-40 outcome. The offline task is then structurally identical to the scored one
and differs only in the climate the forcing describes. That claim is measured in
``scripts/verify_train_shapes.py``, not assumed here.

**Why the split is by site.** The README: "A hidden random split of the sites forms
the Public (live) and Private (final) leaderboards". So a site's rows -- both
scenarios, all fifteen seed ages, thirty rows in all -- land on the same side of
the wall together. A holdout that splits rows instead of sites would put a site's
age-1 row in train and its age-10 row in test, which leaks the site's climate,
its soil and most of its trajectory, and would report a score no leaderboard can
reproduce.

It also means the leaderboard's effective sample size is **sites, not rows**.
185,130 rows sound like a lot of independent evidence; 6171 sites, whose thirty
rows each are strongly correlated, are far less. Anything that compares two scores
has to bootstrap over sites, which is why :func:`site_bootstrap` exists and why
every interval this project quotes comes from it.

**Sizing.** :func:`make_split` holds out 6171 sites -- the competition's own site
count -- so the noise it measures is the noise the leaderboard has, and then halves
that into public-like and private-like folds the way the competition does. The
public/private ratio is not published; 50/50 is the assumption and it is stated
here rather than buried.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import SEED_AGES, TRAIN_Y_SHAPE

#: Sites in the released present-day grid.
N_TRAIN_SITES = TRAIN_Y_SHAPE[0]

#: Sites in the competition test set; the holdout is sized to match.
N_COMP_SITES = 6171

#: Year slot of the initial state, and of the scored outcome.
YEAR0 = 0
YEAR40 = 40


@dataclass(frozen=True)
class Split:
    """Disjoint site index sets. Nothing but ``fit`` may be trained on."""

    fit: np.ndarray
    public: np.ndarray
    private: np.ndarray
    seed: int

    @property
    def holdout(self) -> np.ndarray:
        """The public and private folds together, in ascending site order."""
        return np.sort(np.concatenate([self.public, self.private]))

    def describe(self) -> str:
        return (
            f"seed {self.seed}: fit {len(self.fit):,} sites, "
            f"holdout {len(self.holdout):,} (public {len(self.public):,}, "
            f"private {len(self.private):,}); overlap "
            f"{len(np.intersect1d(self.fit, self.holdout))}"
        )


def make_split(
    *,
    seed: int = 42,
    n_holdout: int = N_COMP_SITES,
    n_sites: int = N_TRAIN_SITES,
    public_fraction: float = 0.5,
) -> Split:
    """Draw a site-held-out split that mirrors the competition's shape."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_sites)
    holdout = np.sort(order[:n_holdout])
    fit = np.sort(order[n_holdout:])
    inner = np.random.default_rng(seed + 1).permutation(len(holdout))
    cut = int(round(len(holdout) * public_fraction))
    public = np.sort(holdout[inner[:cut]])
    private = np.sort(holdout[inner[cut:]])
    if np.intersect1d(fit, holdout).size:
        raise AssertionError("fit and holdout sites overlap")
    if np.intersect1d(public, private).size:
        raise AssertionError("public and private sites overlap")
    return Split(fit=fit, public=public, private=private, seed=seed)


def make_block_split(
    *,
    seed: int = 42,
    n_holdout: int = N_COMP_SITES,
    tile_cells: int = 6,
    public_fraction: float = 0.5,
) -> Split:
    """Hold out whole spatial tiles instead of scattered sites.

    The competition sites are new locations (``notes/board-gap.md``: none of the
    6,171 shares its soil 7-tuple with a training site, and their nearest training
    site is twice as far in soil space as training sites are from each other), so a
    scattered holdout -- every held-out site with its 0.5-degree neighbours in the
    fit set -- measures interpolation between neighbours, which the board cannot
    reward. Tiles of ``tile_cells`` x ``tile_cells`` grid cells (6 cells = 3 degrees)
    are drawn at random until at least ``n_holdout`` sites are held out; the tiles
    are then dealt alternately to the public-like and private-like folds, so both
    folds are themselves blocked. The fit set never contains a cell of a held-out
    tile.
    """
    from .mortality import site_grid

    row, col, _, _ = site_grid()
    tile = (row // tile_cells) * 100000 + (col // tile_cells)
    tiles, inverse = np.unique(tile, return_inverse=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tiles))
    counts = np.bincount(inverse, minlength=len(tiles))
    chosen, total = [], 0
    for t in order:
        if total >= n_holdout:
            break
        chosen.append(t)
        total += int(counts[t])
    chosen = np.asarray(chosen)
    public_tiles = chosen[0::2] if public_fraction == 0.5 else chosen[: int(round(len(chosen) * public_fraction))]
    private_tiles = np.setdiff1d(chosen, public_tiles)
    public = np.sort(np.flatnonzero(np.isin(inverse, public_tiles)))
    private = np.sort(np.flatnonzero(np.isin(inverse, private_tiles)))
    holdout = np.sort(np.concatenate([public, private]))
    fit = np.sort(np.setdiff1d(np.arange(len(row)), holdout))
    if np.intersect1d(fit, holdout).size or np.intersect1d(public, private).size:
        raise AssertionError("block split folds overlap")
    return Split(fit=fit, public=public, private=private, seed=seed)


def site_bootstrap(
    squared_error: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Resample **whole sites** to get an interval on a mean-squared-error score.

    ``squared_error`` is ``(n_sites, k)``: every squared residual belonging to a
    site on one row. Resampling rows instead would understate the spread by
    roughly the square root of the within-site correlation, which is exactly the
    error that makes a leaderboard move look like a real improvement.
    """
    squared_error = np.asarray(squared_error, dtype=np.float64)
    if squared_error.ndim != 2:
        raise ValueError(f"expected (n_sites, k), got {squared_error.shape}")
    n_sites = squared_error.shape[0]
    per_site = squared_error.mean(axis=1)
    rng = np.random.default_rng(seed)
    draws = per_site[rng.integers(0, n_sites, size=(n_boot, n_sites))].mean(axis=1)
    return {
        "score": float(per_site.mean()),
        "se": float(draws.std(ddof=1)),
        "lo95": float(np.percentile(draws, 2.5)),
        "hi95": float(np.percentile(draws, 97.5)),
        "n_sites": n_sites,
    }


def scored_block(y_by_age: np.ndarray, year: int) -> np.ndarray:
    """Pull the ``(n_sites, n_ages, 2)`` height/agb block for one year slot.

    ``y_by_age`` is ``(n_ages, n_sites, 41, 12, 7)`` or a list-like of per-age
    arrays already reduced to ``(n_sites, 41, 7)``.
    """
    arr = np.asarray(y_by_age)
    if arr.ndim == 5:
        arr = arr[:, :, year, 0, :2]
    elif arr.ndim == 4:
        arr = arr[:, :, year, :2]
    else:
        raise ValueError(f"cannot read a scored block out of shape {arr.shape}")
    return np.transpose(arr, (1, 0, 2))


def age_index(age: int) -> int:
    """Position of ``age`` on the seed-age axis."""
    return SEED_AGES.index(age)
