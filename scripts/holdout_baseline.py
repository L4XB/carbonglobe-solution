"""Build the site-held-out 40-step holdout and score reference baselines on it.

The training set carries its own future trajectories -- ``glob_Y{age}`` has 41 year
slots against the forcing's 40 -- so a 40-step rollout evaluation can be carved out
of it directly. Slot 0 is the initial state and slot 40 is the year-40 outcome; that
alignment is measured in ``scripts/verify_train_shapes.py`` (the GPP-vs-temperature
lag test), not assumed here.

The split is **by site** because the competition's is: "A hidden random split of the
sites forms the Public (live) and Private (final) leaderboards". A site's thirty rows
(two scenarios x fifteen seed ages) move together, so holding out rows instead of
sites would leak a site's climate and most of its trajectory into training.

The site axis is strongly spatially ordered -- mean age-100 height runs from 2.74 in
sites 0..5000 to 25.87 in sites 40000..45000 -- so the split permutes before cutting.
A contiguous slice would be a biased holdout.

Baselines scored here, all in the competition's scaled units (height/10.90645,
agb/3.39567) so the numbers are directly comparable to the leaderboard:

``persistence``   year-40 state = year-0 state. The null that any emulator must beat.
``age_climatology`` year-40 state = the fit sites' mean year-40 state for that seed
                  age. Uses no site information at all.
``oracle_zero``   all zeros, the reference for "completely wrong".

Every interval comes from resampling **whole sites**, because sites -- not rows --
are the independent unit the leaderboard draws from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import holdout, io  # noqa: E402

YEAR0, YEAR40 = 0, 40


def load_block() -> np.ndarray:
    """Return ``(n_sites, n_ages, 2, 2)``: [..., slot, var] for slot 0 and slot 40.

    Only height and agb are pulled, and only months 0, because the month axis of
    ``glob_Y`` is degenerate -- all twelve months are bit-identical at every slot,
    which is what "targets are annual" means in the README.
    """
    ages = io.SEED_AGES
    out = np.empty((holdout.N_TRAIN_SITES, len(ages), 2, 2), dtype=np.float64)
    for j, age in enumerate(ages):
        y = io.open_train_y(age)
        out[:, j, 0, :] = np.asarray(y[:, YEAR0, 0, :2], dtype=np.float64)
        out[:, j, 1, :] = np.asarray(y[:, YEAR40, 0, :2], dtype=np.float64)
        print(f"  loaded glob_Y{age:03d}", flush=True)
    return out


def scaled(block: np.ndarray, div: np.ndarray) -> np.ndarray:
    return block / div


def report(name: str, pred: np.ndarray, true: np.ndarray, sites: np.ndarray) -> dict:
    """Score one baseline on one fold and bootstrap it over sites.

    ``pred``/``true`` are ``(n_sites, n_ages, 2)`` in scaled units. The competition
    pools both columns into one mean, so the squared errors are flattened per site.
    """
    err = (pred - true) ** 2
    per_site = err.reshape(err.shape[0], -1)
    stats = holdout.site_bootstrap(per_site, n_boot=2000, seed=7)
    mse_h = float((err[..., 0]).mean())
    mse_a = float((err[..., 1]).mean())
    print(
        f"  {name:18s} score {stats['score']:.6f}  "
        f"[{stats['lo95']:.6f}, {stats['hi95']:.6f}]  se {stats['se']:.6f}   "
        f"mse_height {mse_h:.6f}  mse_agb {mse_a:.6f}  rrmse {np.sqrt(stats['score']):.4f}"
    )
    return {"name": name, **stats, "mse_height": mse_h, "mse_agb": mse_a}


def main() -> None:
    div = np.array([io.load_stats()["y_mean"][0], io.load_stats()["y_mean"][1]])
    print(f"divisors: height {div[0]!r}  agb {div[1]!r}\n")

    print("loading year-0 and year-40 height/agb for all 15 seed ages:")
    block = load_block()
    print(f"  block {block.shape}\n")

    split = holdout.make_split(seed=42)
    print(split.describe())
    print(
        f"  fit sites {len(split.fit):,} of {holdout.N_TRAIN_SITES:,}; "
        f"holdout {len(split.holdout):,} matches the competition's {holdout.N_COMP_SITES:,} sites\n"
    )

    y0 = scaled(block[:, :, 0, :], div)
    y40 = scaled(block[:, :, 1, :], div)

    # Fit-set quantities. Nothing below may look at a holdout site.
    age_mean = y40[split.fit].mean(axis=0)  # (n_ages, 2)
    print("fit-set mean year-40 state per seed age (scaled):")
    for j, age in enumerate(io.SEED_AGES):
        print(f"    age {age:>3}  height {age_mean[j, 0]:.4f}  agb {age_mean[j, 1]:.4f}")
    print()

    for fold_name, sites in (("public", split.public), ("private", split.private)):
        true = y40[sites]
        print(f"--- {fold_name} fold, {len(sites):,} sites, {true.size // 2:,} scored rows ---")
        report("persistence", y0[sites], true, sites)
        report("age_climatology", np.broadcast_to(age_mean, true.shape), true, sites)
        report("oracle_zero", np.zeros_like(true), true, sites)
        print()

    # How much does the fold choice move a score? That is the floor on what an
    # offline comparison can resolve, and the leaderboard has the same problem.
    print("--- public vs private on the same baseline ---")
    for name, pred_fn in (
        ("persistence", lambda s: y0[s]),
        ("age_climatology", lambda s: np.broadcast_to(age_mean, y40[s].shape)),
    ):
        a = float(((pred_fn(split.public) - y40[split.public]) ** 2).mean())
        b = float(((pred_fn(split.private) - y40[split.private]) ** 2).mean())
        print(f"  {name:18s} public {a:.6f}   private {b:.6f}   gap {abs(a - b):.6f} ({abs(a-b)/a:.2%})")


if __name__ == "__main__":
    main()
