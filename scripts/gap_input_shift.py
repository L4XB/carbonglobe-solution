"""Where do the model's inputs sit on the test set relative to the fit sites?

For every input the latent model consumes (136 annual-mean drivers, the 72-dim
seasonal block, the 8 static drivers, the decoded initial state per seed age) print
the standardised mean difference (test mean - fit mean) / fit sd and the ratio of
standard deviations. A pipeline artifact shows up here as an input whose test
distribution is degenerate (sd ratio 0) or displaced by a large fraction of a sd
while the physically comparable inputs are not.

    ./.venv/bin/python scripts/gap_input_shift.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, holdout, io, latent, shift  # noqa: E402

NAMES = {0: "ta_m", 1: "pr", 2: "tsl1", 3: "tsl2", 4: "tsl3", 5: "tsl4", 6: "tsl5", 7: "tsl6", 32: "dst",
         129: "k_sat", 130: "s_theta", 131: "r_theta", 132: "L", 133: "n", 134: "m", 135: "sd"}
for h in range(24):
    NAMES[8 + h] = f"co2_h{h:02d}"
    NAMES[33 + 4 * h] = f"hus_h{h:02d}"
    NAMES[34 + 4 * h] = f"ta_h{h:02d}"
    NAMES[35 + 4 * h] = f"rsds_h{h:02d}"
    NAMES[36 + 4 * h] = f"wind_h{h:02d}"
SEASON_NAMES = [f"{g}_m{m:02d}" for g in ("ta_m", "pr", "hus", "ta_h", "rsds", "wind") for m in range(12)]


def smd_table(name: str, fit: np.ndarray, test: np.ndarray, labels: list[str], top: int = 25) -> None:
    mu, sd = fit.mean(0), fit.std(0)
    sd_safe = np.where(sd == 0, 1.0, sd)
    d = (test.mean(0) - mu) / sd_safe
    r = test.std(0) / sd_safe
    order = np.argsort(-np.abs(d))
    print(f"\n== {name}: {fit.shape[1]} inputs; |SMD| quantiles {np.quantile(np.abs(d), [.5, .9, 1]).round(3)}; "
          f"sd-ratio quantiles {np.quantile(r, [0, .1, .5, .9, 1]).round(3)}")
    print(f"   inputs with |SMD| > 0.25: {(np.abs(d) > 0.25).sum()}, > 0.5: {(np.abs(d) > 0.5).sum()}; sd-ratio < 0.5: {(r < 0.5).sum()}")
    for i in order[:top]:
        print(f"   {labels[i]:>12s}  SMD {d[i]:+.3f}  sd-ratio {r[i]:.3f}  fit mean {mu[i]:.4g} sd {sd[i]:.4g} | test mean {test[:, i].mean():.4g}")


def main() -> None:
    split = holdout.make_split(seed=42)
    stats = io.load_stats()
    x_mean, x_std = stats["x_mean"].astype(np.float64), stats["x_std"].astype(np.float64)
    x_std = np.where(x_std == 0, 1.0, x_std)
    y_mean = stats["y_mean"].astype(np.float64)

    # fit-site inputs (a 1-in-4 subsample of the fit sites is plenty for moments)
    fit_sites = split.fit[::4]
    xa = cache.open_x_annual()
    ann_fit = np.asarray(xa[fit_sites], dtype=np.float64)          # (n, 40, 136)
    season_fit = latent.seasonal_block(cache.open_x_season()[fit_sites]).astype(np.float64)
    traj = cache.open_traj()
    u0_fit = np.asarray(traj[:, fit_sites, 0, :], dtype=np.float64) / y_mean   # (15, n, 7)

    floors = shift.measured_floors()
    for scen in io.SCENARIOS:
        ann_t = np.asarray(np.load(cache.CACHE_DIR / f"test_{scen}_annual.npy"), dtype=np.float64)
        sea_t = np.load(cache.CACHE_DIR / f"test_{scen}_season.npy")
        season_t = latent.seasonal_block(sea_t).astype(np.float64)
        y0 = io.open_test(scen)[1].astype(np.float64)
        u0_t = np.transpose(latent_decode(y0, floors) / y_mean, (1, 0, 2))     # (15, n, 7)

        print(f"\n######## {scen} ########")
        # annual-mean drivers: the 40-year mean per site (what the site 'looks like' to the model)
        smd_table("40-year mean of the annual drivers (per site)", ann_fit.mean(1), ann_t.mean(1),
                  [NAMES[i] for i in range(136)])
        # interannual sd per site of each driver: what the model sees as year-to-year variability
        smd_table("interannual sd of the annual drivers (per site)", ann_fit.std(1), ann_t.std(1),
                  [NAMES[i] for i in range(136)], top=15)
        smd_table("seasonal block (72)", season_fit, season_t, SEASON_NAMES, top=15)
        smd_table("static drivers", ann_fit[:, 0, list(cache.STATIC_DRIVERS)], ann_t[:, 0, list(cache.STATIC_DRIVERS)],
                  [NAMES[i] for i in cache.STATIC_DRIVERS], top=8)
        # decoded initial state, per age, the four channels the model reads
        for j, age in enumerate(io.SEED_AGES):
            if age in (1, 30, 100, 500):
                smd_table(f"decoded y0 at seed age {age} (h, agb, soil, lai) / y_mean", u0_fit[j][:, :4], u0_t[j][:, :4],
                          ["height", "agb", "soil", "lai"], top=4)
        # the z-scored annual inputs as the model receives them: how far into the tails do test cells go?
        z_t = (ann_t - x_mean) / x_std
        z_f = (ann_fit - x_mean) / x_std
        print(f"\n   z-scored annual inputs: |z| > 3 share fit {np.mean(np.abs(z_f) > 3):.5f} test {np.mean(np.abs(z_t) > 3):.5f}; "
              f"|z| > 4: fit {np.mean(np.abs(z_f) > 4):.6f} test {np.mean(np.abs(z_t) > 4):.6f}")


def latent_decode(y0: np.ndarray, floors: np.ndarray) -> np.ndarray:
    """``(6171, 15, 7)`` physical initial state the way ``predict_latent.py`` decodes it (one multiplier)."""
    return 1.14114 * (y0 - floors)


if __name__ == "__main__":
    main()
