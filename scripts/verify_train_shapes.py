"""Pin down the training arrays' semantics instead of inferring them from a README.

Two claims matter for a 40-step rollout and both are measured here:

1. ``glob_Y{age}`` has 41 year slots for 40 forcing years, and slot 0 is the state
   **before** any forcing year is applied. Measured by lag correlation: GPP responds
   to temperature within the year, so the lag that maximises corr(GPP anomaly at
   slot k, temperature at forcing year j) says which slot lines up with which
   forcing year.
2. The 136 drivers of ``glob_X_fea`` and of ``test_x`` are the same 136 variables in
   the same order. Measured by comparing the per-driver mean and sd of each file
   against the shipped ``x_mean``/``x_std``, and by checking which drivers are
   static per site in both.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import io  # noqa: E402

SAMPLE = 4000  # sites drawn for the lag test
RNG = np.random.default_rng(42)


def anomaly(a: np.ndarray, axis: int) -> np.ndarray:
    return a - a.mean(axis=axis, keepdims=True)


def main() -> None:
    X = io.open_train_x()
    print(f"glob_X_fea.npy  {X.shape} {X.dtype}   {X.nbytes / 1e9:.2f} GB")
    Y1 = io.open_train_y(1)
    print(f"glob_Y001.npy   {Y1.shape} {Y1.dtype}   {Y1.nbytes / 1e9:.2f} GB")
    print(f"  forcing years {X.shape[1]}, target year slots {Y1.shape[1]}  -> one extra slot")

    sites = np.sort(RNG.choice(X.shape[0], size=SAMPLE, replace=False))

    print("\n--- claim 1: which year slot lines up with which forcing year ---")
    # Annual mean temperature (driver 0) per site per forcing year.
    temp = np.empty((SAMPLE, X.shape[1]), dtype=np.float64)
    for i, s in enumerate(sites):
        temp[i] = np.asarray(X[s], dtype=np.float64)[:, :, 0].mean(axis=1)

    for age in (10, 100):
        Y = io.open_train_y(age)
        gpp = np.empty((SAMPLE, Y.shape[1]), dtype=np.float64)
        for i, s in enumerate(sites):
            gpp[i] = np.asarray(Y[s, :, 0, 4], dtype=np.float64)

        t_anom = anomaly(temp, axis=1)
        print(f"\n  seed age {age}: corr(GPP anomaly at slot k, temperature anomaly at forcing year k+lag)")
        for lag in (-1, 0, 1):
            # slots that exist for this lag
            ks = np.arange(max(0, -lag), min(Y.shape[1], X.shape[1] - lag))
            g = anomaly(gpp[:, ks], axis=1)
            t = t_anom[:, ks + lag]
            num = (g * t).sum(axis=1)
            den = np.sqrt((g**2).sum(axis=1) * (t**2).sum(axis=1))
            r = np.where(den > 0, num / np.where(den == 0, 1, den), np.nan)
            label = {(-1): "slot k <- forcing year k-1", 0: "slot k <- forcing year k",
                     1: "slot k <- forcing year k+1"}[lag]
            print(f"    lag {lag:+d}  median r {np.nanmedian(r):+.4f}   {label}")

    print("\n  interpretation: the lag with the highest |r| names the alignment.")
    print("  lag 0 winning means slot k is the state AFTER forcing year k, so slot 0")
    print("  is the initial state only if slot 0 predates forcing year 0.")

    # Direct check: is slot 0 a pure initialisation? At seed age 1 a site where
    # nothing grew should sit at the seeding height exactly.
    print("\n--- claim 1b: what slot 0 looks like at seed age 1 ---")
    h0 = np.asarray(Y1[:, 0, 0, 0], dtype=np.float64)
    h1 = np.asarray(Y1[:, 1, 0, 0], dtype=np.float64)
    print(f"  slot 0 height: min {h0.min():.6f}  max {h0.max():.6f}  "
          f"sites exactly at the min: {int((h0 == h0.min()).sum()):,}")
    print(f"  slot 1 height: min {h1.min():.6f}  max {h1.max():.6f}  "
          f"sites exactly at the min: {int((h1 == h1.min()).sum()):,}")
    print(f"  slot1 > slot0 at {float((h1 > h0).mean()):.2%} of sites")

    print("\n--- claim 2: are the 136 drivers the same variables in the same order? ---")
    stats = io.load_stats()
    x_mean, x_std = stats["x_mean"], stats["x_std"]
    acc_sum = np.zeros(136)
    acc_sq = np.zeros(136)
    n = 0
    const_in_time = np.ones(136, dtype=bool)
    for s in sites[:800]:
        b = np.asarray(X[s], dtype=np.float64)
        f = b.reshape(-1, 136)
        acc_sum += f.sum(axis=0)
        acc_sq += (f * f).sum(axis=0)
        n += f.shape[0]
        const_in_time &= f.max(axis=0) == f.min(axis=0)
    tr_mean = acc_sum / n
    tr_std = np.sqrt(np.maximum(acc_sq / n - tr_mean**2, 0))

    z = (tr_mean - x_mean) / np.where(x_std == 0, 1, x_std)
    print(f"  train X vs shipped x_mean/x_std: median z {np.median(z):+.4f}  "
          f"max |z| {np.abs(z).max():.4f}")
    print(f"  train X drivers constant within every site: {int(const_in_time.sum())} "
          f"-> {np.flatnonzero(const_in_time).tolist()}")

    for scenario in io.SCENARIOS:
        tx, _ = io.open_test(scenario)
        te_const = np.ones(136, dtype=bool)
        acc = np.zeros(136)
        m = 0
        for s in range(0, 800):
            b = np.asarray(tx[s], dtype=np.float64).reshape(-1, 136)
            te_const &= b.max(axis=0) == b.min(axis=0)
            acc += b.sum(axis=0)
            m += b.shape[0]
        same = np.array_equal(np.flatnonzero(te_const), np.flatnonzero(const_in_time))
        print(f"  {scenario}: static-per-site driver set identical to training: {same}")
        if not same:
            print(f"    train {np.flatnonzero(const_in_time).tolist()}")
            print(f"    test  {np.flatnonzero(te_const).tolist()}")

    print("\n--- claim 3: how far outside the training climate does the test forcing sit? ---")
    # Per-driver: what quantile of the training range does the test mean sit at?
    lo = np.full(136, np.inf)
    hi = np.full(136, -np.inf)
    for s in sites[:800]:
        f = np.asarray(X[s], dtype=np.float64).reshape(-1, 136)
        lo = np.minimum(lo, f.min(axis=0))
        hi = np.maximum(hi, f.max(axis=0))
    for scenario in io.SCENARIOS:
        tx, _ = io.open_test(scenario)
        tlo = np.full(136, np.inf)
        thi = np.full(136, -np.inf)
        for s in range(0, 800):
            f = np.asarray(tx[s], dtype=np.float64).reshape(-1, 136)
            tlo = np.minimum(tlo, f.min(axis=0))
            thi = np.maximum(thi, f.max(axis=0))
        above = thi > hi
        below = tlo < lo
        print(f"  {scenario}: drivers whose test max exceeds the training max: {int(above.sum())}/136")
        print(f"  {scenario}: drivers whose test min falls below the training min: {int(below.sum())}/136")
        # The worst offenders, in training-sd units.
        gap = np.maximum((thi - hi) / np.where(x_std == 0, 1, x_std), 0)
        worst = np.argsort(-gap)[:8]
        print(f"  {scenario}: largest overshoot above the training max, in training sd:")
        for d in worst:
            print(f"    driver {d:3d}  train max {hi[d]:12.4f}  test max {thi[d]:12.4f}  "
                  f"overshoot {gap[d]:6.2f} sd")


if __name__ == "__main__":
    main()
