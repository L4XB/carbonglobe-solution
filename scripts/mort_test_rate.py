#!/usr/bin/env python
"""How often does the latent model predict a collapse on the scored runs, against the holdout?

The lead's caution: an event model fitted on training frequencies may fire too
often -- or too rarely -- on the scenario runs, and the board says the test
trajectories change less. This script reads the sibling model's holdout predictions
(``work/models/latent_v1/pred_{public,private}.npy``) and its competition file
(``results/latent_v1_m114.csv``) and compares, on the old seed ages (>= 150) and
vegetated rows, the distribution of the predicted 40-year agb change against the
holdout truth. A "predicted collapse" is a predicted year-40 agb more than 1 kgC m-2
below year 0 (0.2945 in scaled units), the same size as an event.

``test_y0`` is decoded exactly as ``scripts/predict_latent.py`` decodes it: floor
per variable from the minimum over both scenarios, one multiplier 1.14114.

    ./.venv/bin/python scripts/mort_test_rate.py [--csv results/latent_v1_m114.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, holdout, io, submission  # noqa: E402
from cgb import mortality as mort  # noqa: E402

M_ALL = 1.14114
DROP_1KG = 1.0 / 3.395674394279066


def describe(name: str, dy: np.ndarray, agb0: np.ndarray, div: float) -> None:
    """``dy`` scaled change year 40 - year 0, ``agb0`` scaled year-0 agb, vegetated old rows."""
    rel = dy / agb0
    print(f"  {name:28s} rows {dy.size:7d}  mean change {dy.mean():+.4f} ({rel.mean():+.3f} rel)  "
          f"P(loss > 1 kgC) {(dy < -DROP_1KG).mean():.4f}  P(loss > 3 kgC) {(dy < -3 * DROP_1KG).mean():.4f}  "
          f"P(rel loss > 15%) {(rel < -0.15).mean():.4f}  mean sq change {np.mean(dy ** 2):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=io.ROOT / "results" / "latent_v1_m114.csv")
    args = ap.parse_args()
    div = cache.scoring_divisors()
    stats = io.load_stats()
    y_mean = stats["y_mean"].astype(np.float64)
    old = [j for j, a in enumerate(io.SEED_AGES) if a >= 150]
    traj = cache.open_traj()

    print("== old seed ages (>= 150), vegetated rows (agb0 > 0.5 kgC), scaled units ==")
    split = holdout.make_split(seed=42)
    for fold, sites in (("public", split.public), ("private", split.private)):
        pred = np.load(io.ROOT / "work" / "models" / "latent_v1" / f"pred_{fold}.npy")[:, :, -1, 1].astype(np.float64)  # (15, n)
        y0 = np.asarray(traj[:, sites, 0, 1], dtype=np.float64) / div[1]
        y40 = np.asarray(traj[:, sites, 40, 1], dtype=np.float64) / div[1]
        veg = y0[old] > mort.BARE_AGB / div[1]
        describe(f"holdout {fold} truth", (y40[old] - y0[old])[veg], y0[old][veg], div[1])
        describe(f"holdout {fold} latent pred", (pred[old] - y0[old])[veg], y0[old][veg], div[1])

    frame = pd.read_csv(args.csv, dtype={"id": str}, keep_default_na=False)
    keys = submission.parse_ids(frame["id"])
    frames = {s: io.open_test(s)[1].astype(np.float64) for s in io.SCENARIOS}
    floors = np.concatenate([f.reshape(-1, 7) for f in frames.values()]).min(axis=0)
    print(f"\n  test_y0 floors {np.round(floors, 6)}, multiplier {M_ALL} on all seven (as predict_latent.py)")
    for sc in io.SCENARIOS:
        raw0 = M_ALL * (frames[sc] - floors)  # (6171, 15, 7) physical
        y0 = raw0[:, :, 1] / y_mean[1]  # (6171, 15) scaled agb
        sel = keys["scenario"].to_numpy() == sc
        sub = frame.loc[sel]
        k = keys.loc[sel]
        pred = np.full((6171, 15), np.nan)
        pred[k["site"].to_numpy(), [io.SEED_AGES.index(a) for a in k["age"].to_numpy()]] = sub["agb"].to_numpy(dtype=np.float64)
        if np.isnan(pred).any():
            raise AssertionError("submission does not cover every (site, age)")
        y0o = y0[:, old].T  # (8, 6171)
        po = pred[:, old].T
        veg = y0o > mort.BARE_AGB / div[1]
        describe(f"test {sc} latent pred", (po - y0o)[veg], y0o[veg], div[1])
        # by initial biomass, the axis the loss lives on
        for lo, hi in ((0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 40.0)):
            m = veg & (y0o * div[1] >= lo) & (y0o * div[1] < hi)
            if m.sum():
                dy = (po - y0o)[m]
                print(f"      agb0 in [{lo:4.1f},{hi:4.1f}): rows {m.sum():6d} mean change {dy.mean():+.4f}  P(loss > 1 kgC) {(dy < -DROP_1KG).mean():.4f}")
    # the same bins on the holdout truth and prediction, for the comparison
    print("\n  holdout (both folds) by initial biomass:")
    sites = split.holdout
    pub_mask = np.isin(sites, split.public)
    pred = np.concatenate(
        [np.load(io.ROOT / "work" / "models" / "latent_v1" / f"pred_{f}.npy")[:, :, -1, 1] for f in ("public", "private")], axis=1
    ).astype(np.float64)
    order = np.argsort(np.concatenate([split.public, split.private]))
    pred = pred[:, order]
    y0 = np.asarray(traj[:, sites, 0, 1], dtype=np.float64) / div[1]
    y40 = np.asarray(traj[:, sites, 40, 1], dtype=np.float64) / div[1]
    y0o, y40o, po = y0[old], y40[old], pred[old]
    veg = y0o > mort.BARE_AGB / div[1]
    for lo, hi in ((0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 40.0)):
        m = veg & (y0o * div[1] >= lo) & (y0o * div[1] < hi)
        dt = (y40o - y0o)[m]
        dp = (po - y0o)[m]
        print(f"      agb0 in [{lo:4.1f},{hi:4.1f}): rows {m.sum():6d} truth mean {dt.mean():+.4f} P(loss>1kgC) {(dt < -DROP_1KG).mean():.4f} | "
              f"latent mean {dp.mean():+.4f} P(loss>1kgC) {(dp < -DROP_1KG).mean():.4f}")
    # initial biomass distribution itself, since the rate depends on it
    print("\n  initial agb (kgC m-2) of vegetated old rows: holdout median {:.2f}, q90 {:.2f}".format(
        np.median(y0o[veg] * div[1]), np.quantile(y0o[veg] * div[1], 0.9)))
    for sc in io.SCENARIOS:
        raw0 = M_ALL * (frames[sc] - floors)
        a = raw0[:, old, 1].T
        v = a > mort.BARE_AGB
        print(f"    test {sc}: median {np.median(a[v]):.2f}, q90 {np.quantile(a[v], 0.9):.2f}  (decoded with x{M_ALL})")


if __name__ == "__main__":
    main()
