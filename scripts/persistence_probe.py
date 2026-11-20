#!/usr/bin/env python
"""Two persistence submissions that differ only in how ``test_y0`` is decoded.

The first thing to get right in this competition is not a model but a unit, and the
README says so in bold: "an **unscaled** submission scores as if it were completely
wrong". There is a second, subtler version of the same trap, found by measuring
``test_y0`` rather than assuming it:

```
per-variable minimum of test_y0
height -0.9009   agb -0.4014   soil -0.5324   lai +0.1179
gpp    -0.4913   npp -0.6563   rh  +0.0847

8,185 of 92,565 (site, age) rows sit at the floor for height AND agb at once, and on
exactly those rows lai, gpp, npp and rh are each a single constant
(0.117938 / 0.259026 / 0.127815 / 0.084656).
```

A forest cannot have negative height or negative biomass, and a bare site is the same
state everywhere, so those floors are **transformed zeros**: ``test_y0`` is an affine
image of the physical state, ``raw = a_k * (y0 - floor_k)``, not the raw state and not
the z-score (``data_stats.npz`` would put the zero at -1.05 for height, not -0.90).

That matters because the submission is the physical year-40 value divided by its
global mean. Decoding wrongly shifts every prediction by ``floor / mean`` -- 0.083 for
height, 0.118 for agb -- which is the same size as the whole gap between first place
and fourth.

So this script writes both readings of the same trivial model, year 40 = year 0:

    shifted   (y0 - floor) / y_mean      the affine reading, a = 1
    raw       y0 / y_mean                the naive reading

One leaderboard slot each settles which space the solution is stored in, and the
answer is worth far more than the two slots, because every later model inherits it.

    ./.venv/bin/python scripts/persistence_probe.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SCENARIOS = ("ssp126", "ssp585")
AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
TARGETS = ["height", "agb", "soil", "lai", "gpp", "npp", "rh"]


def load_scenario(data: Path, scenario: str) -> np.ndarray:
    with np.load(data / f"test_{scenario}.npz") as handle:
        return np.asarray(handle["test_y0"], dtype=np.float64)


def build(frames: dict[str, np.ndarray], sites: list[int], floors: np.ndarray,
          means: np.ndarray, shift: bool) -> pd.DataFrame:
    """One row per (scenario, site, age), year 40 predicted as year 0."""
    rows: list[pd.DataFrame] = []
    for scenario, y0 in frames.items():
        state = y0 - floors if shift else y0
        height = state[:, :, 0] / means[0]
        agb = state[:, :, 1] / means[1]
        identifiers = [
            f"{scenario}_{site}_{age}" for site in sites for age in AGES
        ]
        rows.append(
            pd.DataFrame(
                {
                    "id": identifiers,
                    "height": height.reshape(-1),
                    "agb": agb.reshape(-1),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    stats = np.load(args.cache / "data_stats.npz")
    means = np.asarray(stats["y_mean"], dtype=np.float64)
    print(f"y_mean: height {means[0]:.5f}  agb {means[1]:.5f}")
    # The README names height/10.90645 and agb/3.39567; assert rather than assume.
    assert abs(means[0] - 10.90645) < 1e-4, means[0]
    assert abs(means[1] - 3.39567) < 1e-4, means[1]

    frames = {scenario: load_scenario(args.data, scenario) for scenario in SCENARIOS}
    stacked = np.concatenate([f.reshape(-1, 7) for f in frames.values()], axis=0)
    floors = stacked.min(axis=0)
    print("floors:", dict(zip(TARGETS, np.round(floors, 6))))
    print(f"floor/mean shift: height {floors[0]/means[0]:+.4f}  agb {floors[1]/means[1]:+.4f}")

    sites = pd.read_csv(args.cache / "sites_ssp.csv")
    site_ids = sites.iloc[:, 0].tolist() if sites.shape[1] else []
    print(f"sites: {len(site_ids)} from {list(sites.columns)[:3]}")

    template = pd.read_csv(args.cache / "sample_submission.csv")
    print(f"template rows: {len(template):,}, columns {list(template.columns)}")

    args.out.mkdir(parents=True, exist_ok=True)
    for name, shift in (("shifted", True), ("raw", False)):
        frame = build(frames, site_ids, floors, means, shift)
        merged = template[["id"]].merge(frame, on="id", how="left", validate="one_to_one")
        missing = int(merged.height.isna().sum())
        if missing:
            print(f"  {name}: {missing:,} template ids unmatched -- ids do not line up, not written")
            print("  first unmatched:", merged.loc[merged.height.isna(), "id"].head(3).tolist())
            print("  built ids look like:", frame.id.head(3).tolist())
            continue
        assert len(merged) == len(template)
        path = args.out / f"persistence_{name}.csv"
        merged.to_csv(path, index=False, float_format="%.6f")
        print(
            f"  {name}: {len(merged):,} rows  height mean {merged.height.mean():.4f} "
            f"min {merged.height.min():.4f}  agb mean {merged.agb.mean():.4f} -> {path}"
        )
    print("\nA human or an explicitly authorised step submits these; this script does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
