"""Persistence candidates that decode ``test_y0`` with per-variable scales.

Follows ``persistence_probe.py`` (floor settled on the board: shifted 0.252, raw
0.287) with the next unknown, the decode multiplier ``m_k`` in
``raw_k = m_k * (y0_k - floor_k)``:

* height: ``m_h = 1/0.87632 = 1.14114`` is pinned offline by two process-model
  constants that survive any affine map (``theory_y0_scale.py``): the 35.0002 m
  height cap maps to the test plateau 29.770555 (43 rows sit on it), and the 1.5 m
  seedling height maps to the test age-1 minimum 0.415183 (1/0.87742). They agree
  to 0.1%. Decoded, the test height column has mean ~0.97 of the global mean and a
  maximum of exactly 35.0002/10.90645 = 3.209 - the training set's own scaled
  maximum (``instrument.md`` Sec. 1) - which is what a column pre-divided by its own
  global mean should look like.
* agb: no hard constant. Allometry at old seed ages (train median raw agb / test
  median ``y0 - floor`` inside decoded-height bins 3-30 m) gives 1.13-1.39, weighted
  mean 1.26 - but the same deficit would appear if the scenario spin-up simply grew
  thinner stands, so the two candidates below bracket it and the board decides.

Both files decode height with 1.14114; they differ only in the agb multiplier
(1.00 vs 1.26). The expected difference between them is large (~0.1-0.2 in score,
see ``theory.md`` Sec. 5 Q1), so one slot each is decisive. Nothing here submits.

    .venv/bin/python scripts/theory_scale_probe.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("ssp126", "ssp585")
AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]

CAP_TRAIN = 35.0002  # slot-0 height cap in the training runs (53 sites on it at age 100)
A_HEIGHT = 0.87632  # y0_h = A_HEIGHT * raw_h + floor_h, from the cap anchor
M_HEIGHT = 1.0 / A_HEIGHT  # decode multiplier, 1.14114
#: 1.00 and 1.26 bracket the allometry reading; 1.14 is the "one multiplier for all
#: seven variables" reading (height's pinned 1.141, gpp ~1.15 from GPP given LAI,
#: soil ~1.12), which theory.md Sec. 5 Q1 now considers the most economical.
M_AGB_CANDIDATES = {"agb100": 1.00, "agb114": 1.14, "agb126": 1.26}


def load_y0(data: Path, scenario: str) -> np.ndarray:
    with np.load(data / f"test_{scenario}.npz") as handle:
        return np.asarray(handle["test_y0"], dtype=np.float64)


def build(frames: dict[str, np.ndarray], sites: list[int], floors: np.ndarray,
          means: np.ndarray, m_height: float, m_agb: float) -> pd.DataFrame:
    """One row per (scenario, site, age): year 40 = decoded year 0, scaled by the global mean."""
    rows = []
    for scenario, y0 in frames.items():
        height = m_height * (y0[:, :, 0] - floors[0]) / means[0]
        agb = m_agb * (y0[:, :, 1] - floors[1]) / means[1]
        ids = [f"{scenario}_{site}_{age}" for site in sites for age in AGES]
        rows.append(pd.DataFrame({"id": ids, "height": height.reshape(-1), "agb": agb.reshape(-1)}))
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    stats = np.load(args.cache / "data_stats.npz")
    means = np.asarray(stats["y_mean"], dtype=np.float64)
    assert abs(means[0] - 10.90645) < 1e-4 and abs(means[1] - 3.39567) < 1e-4, means[:2]

    frames = {s: load_y0(args.data, s) for s in SCENARIOS}
    floors = np.concatenate([f.reshape(-1, 7) for f in frames.values()]).min(axis=0)
    print("floors:", np.round(floors, 6))
    # the cap anchor, recomputed here so the constant above cannot silently go stale
    old = [AGES.index(a) for a in AGES if a >= 100]
    cap_test = max(f[:, old, 0].max() for f in frames.values())
    a_h = (cap_test - floors[0]) / CAP_TRAIN
    assert abs(a_h - A_HEIGHT) < 2e-4, (a_h, A_HEIGHT)
    print(f"height cap anchor: test plateau {cap_test:.6f} -> a_h = {a_h:.5f}, decode multiplier m_h = {1 / a_h:.5f}")

    sites = pd.read_csv(args.cache / "sites_ssp.csv").iloc[:, 0].tolist()
    template = pd.read_csv(args.cache / "sample_submission.csv")
    args.out.mkdir(parents=True, exist_ok=True)
    for tag, m_agb in M_AGB_CANDIDATES.items():
        frame = build(frames, sites, floors, means, M_HEIGHT, m_agb)
        merged = template[["id"]].merge(frame, on="id", how="left", validate="one_to_one")
        assert merged.height.notna().all() and len(merged) == len(template)
        assert merged.height.min() >= 0 and merged.agb.min() >= 0
        assert merged.height.max() < 15 and merged.agb.max() < 15
        # the decoded cap must land on the training set's own scaled maximum
        assert abs(merged.height.max() - CAP_TRAIN / means[0]) < 1e-4, merged.height.max()
        path = args.out / f"persistence_mh1141_{tag}.csv"
        merged.to_csv(path, index=False, float_format="%.6f")
        print(f"  m_h={M_HEIGHT:.5f} m_agb={m_agb:.2f}: {len(merged):,} rows  height mean {merged.height.mean():.4f} "
              f"max {merged.height.max():.4f}  agb mean {merged.agb.mean():.4f} max {merged.agb.max():.4f} -> {path}")
    print("\nNothing submitted; the lead routes these to slots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
