#!/usr/bin/env python
"""Do sparse (near-bare) warm training sites regrow tall canopies within the 40-year window? Organisers' truth only.

Random sample of training sites (seed 42): first-decade mean air temperature (channel 0, K), the organisers' y0 and
year-40 December state per seed age. Prints the height and agb change by temperature class x y0-agb class x band.
    python scripts/ed/train_nearbare_check.py --n 6000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io  # noqa: E402

SEED_AGES = (1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--out", default=str(ROOT / "work" / "edv3" / "train_nearbare_sample.csv"))
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    x = io.open_train_x()
    sites = np.sort(rng.choice(x.shape[0], size=args.n, replace=False))
    T = np.array([float(np.asarray(x[s, :10, :, 0], dtype=np.float64).mean()) - 273.15 for s in sites])
    rows = []
    for a in SEED_AGES:
        y = io.open_train_y(a)
        y0 = np.asarray(y[sites, 0, 11, :2], dtype=np.float64)       # (n, 2) height m, agb kgC
        y40 = np.asarray(y[sites, 40, 11, :2], dtype=np.float64)
        for i, s in enumerate(sites):
            rows.append((int(s), a, T[i], y0[i, 0], y0[i, 1], y40[i, 0], y40[i, 1]))
    d = pd.DataFrame(rows, columns=["site", "age", "T", "y0_h", "y0_a", "y40_h", "y40_a"])
    d.to_csv(args.out, index=False)
    d["band"] = np.where(d.age <= 30, "young", np.where(d.age <= 150, "mid", "old"))
    d["tcls"] = pd.cut(d["T"], [-60, 8, 18, 50], labels=["cold<8", "warm8-18", "hot>18"])
    d["acls"] = pd.cut(d.y0_a, [-1, 0.05, 0.5, 2.0, 1e9], labels=["a<0.05", "0.05-0.5", "0.5-2", ">2"])
    d["dh"] = d.y40_h - d.y0_h; d["da"] = d.y40_a - d.y0_a
    g = d.groupby(["tcls", "band", "acls"], observed=True)
    t = g.agg(n=("dh", "size"), y0_h=("y0_h", "median"), dh_med=("dh", "median"), dh_p90=("dh", lambda v: np.quantile(v, 0.9)),
              frac_dh_gt5=("dh", lambda v: float(np.mean(v > 5.0))), y0_a=("y0_a", "median"), da_med=("da", "median"),
              da_p90=("da", lambda v: np.quantile(v, 0.9)))
    pd.set_option("display.width", 200)
    print(f"{len(sites)} training sites; first-decade T classes: {pd.Series(pd.cut(T, [-60, 8, 18, 50])).value_counts().to_dict()}")
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
