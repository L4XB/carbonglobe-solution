#!/usr/bin/env python
"""Climate-cell probes on the old band: the model M on the rows of one climate cell (seed ages >= 150),
decoded persistence R elsewhere. Each scored file gives the cell's optimal damping of M's change,
lambda_cell = (E[T.(M-R)_c] - E[R.(M-R)_c]) / E[(M-R)_c^2], i.e. a regional correction of the old band.
Cells come from the competition climatology (site mean temperature, annual precipitation).

    ./.venv/bin/python scripts/cell_probes.py --model results/mix_g2x2_noco2.csv --tag mix
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io, shift  # noqa: E402

OLD = (150, 200, 250, 300, 350, 400, 450, 500)
YOUNG = (1, 10, 20, 30)


def climate_of_rows(keys) -> tuple[np.ndarray, np.ndarray]:
    sc = keys["scenario"].to_numpy(); site = keys["site"].to_numpy()
    T = np.zeros(len(sc)); P = np.zeros(len(sc))
    for s in io.SCENARIOS:
        sea = np.load(ROOT / "work" / "cache" / f"test_{s}_season.npy")
        m = sc == s
        T[m] = (sea[:, 0, :].mean(1) - 273.15)[site[m]]
        P[m] = sea[:, 1, :].sum(1)[site[m]]
    return T, P


def cells(T: np.ndarray, P: np.ndarray, sc: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "tropwet126": (T > 20) & (P > 1500) & (sc == "ssp126"),
        "tropwet585": (T > 20) & (P > 1500) & (sc == "ssp585"),
        "tropdry": (T > 20) & (P <= 1500),
        "warmtemp": (T > 10) & (T <= 20),
        "cool": T <= 10,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--ref", type=Path, default=ROOT / "results" / "latent_ref_y0_m114114.csv")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--band", choices=("old", "young"), default="old")
    args = ap.parse_args()
    M = shift.read_predictions(args.model).astype(np.float64)
    R = shift.read_predictions(args.ref).astype(np.float64)
    keys = shift.template_keys()
    ages = keys["age"].to_numpy(); sc = keys["scenario"].to_numpy()
    old = np.isin(ages, OLD if args.band == "old" else YOUNG)
    T, P = climate_of_rows(keys)
    for name, cm in cells(T, P, sc).items():
        m = cm & old
        pred = R.copy(); pred[m] = M[m]
        d = pred - R
        path = ROOT / "results" / f"cellprobe_{args.tag}_{args.band}_{name}.csv"
        shift.write_predictions(pred, path)
        print(f"{path.name}: rows {m.sum():6d} | E[(p-R)^2] all cells {np.mean(d**2):.4f} | mean change h {d[m,0].mean():+.3f} a {d[m,1].mean():+.3f}")


if __name__ == "__main__":
    main()
