#!/usr/bin/env python
"""Rescale the height and/or agb column of a prediction file (a one-direction board probe).

    ./.venv/bin/python scripts/scale_columns.py --in results/a.csv --out results/a_h0876.csv --h 0.8763 [--a 1.0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgb import shift  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--a", type=float, default=1.0)
    args = ap.parse_args()
    pred = shift.read_predictions(args.inp).astype(np.float64)
    out = pred.copy()
    out[:, 0] *= args.h
    out[:, 1] *= args.a
    shift.write_predictions(out, args.out)
    print(f"{args.out.name}: height x{args.h}, agb x{args.a}; E[(out - in)^2] = {float(np.mean((out - pred) ** 2)):.6f}; "
          f"column means {out.mean(0)}")


if __name__ == "__main__":
    main()
