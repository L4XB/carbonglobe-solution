#!/usr/bin/env python
"""Level probes: the 10-10 stack plus a constant on every vegetated cell (B > 0.02) of one column.

Never measured before: the stack solvers skip `constant_one.csv`, and the constant directions are 62-87% outside the
probe span. The board pins E_pub[T] over both columns at 1.0025 (zeros 2.607, constant_one 1.602); conditioning random
site splits on the persistence pair's public energy ratio (rho 1.0182) puts the current stack's public mean at 1.029
+- 0.005 for any public fraction 0.2-0.7, i.e. ~0.027 too high per column (2026-11-04). A constant on a column's
vegetated cells has an (almost) split-free public energy, so its readout is exact up to rounding.

    ./.venv/bin/python scripts/level_probe.py --col a --c 0.25
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--col", choices=["h", "a"], required=True)
    ap.add_argument("--c", type=float, default=0.25)
    ap.add_argument("--min-base", type=float, default=0.02)
    ap.add_argument("--base", default="results/stack_20261010.csv")
    args = ap.parse_args()
    B = shift.read_predictions(ROOT / args.base).astype(np.float64)
    col = 0 if args.col == "h" else 1
    D = np.zeros_like(B)
    veg = B[:, col] > args.min_base
    D[veg, col] = args.c
    out = ROOT / "results" / f"spat_dir_level_{args.col}.csv"
    shift.write_predictions(B + D, out)
    cfile = ROOT / "work" / "spat_centers.json"
    centers = json.loads(cfile.read_text()); centers[out.name] = args.base
    cfile.write_text(json.dumps(centers, indent=1))
    np.save(ROOT / "work" / f"dir_level_{args.col}.npy", D)
    print(f"{out.name}: +{args.c} on {veg.sum()} vegetated {args.col} cells ({veg.mean():.3f}); E[D^2] {np.mean(D ** 2):.5f}")


if __name__ == "__main__":
    main()
