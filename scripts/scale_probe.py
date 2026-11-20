#!/usr/bin/env python
"""Scale probes: the 10-10 stack with one column multiplied by (1 + eps).

Companion of `scripts/level_probe.py`. With E_pub[T] pinned by constant_one/zeros and the split-conditioned public moments
of the current stack (2026-11-04), the best affine fix on vegetated cells is T ~ C - 0.08 v + 0.032 C (gain 0.0021 +-
0.0007 vs 0.0010 for the level alone): the stack's level and scale were never fitted (the solvers skip constant_one).
A column scale has a nearly split-free public energy (conditioned ratio in work/split_ratio_static.json).

    ./.venv/bin/python scripts/scale_probe.py --col a --eps 0.2
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
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--base", default="results/stack_20261010.csv")
    args = ap.parse_args()
    B = shift.read_predictions(ROOT / args.base).astype(np.float64)
    col = 0 if args.col == "h" else 1
    D = np.zeros_like(B)
    D[:, col] = args.eps * B[:, col]
    out = ROOT / "results" / f"spat_dir_scale_{args.col}.csv"
    shift.write_predictions(B + D, out)
    cfile = ROOT / "work" / "spat_centers.json"
    centers = json.loads(cfile.read_text()); centers[out.name] = args.base
    cfile.write_text(json.dumps(centers, indent=1))
    np.save(ROOT / "work" / f"dir_scale_{args.col}.npy", D)
    print(f"{out.name}: x{1 + args.eps} on column {args.col}; E[D^2] {np.mean(D ** 2):.5f}")


if __name__ == "__main__":
    main()
