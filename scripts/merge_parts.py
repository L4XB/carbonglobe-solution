#!/usr/bin/env python
"""Composite of contrast files that share a base: out = base + sum_i w_i x (file_i - base), cell by cell, both columns.

The contrast files of one batch differ from the base on disjoint cells (heights of young rows, heights of mature rows, loss rows ...), so the
weights are the fitted steps of each contrast (gain of a contrast at step 1 = (2 kappa - 1) E; at step w = (2 kappa w - w^2) E, best w = kappa).

    python scripts/merge_parts.py --base results/cand48_A.csv --out cand48_H1 results/cand48_CY.csv:1.0 results/cand48_CM.csv:0.6 [results/cand48_B1.csv:1]
Writes results/<out>.csv and work/dir_<out>.npy (file - C, for joint_ridge --cand); prints the overlap of the contrasts and sanity numbers.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

FL_A = 0.401412 / 3.39567


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", nargs="+", help="file:weight")
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--centre", default="results/spatcorr_stack_20261106_s08.csv", help="C of the screens (work/dir_<out>.npy = file - C)")
    args = ap.parse_args()
    B = shift.read_predictions(ROOT / args.base).astype(np.float64); P = B.copy(); touched = np.zeros(B.shape, dtype=np.int16)
    for it in args.items:
        f, w = it.rsplit(":", 1); w = float(w)
        D = shift.read_predictions(ROOT / f).astype(np.float64) - B
        ch = np.abs(D) > 1e-9; touched += ch
        P += w * D
        print(f"   {Path(f).name:22s} x {w:4.2f}: cells changed height {int(ch[:, 0].sum())}, agb {int(ch[:, 1].sum())}; energy of the full contrast {np.sum(D ** 2):.1f}, applied {np.sum((w * D) ** 2):.1f}")
    print(f"   cells touched by more than one contrast: {int((touched > 1).sum())}")
    P[:, 0] = np.maximum(P[:, 0], 0.0); P[:, 1] = np.maximum(P[:, 1], -FL_A)
    assert np.isfinite(P).all() and P.shape == (185130, 2) and P.max() < 20
    out = ROOT / "results" / f"{args.out}.csv"
    shift.write_predictions(P, out)
    C = shift.read_predictions(ROOT / args.centre).astype(np.float64)
    np.save(ROOT / "work" / f"dir_{args.out}.npy", (P - C).astype(np.float32))
    print(f"wrote {out.relative_to(ROOT)}: energy against the base {np.sum((P - B) ** 2):.1f} squared units; column means {P[:, 0].mean():.4f} {P[:, 1].mean():.4f}; min agb {P[:, 1].min():.4f}")


if __name__ == "__main__":
    main()
