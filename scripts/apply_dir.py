#!/usr/bin/env python
"""Add a multiple of a measured direction to a file (post step of scripts/design_run.py).

    file <- file + step x D,   D = <dir-file> - <dir-centre> (two scored submission files)   or   D = <dir-npy> (rows x 2 array)
Use: a probe whose board reading was never taken into the candidate, e.g. results/spat_tw47_cx12_cl.csv against results/spatcorr_stack_20261106_s08.csv
(exact-input ED with fire in the collapse cluster, projected off the old probe span; read 0.45 +- 0.15 on 11-10; scripts/diag/probe_residuals.py shows
what every scored file still wants of the current candidate). Heights stay within [0, 35 m], agb above the floor.

    python scripts/apply_dir.py --inp results/x.csv --out results/y.csv --dir-file results/spat_tw47_cx12_cl.csv --dir-centre results/spatcorr_stack_20261106_s08.csv --step 0.4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

HCAP = 35.0 / 1.14114 / 10.90645
FL = 0.401412 / 3.39567


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--dir-file", default=""); ap.add_argument("--dir-centre", default=""); ap.add_argument("--dir-npy", default="")
    ap.add_argument("--step", type=float, required=True)
    args = ap.parse_args()
    P = shift.read_predictions(ROOT / args.inp).astype(float)
    if args.dir_npy:
        D = np.load(ROOT / args.dir_npy).astype(float)
    else:
        D = shift.read_predictions(ROOT / args.dir_file).astype(float) - shift.read_predictions(ROOT / args.dir_centre).astype(float)
    assert D.shape == P.shape
    Q = P + args.step * D
    Q[:, 0] = np.clip(Q[:, 0], 0.0, np.maximum(HCAP, P[:, 0])); Q[:, 1] = np.maximum(Q[:, 1], -FL)
    shift.write_predictions(Q, ROOT / args.out)
    print(f"{args.inp} -> {args.out}: + {args.step:g} x {args.dir_npy or Path(args.dir_file).name}: energy of the direction {np.sum(D * D):.0f}, applied {np.sum((Q - P) ** 2):.1f} (unclipped {args.step ** 2 * np.sum(D * D):.1f}), column means {P[:, 0].mean():.4f} {P[:, 1].mean():.4f} -> {Q[:, 0].mean():.4f} {Q[:, 1].mean():.4f}")


if __name__ == "__main__":
    main()
