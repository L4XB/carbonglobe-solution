#!/usr/bin/env python
"""Write a (site field) x (row weight) direction as an .npy for `dir_probe.py --no-span`.

    ./.venv/bin/python scripts/mode_direction.py --mode 2 --weight growth --block mid --out work/dir_mid_m2.npy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

BLOCKS = {"oa": [("old", 1)], "oh": [("old", 0)], "ma": [("mid", 1)], "mh": [("mid", 0)], "ya": [("young", 1)],
          "yh": [("young", 0)], "young": [("young", 0), ("young", 1)], "mid": [("mid", 0), ("mid", 1)],
          "old": [("old", 0), ("old", 1)]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", type=int, default=-1, help="single mode index; -1 = random combination of --modes")
    ap.add_argument("--modes", default="1-40")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modefile", default="work/site_graph_modes240.npy")
    ap.add_argument("--weight", default="growth", choices=["R", "growth", "BA", "BR"],
                    help="R = decoded persistence, growth = |0.876 A46 - R|, BA = |B - 0.876 A46|, BR = |B - R|")
    ap.add_argument("--base", default="results/stack_20261010.csv")
    ap.add_argument("--scenario", default="both", choices=["both", "contrast"])
    ap.add_argument("--block", required=True, choices=sorted(BLOCKS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    B = shift.read_predictions(ROOT / args.base).astype(np.float64)
    A = 0.876 * np.load(ROOT / "work" / "ageshift46_dec.npy")
    W = {"R": R, "growth": np.abs(A - R), "BA": np.abs(B - A), "BR": np.abs(B - R)}[args.weight]
    modes = np.load(ROOT / args.modefile)
    if args.mode >= 0:
        f = modes[:, args.mode].copy()
    else:
        lo, hi = (int(v) for v in args.modes.split("-"))
        f = modes[:, lo:hi + 1] @ np.random.default_rng(args.seed).normal(size=hi - lo + 1)
    f /= f.std()
    keys = shift.template_keys(); masks = shift.band_masks(keys["age"].to_numpy()); st = keys["site"].to_numpy()
    sgn = np.where(keys["scenario"].to_numpy() == "ssp585", 1.0, -1.0) if args.scenario == "contrast" else 1.0
    D = np.zeros_like(R)
    for band, col in BLOCKS[args.block]:
        m = masks[band]
        D[m, col] = (f[st] * sgn * W[:, col])[m]
    np.save(ROOT / args.out, D)
    print(f"wrote {args.out}: E[D^2] {np.mean(D ** 2):.5f}")


if __name__ == "__main__":
    main()
