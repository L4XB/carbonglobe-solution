#!/usr/bin/env python
"""Truth estimate on a set of blocks from the probes confined to those blocks, and the blend with those blocks replaced.

A probe whose deviation from R lies inside the block set S gives an exact functional of the truth on S. Starting from
H = 0.876 * A(a + 46), T_S is estimated as H + sum_j c_j D_j (ridge) fitted to those functionals; the blend's part on S
(which full model files shape through weights chosen for other blocks) is replaced by it.

    ./.venv/bin/python scripts/block_solve.py --blocks yh,ya --tag young
    ./.venv/bin/python scripts/block_solve.py --blocks mh,oh --tag midoldh
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

M = 1.14114
SKIP = ("fsolve", "hybrid", "blend", "zeros", "persistence", "constant_one", "colprobe_latent_v1x2_level090", "pair_", "perp_", "chain_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blocks", required=True, help="comma list of yh,ya,mh,ma,oh,oa")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--blend", default="results/fsolve_rho_r2e4_20261005.csv")
    ap.add_argument("--start", default="H46", choices=["H46", "R", "blend"])
    ap.add_argument("--ridge", type=float, default=1e-4)
    args = ap.parse_args()
    sc = json.load(open(ROOT / "results" / "board_scores.json"))
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    B = shift.read_predictions(ROOT / args.blend).astype(np.float64)
    H = 0.876 * np.load(ROOT / "work" / "ageshift46_dec.npy")
    masks = shift.band_masks(shift.template_keys()["age"].to_numpy())
    S = np.zeros_like(R, dtype=bool)
    for b in args.blocks.split(","):
        S[masks[{"y": "young", "m": "mid", "o": "old"}[b[0]]], {"h": 0, "a": 1}[b[1]]] = True
    start = {"H46": H, "R": R, "blend": B}[args.start]
    C = sc["zeros.csv"]; s1, s05 = sc["persistence_shifted.csv"], sc["persistence_s05.csv"]
    ps = shift.read_predictions(ROOT / "results" / "persistence_shifted.csv").astype(np.float64)
    rho = 2 * (s1 - 2 * s05 + C) / float(np.mean(ps ** 2))
    e_ps2 = 2 * (s1 - 2 * s05 + C); s_R = M * M * e_ps2 - M * (e_ps2 + C - s1) + C
    names, Ds, ys = [], [], []
    for n, s in sc.items():
        if any(x in n for x in SKIP):
            continue
        D = shift.read_predictions(ROOT / "results" / n).astype(np.float64) - R
        if np.abs(D[~S]).max() > 1e-9 or np.abs(D).max() < 1e-9:
            continue
        X = (s - s_R - rho * float(np.mean(D * D))) / 2                       # E_pub[(R - T) D]
        names.append(n); Ds.append(D); ys.append(-X - rho * float(np.mean((start - R) * D)))  # E[(T - start) D]
    Dm = np.stack([D.ravel() for D in Ds], 1)
    G = rho * Dm.T @ Dm / Dm.shape[0]
    y = np.array(ys)
    c = np.linalg.solve(G + args.ridge * np.eye(len(names)), y)
    That = np.where(S, np.clip(start + (Dm @ c).reshape(R.shape), 0, None), 0.0)
    print(f"{len(names)} probes inside {args.blocks}:")
    for n, v in sorted(zip(names, c), key=lambda t: -abs(t[1])):
        print(f"   {v:+.3f}  {n}")
    N2 = 2 * len(R)
    print(f"in-span gain over the start: {float(c @ y):.5f}; E[(B - That)^2] on the blocks {np.sum((B - That)[S] ** 2) / N2:.5f}; "
          f"E[(B - start)^2] {np.sum((B - start)[S] ** 2) / N2:.5f}")
    out = B.copy(); out[S] = That[S]
    path = ROOT / "results" / f"blocksolve_{args.tag}_{Path(args.blend).stem}.csv"
    shift.write_predictions(out, path)
    print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
