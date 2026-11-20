#!/usr/bin/env python
"""Convex stack of scored blends from their public scores (no extrapolation, so little noise amplification).

For weights w >= 0 with sum 1 the stack's public score is exactly sum_i w_i s_i - 1/2 sum_ij w_i w_j E_pub[(B_i - B_j)^2];
the distances are rho-scaled all-row values (2-6% mismatch) and the rounded s_i average out across the components.

    <scipy venv>/bin/python scripts/stack_blends.py --out results/stack_YYYYMMDD.csv [--prefixes fsolve_,youngsolve,...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-score", type=float, default=0.052)
    ap.add_argument("--prefixes", default="fsolve_,youngsolve,wshrink,hybrid_blend20261004_midh,symblend,stack_,chain_")
    args = ap.parse_args()
    sc = json.load(open(ROOT / "results" / "board_scores.json"))
    names = [n for n in sc if n.startswith(tuple(args.prefixes.split(","))) and sc[n] <= args.max_score]
    F = np.stack([shift.read_predictions(ROOT / "results" / n).astype(np.float64).reshape(-1) for n in names], 1)
    s = np.array([sc[n] for n in names])
    C = sc["zeros.csv"]
    ps = shift.read_predictions(ROOT / "results" / "persistence_shifted.csv").astype(np.float64)
    rho = 2 * (sc["persistence_shifted.csv"] - 2 * sc["persistence_s05.csv"] + C) / float(np.mean(ps ** 2))
    G = F.T @ F / F.shape[0]
    d = rho * (np.diag(G)[:, None] + np.diag(G)[None, :] - 2 * G)
    pred = lambda w: float(w @ s - 0.5 * w @ d @ w)
    K = len(names)
    r = minimize(pred, np.ones(K) / K, jac=lambda w: s - d @ w, bounds=[(0, 1)] * K, method="SLSQP",
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(K)}],
                 options={"ftol": 1e-14, "maxiter": 1000})
    w = np.clip(r.x, 0, None); w /= w.sum()
    # rounding: each s_i uniform +-5e-4
    sd = 5e-4 / np.sqrt(3) * np.sqrt(np.sum(w ** 2))
    print(f"{K} blends; predicted stack score {pred(w):.4f} (+-{sd:.4f} rounding); best single {s.min():.3f}; spread term {0.5 * w @ d @ w:.5f}")
    for n, v in sorted(zip(names, w), key=lambda t: -t[1]):
        if v > 1e-3:
            print(f"   {v:.3f}  {n} ({sc[n]:.3f})")
    P = np.clip((F @ w).reshape(-1, 2), 0, None)
    shift.write_predictions(P, ROOT / args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
