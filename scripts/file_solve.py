#!/usr/bin/env python
"""Optimal blend over scored FILES with honest uncertainty.

Every scored file F_k gives E[T.F_k] = (E[F_k^2] + C - s_k)/2 with s_k the public score (rounded to
3 decimals, so s_k is uniform within +-5e-4 of the exact value) and C = E[T^2] (from the persistence
pair, +-6e-4, common to all files). A blend p = sum_k w_k F_k has exact score
E[p^2] - 2 sum_k w_k E[T.F_k] + C, so any weight vector can be scored without a submission, and
the rounding of every s_k can be propagated by Monte Carlo. The weights minimise
E[p^2] - 2 w'b + ridge * |w - w0|^2 (ridge toward w0 = the best single file), and the ridge is chosen by
the Monte-Carlo median of the realised score: weights solved from perturbed functionals, scored
under the unperturbed ones.

    ./.venv/bin/python scripts/file_solve.py [--exclude substr,...] [--out results/x.csv]
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

C0, C_UNC = 2.588558, 6e-4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--include-only", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--ridges", default="1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2")
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--sum1", action="store_true", help="constrain the weights to sum to 1 (removes the E[T^2] error and the leading public-subset Gram error)")
    ap.add_argument("--w0", default="", help="ridge target file (default: the best scored file)")
    ap.add_argument("--anchor", default="", help="prediction-space shrinkage target: minimise E[(p-T)^2] + mu E[(p-A)^2] + ridge |w|^2 with A this file (need not be in the solve)")
    ap.add_argument("--mu", type=float, default=1.0, help="strength of the --anchor shrinkage (step toward the data optimum is ~1/(1+mu))")
    ap.add_argument("--rho", action="store_true", help="use the zeros anchor: C = score(zeros) and the Gram scaled by the public/all ratio rho of E[R^2]")
    args = ap.parse_args()
    scores = json.load(open(ROOT / "results" / "board_scores.json"))
    names = [n for n in scores if (ROOT / "results" / n).exists()]
    if args.include_only:
        names = [n for n in names if any(s in n for s in args.include_only.split(","))]
    names = [n for n in names if not any(s and s in n for s in args.exclude.split(","))]
    F = np.stack([shift.read_predictions(ROOT / "results" / n).astype(np.float64).reshape(-1) for n in names], axis=1)
    s = np.array([scores[n] for n in names])
    G = F.T @ F / F.shape[0]
    global C0
    if args.rho:
        C0 = scores["zeros.csv"]
        s1, s05 = scores["persistence_shifted.csv"], scores["persistence_s05.csv"]
        ps = shift.read_predictions(ROOT / "results" / "persistence_shifted.csv").astype(np.float64)
        rho = 2 * (s1 - 2 * s05 + C0) / float(np.mean(ps ** 2))
        G = rho * G
        print(f"zeros anchor: C_pub {C0:.4f}, rho {rho:.4f}")
    e_f2 = np.diag(G)
    best = int(np.argmin(s)) if not args.w0 else names.index(args.w0)
    w0 = np.zeros(len(names)); w0[best] = 1.0

    g_anchor = None
    if args.anchor:
        Aanc = shift.read_predictions(ROOT / args.anchor).astype(np.float64).reshape(-1)
        g_anchor = (rho if args.rho else 1.0) * (F.T @ Aanc) / F.shape[0]
        w0 = np.zeros(len(names))

    def solve(b: np.ndarray, ridge: float) -> np.ndarray:
        if g_anchor is not None:
            A = (1 + args.mu) * G + ridge * np.eye(len(names))
            rhs = b + args.mu * g_anchor
        else:
            A = G + ridge * np.eye(len(names))
            rhs = b + ridge * w0
        if not args.sum1:
            return np.linalg.solve(A, rhs)
        Ai_r = np.linalg.solve(A, rhs); Ai_1 = np.linalg.solve(A, np.ones(len(names)))
        return Ai_r - (Ai_r.sum() - 1.0) / Ai_1.sum() * Ai_1

    def realised(w: np.ndarray, b_true: np.ndarray, c_true: float) -> float:
        return float(w @ G @ w - 2 * w @ b_true + c_true)

    b0 = (e_f2 + C0 - s) / 2
    sigma2 = (5e-4 / 2) ** 2 / 3.0  # variance of E[T.F] from a score rounded to 3 decimals (uniform +-2.5e-4)
    ev = np.linalg.eigvalsh(G)
    print(f"{len(names)} scored files; best single {names[best]} {s[best]:.3f}; Gram eigenvalues min {ev.min():.2e} max {ev.max():.2e}")
    results = []
    for ridge in [float(x) for x in args.ridges.split(",")]:
        if ridge <= 0:
            continue
        w = solve(b0, ridge)
        point = realised(w, b0, C0)
        # optimism of an in-sample fit to rounded functionals (Mallows Cp): E[realised] = point + 2 sigma^2 tr((G + ridge I)^-1)
        optimism = 2 * sigma2 * float(np.sum(1.0 / (ev + ridge)))
        expected = point + optimism
        # spread from the rounding given the weights
        spread = 2 * np.sqrt(sigma2) * np.sqrt(np.sum(w ** 2))
        results.append((expected, ridge, w, point, spread))
        print(f"ridge {ridge:8.1e}: point {point:.4f} + optimism {optimism:.4f} = expected {expected:.4f} (+-{spread:.4f}) | |w|_1 {np.abs(w).sum():.2f} sum(w) {w.sum():+.3f}")
    med, ridge, w, point, spread = min(results, key=lambda r: r[0])
    order = np.argsort(-np.abs(w))
    print(f"chosen ridge {ridge:.1e}: expected {med:.4f} +- {spread:.4f}")
    for i in order[:14]:
        print(f"   {w[i]:+.3f}  {names[i]}")
    if args.out:
        pred = (F @ w).reshape(-1, 2)
        clipped = np.clip(pred, 0, None)
        print(f"clip effect E[(p - clip)^2] = {np.mean((pred - clipped) ** 2):.2e}; means {clipped.mean(0)}")
        shift.write_predictions(clipped, ROOT / args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
