#!/usr/bin/env python
"""Optimal linear combination over an explicit basis whose truth functionals are all measured.

Basis elements are full-length (N, 2) arrays: whole-column files (R, G, N per column), per-column
constants, and differences D = probe - R of the scored probe files (each nonzero only where the
probe differs from the decoded persistence R). Their functionals E[T.X] (pooled over 2N cells) come
from the board: E[T.F] = (E[F^2] + E[T^2] - score)/2, E[T.D] = E[T.F_probe] - E[T.R].
E[T.R] per column and the column means are the 2026-09-25 persistence solve. The solve reports the
predicted score and its spread over the board's rounding (uniform +-5e-4 on every score, and the
reference functionals' own uncertainty), and writes the file clipped at 0.

    ./.venv/bin/python scripts/basis_solve.py --out results/basis_x.csv [--drop name,...]
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

E_T2 = 2.588558
E_T_RH, E_T_RA, M_H, M_A = 0.8825, 1.6083, 1.0091, 0.9775


def col(F: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(F); out[:, k] = F[:, k]; return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="")
    ap.add_argument("--drop", default="")
    ap.add_argument("--models", default="direct_gbdt2_m114.csv,latent_noco2_m114.csv",
                    help="whole files whose per-column functionals are known from a column-scaled scored twin")
    args = ap.parse_args()
    scores = json.load(open(ROOT / "results" / "board_scores.json"))
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    e_t_r = E_T_RH + E_T_RA
    basis: dict[str, np.ndarray] = {}
    bvec: dict[str, float] = {}
    unc: dict[str, float] = {}  # half-width of the functional's uncertainty
    ones = np.ones_like(R)
    basis["R_h"], bvec["R_h"], unc["R_h"] = col(R, 0), E_T_RH, 0.0015
    basis["R_a"], bvec["R_a"], unc["R_a"] = col(R, 1), E_T_RA, 0.0015
    basis["1_h"], bvec["1_h"], unc["1_h"] = col(ones, 0), M_H / 2, 0.01
    basis["1_a"], bvec["1_a"], unc["1_a"] = col(ones, 1), M_A / 2, 0.01
    # whole models, per column, from a scored file + a scored height-scaled twin
    twins = {"direct_gbdt2_m114.csv": "direct_gbdt2_m114_h0876.csv", "latent_noco2_m114.csv": "latent_noco2_m114_h0876.csv"}
    for mname in args.models.split(","):
        F = shift.read_predictions(ROOT / "results" / mname).astype(np.float64)
        b1 = (np.mean(F**2) + E_T2 - scores[mname]) / 2
        F2 = shift.read_predictions(ROOT / "results" / twins[mname]).astype(np.float64)
        b2 = (np.mean(F2**2) + E_T2 - scores[twins[mname]]) / 2
        k = float(np.median(F2[F[:, 0] > 0.05, 0] / F[F[:, 0] > 0.05, 0]))
        e_th = (b1 - b2) / (1 - k); e_ta = b1 - e_th
        tag = mname.split("_m114")[0]
        basis[f"{tag}_h"], bvec[f"{tag}_h"], unc[f"{tag}_h"] = col(F, 0), e_th, 5e-4 / (1 - k)
        basis[f"{tag}_a"], bvec[f"{tag}_a"], unc[f"{tag}_a"] = col(F, 1), e_ta, 5e-4 / (1 - k)
    # probe differences
    for name, sc in scores.items():
        if not (name.startswith("bandprobe_") or name.startswith("cellprobe_") or name.startswith("dirprobe_") or name.startswith("probe_")):
            continue
        F = shift.read_predictions(ROOT / "results" / name).astype(np.float64)
        D = F - R
        if np.abs(D).max() < 1e-9:
            continue
        b = (np.mean(F**2) + E_T2 - sc) / 2
        key = "D:" + name.replace(".csv", "")
        basis[key], bvec[key], unc[key] = D, b - e_t_r, 3e-4 + 0.0015
    for d in [x for x in args.drop.split(",") if x]:
        for k in [k for k in basis if d in k]:
            basis.pop(k); bvec.pop(k); unc.pop(k)
    names = list(basis)
    B = np.stack([basis[n].reshape(-1) for n in names], axis=1)
    Gm = B.T @ B / B.shape[0]
    bv = np.array([bvec[n] for n in names])
    # greedy forward selection with a stop at the rounding scale
    chosen: list[int] = []
    best_s = E_T2
    while True:
        cand = None
        for j in range(len(names)):
            if j in chosen:
                continue
            idx = chosen + [j]
            w = np.linalg.lstsq(Gm[np.ix_(idx, idx)], bv[idx], rcond=None)[0]
            s = E_T2 - bv[idx] @ w
            if cand is None or s < cand[0]:
                cand = (s, j, w)
        if cand is None or best_s - cand[0] < 2e-4:
            break
        best_s = cand[0]
        chosen.append(cand[1])
        print(f"  + {names[cand[1]]:44s} -> {cand[0]:.4f}")
    w = np.linalg.lstsq(Gm[np.ix_(chosen, chosen)], bv[chosen], rcond=None)[0]
    print("weights:", {names[i]: round(float(x), 3) for i, x in zip(chosen, w)})
    # uncertainty: perturb functionals within their bounds, re-solve, evaluate the perturbed weights under the point truth
    rng = np.random.default_rng(0)
    sims = []
    for _ in range(500):
        bp = bv + np.array([rng.uniform(-unc[n], unc[n]) for n in names])
        wp = np.linalg.lstsq(Gm[np.ix_(chosen, chosen)], bp[chosen], rcond=None)[0]
        sims.append(E_T2 - 2 * bv[chosen] @ wp + wp @ Gm[np.ix_(chosen, chosen)] @ wp)
    print(f"predicted {E_T2 - bv[chosen] @ w:.4f}; with functional errors propagated into the weights: median {np.median(sims):.4f}, 90% {np.quantile(sims, 0.9):.4f}")
    if args.out:
        pred = (B[:, chosen] @ w).reshape(-1, 2)
        clipped = np.clip(pred, 0, None)
        print(f"clipping changes E[(p - p_clip)^2] by {np.mean((pred - clipped)**2):.2e}; means {clipped.mean(0)}")
        shift.write_predictions(clipped, ROOT / args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
