#!/usr/bin/env python
"""Young-band truth from the young-only probes, and blend candidates with the young rows replaced.

Every scored probe whose deviation from R touches only seed ages 1-30 gives an exact young-band functional
E_pub[(R - T) D_j]. Starting from H = 0.876 * A(a + 46) (the chronosequence the young probes all agree with), the
young truth is estimated as H + sum_j c_j D_j with c fitted (ridge) to those functionals. Writes
results/youngH46_blend<tag>.csv (blend with young rows = H) and results/youngsolve_blend<tag>.csv (young rows = solve).

    ./.venv/bin/python scripts/young_block_solve.py --blend results/fsolve_rho_r2e4_20261005.csv --tag 20261005
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blend", default="results/fsolve_rho_r2e4_20261005.csv")
    ap.add_argument("--tag", default="20261005")
    ap.add_argument("--ridge", type=float, default=1e-4)
    args = ap.parse_args()
    sc = json.load(open(ROOT / "results" / "board_scores.json"))
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    B = shift.read_predictions(ROOT / args.blend).astype(np.float64)
    H = 0.876 * np.load(ROOT / "work" / "ageshift46_dec.npy")
    ag = shift.template_keys()["age"].to_numpy(); young = ag <= 30
    C = sc["zeros.csv"]; s1, s05 = sc["persistence_shifted.csv"], sc["persistence_s05.csv"]
    ps = shift.read_predictions(ROOT / "results" / "persistence_shifted.csv").astype(np.float64)
    rho = 2 * (s1 - 2 * s05 + C) / float(np.mean(ps ** 2))
    e_ps2 = 2 * (s1 - 2 * s05 + C); s_R = M * M * e_ps2 - M * (e_ps2 + C - s1) + C
    names, Ds, Xs = [], [], []
    for n, s in sc.items():
        if any(x in n for x in ("fsolve", "hybrid", "blend", "zeros", "persistence", "constant_one", "colprobe_latent_v1x2_level090", "pair_", "perp_")):
            continue
        F = shift.read_predictions(ROOT / "results" / n).astype(np.float64); D = F - R
        if np.abs(D[~young]).max() > 1e-9 or np.abs(D).max() < 1e-9:
            continue
        names.append(n); Ds.append(D); Xs.append((s - s_R - rho * float(np.mean(D * D))) / 2)   # E_pub[(R-T)D]
    print(f"young-only probes: {names}")
    # target: E[(T - H) D_j] = -X_j - E[(H - R) D_j]
    y = np.array([-X - rho * float(np.mean((H - R) * D)) for X, D in zip(Xs, Ds)])
    Dm = np.stack([D.ravel() for D in Ds], 1)
    G = rho * Dm.T @ Dm / Dm.shape[0]
    c = np.linalg.solve(G + args.ridge * np.eye(len(names)), y)
    delta = (Dm @ c).reshape(R.shape)
    That = np.clip(H + delta, 0, None)
    gain_vs_H = float(c @ y)  # ~ E[(T-H)^2] - E[(T - That)^2] on the public half (in-span)
    print("coefficients:", {n: round(float(v), 3) for n, v in zip(names, c)}, f"| in-span gain over H {gain_vs_H:.5f}")
    N2 = 2 * len(R)
    for nm, P in (("H46", H), ("young solve", That)):
        print(f"  blend young vs {nm}: E[(B-P)^2] pooled h {np.sum((B[young,0]-P[young,0])**2)/N2:.5f} a {np.sum((B[young,1]-P[young,1])**2)/N2:.5f}")
    for nm, P in (("youngH46", H), ("youngsolve", That)):
        out = B.copy(); out[young] = np.clip(P[young], 0, None)
        shift.write_predictions(out, ROOT / "results" / f"{nm}_blend{args.tag}.csv")
    print("wrote results/youngH46_blend*.csv, results/youngsolve_blend*.csv")


if __name__ == "__main__":
    main()
