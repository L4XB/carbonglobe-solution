#!/usr/bin/env python
"""Calibrate the board-blend solver against simulated public/private site splits.

A pseudo-truth T~ = (current best blend) + a site-correlated residual (scaled so the scored files'
simulated scores match the real ones in size) is scored on a random public half of the sites, the
scores are rounded to 3 decimals, E[T^2] is re-derived from the persistence pair exactly as for the
real board, and the solver is run on these fake scores with the Gram matrix over ALL rows (as in
reality). Each setting's realised public score is compared with its own prediction. Settings: ridge
strength and whether the weights are constrained to sum to 1 (which removes the E[T^2] error and the
leading-order public-subset Gram error).

    ./.venv/bin/python scripts/solve_calibration.py --sims 12
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


def solve(G: np.ndarray, b: np.ndarray, ridge: float, w0: np.ndarray, sum1: bool) -> np.ndarray:
    A = G + ridge * np.eye(len(b))
    rhs = b + ridge * w0
    if not sum1:
        return np.linalg.solve(A, rhs)
    Ai_r = np.linalg.solve(A, rhs)
    Ai_1 = np.linalg.solve(A, np.ones(len(b)))
    mu = (Ai_r.sum() - 1.0) / Ai_1.sum()
    return Ai_r - mu * Ai_1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sims", type=int, default=12)
    ap.add_argument("--pub", type=float, default=0.5)
    ap.add_argument("--exclude", default="persistence_s05,colprobe_latent_v1x2_level090,persistence_raw,zeros")
    ap.add_argument("--resid", type=float, default=0.07, help="target E[eps^2] of the pseudo-truth residual")
    ap.add_argument("--exact", action="store_true", help="no score rounding (isolates the Gram-mismatch part of the miss)")
    ap.add_argument("--true-gram", action="store_true", help="use the public-half Gram (isolates the rounding part of the miss)")
    ap.add_argument("--block-rho", action="store_true", help="scale the all-row Gram per band x column block by that block's exact public/all ratio of E[R^2]")
    ap.add_argument("--base", default="", help="pseudo-truth base file (default: the best scored fsolve file); it need not be in the solve")
    args = ap.parse_args()
    scores = json.load(open(ROOT / "results" / "board_scores.json"))
    names = [n for n in scores if (ROOT / "results" / n).exists() and not any(x in n for x in args.exclude.split(",") if x)]
    F = np.stack([shift.read_predictions(ROOT / "results" / n).astype(np.float64) for n in names], axis=2)  # (N, 2, K)
    N, _, K = F.shape
    keys = shift.template_keys()
    site = keys["site"].to_numpy()
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    base_name = args.base or min((n for n in scores if n.startswith("fsolve")), key=lambda n: scores[n])
    best = shift.read_predictions(ROOT / "results" / base_name).astype(np.float64)
    best_name = base_name if base_name in names else min(names, key=lambda n: scores[n])  # w0 = best file inside the solve
    print(f"pseudo-truth base: {base_name} ({scores[base_name]}); ridge target w0: {best_name} ({scores[best_name]}); {len(names)} files")
    pshift = shift.read_predictions(ROOT / "results" / "persistence_shifted.csv").astype(np.float64)
    Fl = F.reshape(N * 2, K)
    G_all = Fl.T @ Fl / Fl.shape[0]
    bm = shift.band_masks(keys["age"].to_numpy())
    blocks = [(bm[b], c) for b in ("young", "mid", "old") for c in (0, 1)]
    G_blk = [F[m][:, c, :].T @ F[m][:, c, :] / (N * 2) for m, c in blocks]
    A_all = float(np.mean(pshift ** 2))
    rng = np.random.default_rng(1)
    w0 = np.zeros(K); w0[names.index(best_name)] = 1.0
    settings = [(r, "rho") for r in (1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 3e-3)]
    rec = {st: [] for st in settings}
    real_scores = np.array([scores[n] for n in names])
    for sim in range(args.sims):
        # pseudo-truth: best blend + site-correlated residual scaled by the stand size
        u = rng.normal(size=(6171, 2))
        eps = u[site] * (0.08 + 0.22 * R) + rng.normal(size=R.shape) * 0.06 * (0.3 + R)
        eps *= np.sqrt(args.resid / np.mean(eps ** 2))
        T = np.clip(best + eps, 0, None)
        pub_sites = rng.random(6171) < args.pub
        m = pub_sites[site]
        Fp = F[m].reshape(-1, K)
        Tp = T[m].reshape(-1)
        G_pub = Fp.T @ Fp / Fp.shape[0]
        s_exact = np.array([np.mean((Fp[:, k] - Tp) ** 2) for k in range(K)])
        s = s_exact if args.exact else np.round(s_exact, 3)
        # E[T^2] from the persistence pair exactly as done for the real board (A over all rows)
        s1 = float(np.round(np.mean((pshift[m] - T[m]) ** 2), 3)); s05 = float(np.round(np.mean((0.5 * pshift[m] - T[m]) ** 2), 3))
        B = 0.75 * A_all - (s1 - s05); C = s1 - A_all + 2 * B
        b = (np.diag(G_all) + C - s) / 2
        if sim == 0:
            print(f"pseudo-truth file scores vs real (first 6): {np.round(s[:6], 3)} vs {real_scores[:6]}; C est {C:.4f} true-pub {np.mean(Tp**2):.4f}")
        # the zeros anchor: C exact on the public half; rho from the persistence pair
        C_pub_obs = float(np.round(np.mean(T[m] ** 2), 3))
        e_ps2_pub = 2 * (s1 - 2 * s05 + C_pub_obs)
        rho = e_ps2_pub / A_all
        for st in settings:
            ridge, s1c = st
            if s1c == "rho":
                if args.true_gram:
                    Gr = G_pub
                elif args.block_rho:
                    Gr = np.zeros_like(G_all)
                    for (mb, c), Gb in zip(blocks, G_blk):
                        r_pub = np.mean(R[m & mb, c] ** 2) * (m & mb).sum() / max(m.sum(), 1)
                        r_all = np.mean(R[mb, c] ** 2) * mb.sum() / len(mb)
                        Gr += (r_pub / r_all) * Gb
                else:
                    Gr = rho * G_all
                br = (np.diag(Gr) + C_pub_obs - s) / 2
                w = solve(Gr, br, ridge, w0, True)
                pred = w @ Gr @ w - 2 * w @ br + C_pub_obs
            else:
                w = solve(G_all, b, ridge, w0, s1c)
                pred = w @ G_all @ w - 2 * w @ b + C
            P = np.clip(F @ w, 0, None)
            real = float(np.mean((P[m] - T[m]) ** 2))
            priv = float(np.mean((P[~m] - T[~m]) ** 2))
            rec[st].append((pred, real, priv))
    print(f"{'ridge':>8s} {'sum1':>5s} {'realised mean':>14s} {'pred mean':>10s} {'miss mean':>10s} {'miss sd':>8s} {'private':>8s} {'priv-pub':>9s}")
    for st in settings:
        a = np.array(rec[st])
        print(f"{st[0]:8.0e} {str(st[1]):>5s} {a[:,1].mean():14.4f} {a[:,0].mean():10.4f} {np.mean(a[:,1]-a[:,0]):+10.4f} {np.std(a[:,1]-a[:,0]):8.4f} {a[:,2].mean():8.4f} {np.mean(a[:,2]-a[:,1]):+9.4f}")


if __name__ == "__main__":
    main()
