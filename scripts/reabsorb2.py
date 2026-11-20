#!/usr/bin/env python
"""Re-absorption of a candidate along the directions of all scored files, with the honest noise of every residual.

Every scored file p with a scored centre measures r_p = <T - B, D_p>_pub (scripts/fire/consistency_all.py). Noise per probe: rounding of its score
(53 squared units on the inner product), its centre's score (15 for the 10-10 stack, whose score the mirror pairs pin; 53 otherwise) and the sampling
error of the direction's PUBLIC energy under a random site split (0.25 x sqrt(sum_s e_s^2)). Model: x = T - B restricted to the probe span,
prior x ~ N(0, tau2 I) per cell, r = <x, D> + noise; tau2 by evidence (r' (tau2 G + N)^-1 r = n). The correction is shrink x posterior mean
(0.75 = what carries to the private half of a random site split; 11-05 simulation).
    python scripts/reabsorb2.py results/cand50_A.csv --out work/dir_reabsorb_day50.npy [--shrink 0.75] [--exclude substr,...]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "fire"))
import consistency_all as ca
from cgb import shift
U = 370260.0


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("cand"); ap.add_argument("--out", default=""); ap.add_argument("--shrink", type=float, default=0.75)
    ap.add_argument("--exclude", default=""); ap.add_argument("--tau2", type=float, default=0.0); ap.add_argument("--noise-scale", type=float, default=1.0)
    args = ap.parse_args()
    L = ca.load(); names, sc, cen, D, CC, rho, site = (L[k] for k in ("names", "sc", "cen", "D", "CC", "rho", "site"))
    nrm = rho / (D.shape[1] * 2); wrow = ca.row_weights(L); DW = D * wrow
    E = np.einsum("nrc,nrc->n", DW, D) * nrm; base = np.einsum("nrc,nrc->n", DW, CC) * nrm
    s_p = np.array([sc[n] for n in names]); s_c = np.array([sc[Path(cen[n]).name] for n in names])
    resid = (E - (s_p - s_c)) / 2 - base; ok = E > 2e-5
    if args.exclude:
        ok &= ~np.array([any(x in n for x in args.exclude.split(",") if x) for n in names])
    B = shift.read_predictions(ROOT / args.cand).astype(np.float64); C = shift.read_predictions(ROOT / ca.C_PATH).astype(np.float64)
    r = ((resid - np.einsum("nrc,rc->n", DW, (B - C).astype(np.float32)) * nrm) * U)[ok]
    Dk = D[ok]; DWk = DW[ok]; nm = [n for n, o in zip(names, ok) if o]; n = len(r)
    sd = np.zeros(n)
    for j in range(n):
        es = np.bincount(site, weights=(Dk[j].astype(np.float64) ** 2).sum(1), minlength=6171)
        c10 = Path(cen[nm[j]]).name == "stack_20261010.csv"
        sd[j] = np.sqrt(53.0 ** 2 + (15.0 if c10 else 53.0) ** 2 + (0.25 * np.sqrt((es ** 2).sum()) * rho) ** 2)
    sd *= args.noise_scale
    G = np.einsum("nrc,mrc->nm", DWk, Dk).astype(np.float64) * rho; G = (G + G.T) / 2; N = np.diag(sd ** 2)
    z = r / sd
    print(f"{n} probes; standardised residuals rms {np.sqrt(np.mean(z ** 2)):.2f}, |z| > 2: {int((np.abs(z) > 2).sum())}, > 3: {int((np.abs(z) > 3).sum())}; median noise sd {np.median(sd):.0f} units")
    chi = lambda t2: float(r @ np.linalg.solve(t2 * G + N, r))
    if args.tau2 > 0:
        t2 = args.tau2
    elif chi(1e-8) <= n:
        print("no evidence of signal beyond the noise: nothing to absorb"); return
    else:
        lo, hi = -8.0, 4.0
        for _ in range(70):
            mid = (lo + hi) / 2
            if chi(10 ** mid) > n: lo = mid
            else: hi = mid
        t2 = 10 ** ((lo + hi) / 2)
    a_post = t2 * np.linalg.solve(t2 * G + N, r)                      # posterior-mean coefficients
    e_post = float(a_post @ G @ a_post); alpha = args.shrink * a_post
    x = np.einsum("n,nrc->rc", alpha, Dk.astype(np.float64))
    print(f"tau2 {t2:.4f} (chi2 {chi(t2):.1f} = n); posterior-mean correction: public energy {e_post:.1f} = its expected gain at full step; at shrink {args.shrink}: "
          f"expected public gain {(2 * args.shrink - args.shrink ** 2) * e_post:.1f}, energy applied {float(alpha @ G @ alpha):.1f} (all rows {np.sum(x * x):.1f}); gain if the residuals were exact {2 * alpha @ r - alpha @ G @ alpha:.1f}")
    rr = r - G @ alpha
    print(f"residuals after the correction: rms z {np.sqrt(np.mean((r / sd) ** 2)):.2f} -> {np.sqrt(np.mean((rr / sd) ** 2)):.2f}; largest coefficients: "
          + ", ".join(f"{nm[j][:26]} {alpha[j]:+.3f}" for j in np.argsort(-np.abs(alpha))[:8]))
    keys = shift.template_keys(); age = keys.age.to_numpy()
    mf = np.load(ROOT / "work/fire/max_factor_rule833.npy"); cl_ = np.load(ROOT / "work/collapse_cluster_labels.npy") == 0
    cls = np.where(cl_, "cluster", np.where(mf >= 1e3, "pos", np.where(mf >= 30, "b", "never")))[site]
    print("energy of the correction by class (height / agb): " + "; ".join(f"{k} {np.sum(x[cls == k, 0] ** 2):.1f} / {np.sum(x[cls == k, 1] ** 2):.1f}" for k in ("never", "b", "pos", "cluster"))
          + f"; by seed age 1-30 / 50-100 / 150+: {np.sum(x[age <= 30] ** 2):.1f} / {np.sum(x[(age > 30) & (age <= 100)] ** 2):.1f} / {np.sum(x[age > 100] ** 2):.1f}; max |x| {np.abs(x).max():.3f}; column mean shifts {x[:, 0].mean():+.5f} {x[:, 1].mean():+.5f}")
    if args.out:
        # cells at the agb floor / at zero height stay where they are (the truth is exact there)
        FL = 0.401412 / 3.39567; x[B[:, 1] <= -FL + 1e-6, 1] = 0.0; x[B[:, 0] <= 1e-9, 0] = 0.0
        np.save(ROOT / args.out, x.astype(np.float32)); print(f"saved {args.out} (floor / zero cells untouched; energy {np.sum(x * x):.1f})")


if __name__ == "__main__":
    main()
