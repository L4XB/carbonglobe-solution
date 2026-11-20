#!/usr/bin/env python
"""Joint screen with priors: truth - C = sum_i kappa_i h_i, kappa_i ~ N(mu_i, tau_i^2), fitted to the residuals of all scored
board files (see consistency_all.py / joint_screen.py).
    joint_ridge.py name=path[:mu[:tau]] ...        (defaults mu 1, tau 0.5; tau 0 pins kappa at mu)
    joint_ridge.py --prefix <p> [--mu 0.8] [--tau 0.5] [name:mu:tau ...]    all work/dirx_<p>_<part>.npy (the builder's --dump-parts); '_base' is pinned at 1
Prints the posterior kappas (and the unregularised joint fit for comparison).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "fire"))
import consistency_all as ca  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="*")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--mu", type=float, default=0.8)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--min-energy", type=float, default=5.0, help="parts below this energy (squared units) are pinned at the prior")
    ap.add_argument("--exclude", default="", help="comma list of substrings: scored files whose name contains one are left out of the fit (leave-one-out checks)")
    ap.add_argument("--predict", default="", help="substring of a scored file: print its residual <T - C, D> and the posterior prediction with the contributions per direction (combine with --exclude for a leave-one-out prediction)")
    ap.add_argument("--cand", default="", help="comma list name=work/dir_<file>.npy (file - C): predicted public score of each candidate under the posterior truth C + sum kappa h (score of C 0.0438)")
    ap.add_argument("--top", type=int, default=0, help="print the scored files with the largest weight on this many of the first listed directions")
    args = ap.parse_args()
    spec: dict[str, tuple[str, float, float]] = {}
    over = {}
    for it in args.items:
        if "=" in it:
            nm, rest = it.split("=", 1); f = rest.split(":")
            spec[nm] = (f[0], float(f[1]) if len(f) > 1 else 1.0, float(f[2]) if len(f) > 2 else 0.5)
        else:
            f = it.split(":"); over[f[0]] = (float(f[1]), float(f[2]) if len(f) > 2 else args.tau)
    if args.prefix:
        for p in sorted(glob.glob(str(ROOT / f"work/dirx_{args.prefix}_*.npy"))):
            nm = Path(p).stem[len(f"dirx_{args.prefix}_"):]
            mu, tau = over.get(nm, (args.mu, args.tau))
            if nm == "_base": mu, tau = 1.0, 0.0
            spec[nm] = (p, mu, tau)
    L = ca.load(); names, sc, cen, D, CC, rho = (L[k] for k in ("names", "sc", "cen", "D", "CC", "rho"))
    nrm = rho / (D.shape[1] * 2)
    wrow = ca.row_weights(L); DW = D * wrow
    E = np.einsum("nrc,nrc->n", DW, D) * nrm; base = np.einsum("nrc,nrc->n", DW, CC) * nrm
    s_p = np.array([sc[n] for n in names]); s_c = np.array([sc[Path(cen[n]).name] for n in names])
    resid = (E - (s_p - s_c)) / 2 - base; sd = 0.00041; ok = E > 2e-5
    if args.exclude:
        ex = [x for x in args.exclude.split(",") if x]
        drop = np.array([any(x in n for x in ex) for n in names]); ok &= ~drop
        print(f"left out: {[n for n, d_ in zip(names, drop) if d_]}")
    nm_list = list(spec); H = [np.load(spec[k][0]).astype(np.float32) for k in nm_list]
    Xall = np.stack([np.einsum("nrc,rc->n", DW, h) * nrm for h in H], axis=1)
    X = Xall[ok]; r = resid[ok]
    mu = np.array([spec[k][1] for k in nm_list]); tau = np.array([spec[k][2] for k in nm_list])
    en = np.array([float(np.sum(h.astype(np.float64) ** 2)) for h in H])
    pin = (tau <= 0) | (en < args.min_energy)
    r_adj = r - X[:, pin] @ mu[pin]; Xf = X[:, ~pin]; muf = mu[~pin]; tauf = tau[~pin]
    A = Xf.T @ Xf / sd ** 2; b = Xf.T @ r_adj / sd ** 2
    post = np.linalg.solve(A + np.diag(1 / tauf ** 2), b + muf / tauf ** 2); pcov = np.linalg.inv(A + np.diag(1 / tauf ** 2))
    free = np.linalg.lstsq(Xf, r_adj, rcond=None)[0]; fcov = sd ** 2 * np.linalg.pinv(Xf.T @ Xf)
    kap = mu.copy(); kap[~pin] = post
    chi_prior = float(np.sum(((r - X @ mu) / sd) ** 2)); chi_post = float(np.sum(((r - X @ kap) / sd) ** 2)); chi0 = float(np.sum((r / sd) ** 2))
    print(f"{int(ok.sum())} probes, {int((~pin).sum())} free directions; chi2: truth = C {chi0:.1f} | at the priors {chi_prior:.1f} | posterior {chi_post:.1f}")
    j = 0
    for i, k in enumerate(nm_list):
        if pin[i]:
            print(f"   {k:14s} energy {en[i]:8.1f}  pinned at {mu[i]:.2f}")
        else:
            print(f"   {k:14s} energy {en[i]:8.1f}  prior {mu[i]:.2f} +- {tau[i]:.2f}  posterior {post[j]:+.3f} +- {np.sqrt(pcov[j, j]):.3f}   (free fit {free[j]:+.3f} +- {np.sqrt(max(fcov[j, j], 0)):.3f})")
            j += 1
    if args.predict:
        for j, n in enumerate(names):
            if args.predict in n:
                contrib = Xall[j] * kap; order = np.argsort(-np.abs(contrib))[:10]
                print(f"   {n}: residual {resid[j] * 370260:+.0f} squared units, predicted {float(contrib.sum()) * 370260:+.0f} (sd of one residual {sd * 370260:.0f}); largest contributions: "
                      + ", ".join(f"{nm_list[i]} {contrib[i] * 370260:+.0f} (x {Xall[j, i] * 370260:+.0f})" for i in order))
    if args.cand:
        That = sum(k_ * h.astype(np.float64) for k_, h in zip(kap, H))            # posterior truth - C
        for item in args.cand.split(","):
            nm, path = item.split("="); Dc = np.load(ROOT / path).astype(np.float64)
            gain = (2 * float(np.sum(That * Dc * wrow)) - float(np.sum(Dc * Dc * wrow))) * nrm
            print(f"   candidate {nm:10s}: predicted gain against C {gain:+.5f} -> score {ca.S_C - gain:.4f}   (energy against C {float(np.sum(Dc * Dc * wrow)) * nrm:.5f})")
    if args.top:
        okn = [n for n, o in zip(names, ok) if o]
        for i, k in enumerate(nm_list[: args.top]):
            xi = X[:, i]; order = np.argsort(-np.abs(xi))[:8]
            print(f"   largest projections on {k}: " + "; ".join(f"{okn[j][:34]} x {xi[j] * 370260:+.0f} resid {r[j] * 370260:+.0f}" for j in order))
    print("steps: " + ",".join(f"{k}={min(max(v, 0.0), 1.2):.2f}" for k, v, p in zip(nm_list, kap, pin) if not p))


if __name__ == "__main__":
    main()
