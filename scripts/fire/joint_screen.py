#!/usr/bin/env python
"""Joint screen: truth - C = sum_i kappa_i h_i fitted to the residuals of all scored board files (see consistency_all.py).
    python scripts/fire/joint_screen.py name1=dir1.npy name2=dir2.npy ...
Prints the joint least-squares kappas with errors, chi2, and the predicted public gain of C + sum kappa_i h_i.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "fire"))
import consistency_all as ca  # noqa: E402


def main() -> None:
    L = ca.load(); names, sc, cen, D, CC, rho = (L[k] for k in ("names", "sc", "cen", "D", "CC", "rho"))
    nrm = rho / (D.shape[1] * 2)
    wrow = ca.row_weights(L); DW = D * wrow            # public-share weights of the cluster sites, as in consistency_all.py
    E = np.einsum("nrc,nrc->n", DW, D) * nrm; base = np.einsum("nrc,nrc->n", DW, CC) * nrm
    s_p = np.array([sc[n] for n in names]); s_c = np.array([sc[Path(cen[n]).name] for n in names])
    resid = (E - (s_p - s_c)) / 2 - base; sd = 0.00041; ok = E > 2e-5
    hs = {}
    for item in sys.argv[1:]:
        nm, path = item.split("=", 1); hs[nm] = np.load(ROOT / path).astype(np.float32)
    X = np.stack([np.einsum("nrc,rc->n", DW, h) * nrm for h in hs.values()], axis=1)[ok]      # (n_probe, n_dir)
    r = resid[ok]
    G = np.array([[float(np.sum(a.astype(np.float64) * b * wrow)) * nrm for b in hs.values()] for a in hs.values()])
    A = X.T @ X; kap = np.linalg.solve(A + 1e-18 * np.eye(len(hs)), X.T @ r); cov = sd ** 2 * np.linalg.inv(A + 1e-18 * np.eye(len(hs)))
    chi0 = float(np.sum((r / sd) ** 2)); chik = float(np.sum(((r - X @ kap) / sd) ** 2)); chi1 = float(np.sum(((r - X.sum(1)) / sd) ** 2))
    print(f"{int(ok.sum())} probes; chi2: truth = C {chi0:.1f} | all kappa = 1 {chi1:.1f} | fitted kappas {chik:.1f}")
    for i, nm in enumerate(hs):
        xi = X[:, i]; k1 = float(xi @ r / max(xi @ xi, 1e-30))
        print(f"   {nm:22s} E_pub {G[i, i]:.5f}  joint kappa {kap[i]:+.3f} +- {np.sqrt(cov[i, i]):.3f}   (alone {k1:+.3f} +- {sd / np.sqrt(max(xi @ xi, 1e-30)):.3f})")
    # predicted gain of the fitted combination in the public metric: 2 b'k - k'Gk with b_i = <T - C, h_i> unknown;
    # under the hypothesis T - C = sum k h: gain = k'Gk
    print(f"   energy of the fitted combination k'Gk = {float(kap @ G @ kap):.5f} (the public gain if the fit were the truth); at kappa = 1: {float(np.ones(len(hs)) @ G @ np.ones(len(hs))):.5f}")


if __name__ == "__main__":
    main()
