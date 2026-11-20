#!/usr/bin/env python
"""Public/private split simulation of the probe -> correction pipeline (runs on the home server).

Pseudo-truth residual e = kappa * sum_k lambda_k D_k (transferable, in the probe directions; lambda_k = the observed
readouts) + site noise (per site and scenario, z ~ N(0,1) x sigma x |old-agb deviation|; does not transfer).
For each random 50/50 site split: simulated scores s_k = s_C + E_P[D^2] - 2 E_P[e D] + rounding, readouts with
rho * E_all[D^2] exactly as `spatial_correction.py` does, James-Stein shrinkage (optionally with extra site noise for
concentrated probes), correction at a given scale; realised public / private gains and the predicted gain.

    .venv/bin/python split_sim.py --probes sim_probes.npz --kappa 0.8 --sigma 0.5 --splits 100
"""
from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", default="sim_probes.npz")
    ap.add_argument("--kappa", type=float, default=0.8)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    z = np.load(args.probes, allow_pickle=True)
    D = z["D"].astype(np.float64)                        # (K, N, 2)
    K, N, _ = D.shape
    site = z["site"]; rho = float(z["rho"]); dev = z["dev"].astype(np.float64)
    Df = D.reshape(K, -1)                                # (K, 2N) cells
    cell_site = np.repeat(site, 2)
    n_sites = int(site.max()) + 1
    e2_all = (Df ** 2).mean(axis=1)
    G_all = Df @ Df.T / Df.shape[1]
    # observed readouts (lambda) in the same convention as spatial_correction (centre score known, rho E_all)
    s_k = z["s_k"]; s_c = z["s_c"]; isR = z["cen_is_R"]
    s_R = 0.25171
    base = np.where(isR, s_R, s_c)
    y_obs = -(s_k - base - rho * e2_all) / 2              # E[(T - C) D]; centres differ from B only for R-centred probes
    lam_obs = y_obs / (rho * e2_all)
    # concentration of each probe (effective number of sites)
    es = np.zeros((K, n_sites))
    for k in range(K):
        es[k] = np.bincount(cell_site, weights=Df[k] ** 2, minlength=n_sites)
    n_eff = es.sum(1) ** 2 / np.maximum((es ** 2).sum(1), 1e-30)
    mis = np.maximum(0.02, 1 / np.sqrt(n_eff))
    rng = np.random.default_rng(args.seed)
    # (method, scale, x): JS with site-noise x, or the joint ridge solve with ridge x (`spatial_correction.py --ridge`)
    settings = [("js", s, x) for s in (0.6, 0.7, 0.85, 1.0) for x in (0.0, 0.07)]
    settings += [("ridge", s, r) for s in (0.7, 0.9, 1.0) for r in (1e-4, 3e-4)]
    # ridge 1e-4 with separate scales for global probes (s) and probes on < 300 effective sites (x)
    settings += [("ridge2", s, x) for s, x in ((0.9, 0.6), (0.9, 0.5), (1.0, 0.6), (0.8, 0.6), (0.8, 0.5))]
    settings += [("ridgejs", s, 1e-4) for s in (0.8, 0.9, 1.0)]
    import os
    if os.environ.get("SIM_GRID"):
        settings = [("ridge", s, r) for s in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85) for r in (3e-5, 1e-4, 3e-4, 1e-3)]
    acc = {st: [] for st in settings}
    shares = []
    e_tr = args.kappa * (lam_obs[:, None] * Df).sum(0)    # transferable residual (cells)
    for it in range(args.splits):
        pub_site = rng.random(n_sites) < 0.5
        P = pub_site[cell_site]
        zsite = rng.normal(size=(2, n_sites))           # scenario x site
        scen = np.repeat(np.arange(N) // (N // 2), 2)    # template is scenario-major
        e_site = np.zeros(2 * N)
        oa = np.zeros((N, 2)); oa[:, 1] = np.abs(dev[:, 1]); oa = oa.ravel()
        e_site = args.sigma * zsite[scen, cell_site] * oa
        e = e_tr + e_site
        DP, DQ = Df[:, P], Df[:, ~P]
        conc_k = n_eff < 300
        share_pub = float(np.sum(DP[conc_k] ** 2) / max(np.sum(Df[conc_k] ** 2), 1e-30))
        shares.append(share_pub)
        nP, nQ = P.sum(), (~P).sum()
        G_P = DP @ DP.T / nP; G_Q = DQ @ DQ.T / nQ
        yP_true = DP @ e[P] / nP; yQ_true = DQ @ e[~P] / nQ
        # simulated readouts: s_k = s_C + E_P[D^2] - 2 E_P[e D] + rounding; y_est uses rho E_all
        rnd = rng.uniform(-0.0005, 0.0005, size=K)
        y_est = (rho * e2_all - np.diag(G_P)) / 2 + yP_true - rnd / 2
        for (method, scale, xnoise) in settings:
            Gest = rho * G_all
            if method == "ridge":
                c = np.linalg.solve(Gest + xnoise * np.eye(K), y_est) * scale
            elif method == "ridgejs":
                sig = np.sqrt(2 * (0.00029 / 2) ** 2 + (mis * rho * e2_all / 2) ** 2)
                c = np.linalg.solve(Gest + xnoise * np.eye(K), y_est)
                c = c * np.clip(1 - 1 / np.maximum((y_est / sig) ** 2, 1e-9), 0, 1) * scale
            elif method == "ridge2":
                c = np.linalg.solve(Gest + 1e-4 * np.eye(K), y_est) * np.where(n_eff < 300, xnoise, scale)
            else:
                sig = np.sqrt(2 * (0.00029 / 2) ** 2 + (mis * rho * e2_all / 2) ** 2 + (xnoise * rho * e2_all * (n_eff < 300)) ** 2)
                c = np.linalg.solve(Gest + 1e-9 * np.eye(K), y_est)
                zz = y_est / sig
                c = c * np.clip(1 - 1 / np.maximum(zz ** 2, 1e-9), 0, 1) * scale
            pred = 2 * c @ y_est - c @ Gest @ c
            rp = 2 * c @ yP_true - c @ G_P @ c
            rq = 2 * c @ yQ_true - c @ G_Q @ c
            acc[(method, scale, xnoise)].append((pred, rp, rq))
    print(f"kappa {args.kappa}  sigma {args.sigma}  splits {args.splits}  K {K}  concentrated probes {(n_eff < 300).sum()}")
    print(" method  scale   x        predicted   realised public   realised private   pub/pred  priv/pub")
    sh = np.array(shares); hi = sh > 0.58
    print(f"public share of the cluster probes' energy: mean {sh.mean():.3f}, splits above 0.58: {hi.sum()}")
    for st in settings:
        a = np.array(acc[st])
        if hi.sum() >= 3:
            b = a[hi]
            print(f"  [share>0.58] {st[0]:7s} {st[1]:.2f} {st[2]:.0e}: public {b[:,1].mean():.5f}  private {b[:,2].mean():.5f}")
        print(f"  {st[0]:5s}  {st[1]:.2f}  {st[2]:.0e}   {a[:,0].mean():.5f}    {a[:,1].mean():.5f} (+-{a[:,1].std():.5f})   "
              f"{a[:,2].mean():.5f} (+-{a[:,2].std():.5f})   {a[:,1].mean()/a[:,0].mean():.2f}      {a[:,2].mean()/a[:,1].mean():.2f}")


if __name__ == "__main__":
    main()
