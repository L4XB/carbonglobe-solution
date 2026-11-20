#!/usr/bin/env python
"""Public/private split simulation with BLOCK splits (2026-11-05).

The day-42 level probes and the persistence pair are ~6 sigma inconsistent with a uniform random 50% site split
(public sites: +1.8% second-moment energy but ~-2% mean) and plausible for splits in blocks of 20-300 consecutive site
indices (the site order is geographic). This variant of `split_sim.py` draws the public half in blocks of `--chunk`
sites and makes the non-transferable part of the pseudo-truth block-correlated (share `--wblock` of its variance),
then compares uniform ridge scales with settings that damp the regional / mode probes, whose public readouts fit
block-level noise that does not carry to the private blocks.

    .venv/bin/python scripts/sim/split_sim_block.py --probes work/sim_probes_1105.npz --chunk 100 --wblock 0.5
"""
from __future__ import annotations

import argparse

import numpy as np

REGIONAL = ("spat_dir_cl0_", "spat_dir_mb", "spat_dir_rg", "spat_dir_gm", "spat_dir_region_", "spat_dir_covrule",
            "spat_dir_vulnR", "spat_dir_gbin", "spat_oa_m", "spat_oa_rc", "spat_oafield", "spat_young_m",
            "spat_young_rc", "spat_youngfield", "spat_young_r", "spat_mid_r", "spat_oh_r", "perp_spat_", "spat_dir_agbclsp")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", default="work/sim_probes_1105.npz")
    ap.add_argument("--kappa", type=float, default=0.7)
    ap.add_argument("--sigma", type=float, default=0.45)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--wblock", type=float, default=0.5)
    ap.add_argument("--splits", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rho-tol", type=float, default=0.004,
                    help="keep only splits whose public R^2 energy ratio is within this of the measured rho (1.018)")
    args = ap.parse_args()
    z = np.load(args.probes, allow_pickle=True)
    names = [str(n) for n in z["names"]]
    D = z["D"].astype(np.float64)                        # (K, N, 2)
    K, N, _ = D.shape
    site = z["site"]; rho = float(z["rho"]); dev = z["dev"].astype(np.float64)
    Df = D.reshape(K, -1)                                # (K, 2N) cells
    cell_site = np.repeat(site, 2)
    n_sites = int(site.max()) + 1
    e2_all = (Df ** 2).mean(axis=1)
    G_all = Df @ Df.T / Df.shape[1]
    s_k = z["s_k"]; s_c = z["s_c"]; isR = z["cen_is_R"]
    base = np.where(isR, 0.25171, s_c)
    y_obs = -(s_k - base - rho * e2_all) / 2
    lam_obs = y_obs / (rho * e2_all)
    regional = np.array([n.startswith(REGIONAL) for n in names])
    print(f"{K} probes, {regional.sum()} regional/mode, chunk {args.chunk}, wblock {args.wblock}, kappa {args.kappa}, sigma {args.sigma}")
    rng = np.random.default_rng(args.seed)
    settings = [("uniform", s, s) for s in (0.6, 0.7, 0.8, 0.9)]
    settings += [("damp_regional", g, r) for g, r in ((0.8, 0.5), (0.8, 0.3), (0.9, 0.5), (0.85, 0.4))]
    acc = {st: [] for st in settings}
    e_tr = args.kappa * (lam_obs[:, None] * Df).sum(0)    # transferable residual (cells)
    n_blocks = int(np.ceil(n_sites / args.chunk))
    block_of_site = np.arange(n_sites) // args.chunk
    scen = np.repeat(np.arange(N) // (N // 2), 2)         # template is scenario-major
    oa = np.zeros((N, 2)); oa[:, 1] = np.abs(dev[:, 1]); oa = oa.ravel()
    A = np.linalg.solve(rho * G_all + 1e-4 * np.eye(K), np.eye(K))   # ridge operator, fixed
    R2 = (z["R"].astype(np.float64) ** 2).ravel()
    done = tries = 0
    while done < args.splits and tries < 200000:
        tries += 1
        pub_block = rng.random(n_blocks) < 0.5
        P = pub_block[block_of_site][cell_site]
        if abs(R2[P].mean() / R2.mean() - rho) > args.rho_tol:
            continue
        done += 1
        zs = rng.normal(size=(2, n_sites)); zb = rng.normal(size=(2, n_blocks))
        zz = np.sqrt(1 - args.wblock) * zs + np.sqrt(args.wblock) * zb[:, block_of_site]
        e = e_tr + args.sigma * zz[scen, cell_site] * oa
        DP, DQ = Df[:, P], Df[:, ~P]
        nP, nQ = P.sum(), (~P).sum()
        G_P = DP @ DP.T / nP; G_Q = DQ @ DQ.T / nQ
        yP_true = DP @ e[P] / nP; yQ_true = DQ @ e[~P] / nQ
        rnd = rng.uniform(-0.0005, 0.0005, size=K)
        y_est = (rho * e2_all - np.diag(G_P)) / 2 + yP_true - rnd / 2
        c0 = A @ y_est
        for st in settings:
            _, g, r = st
            c = c0 * np.where(regional, r, g)
            rp = 2 * c @ yP_true - c @ G_P @ c
            rq = 2 * c @ yQ_true - c @ G_Q @ c
            acc[st].append((rp, rq))
    print(f"kept {done} splits of {tries} drawn (rho within {args.rho_tol})")
    print(" setting          global regional   realised public        realised private      priv/pub")
    for st in settings:
        a = np.array(acc[st])
        print(f"  {st[0]:14s} {st[1]:.2f}   {st[2]:.2f}     {a[:,0].mean():.5f} (+-{a[:,0].std():.5f})   {a[:,1].mean():.5f} (+-{a[:,1].std():.5f})   {a[:,1].mean()/a[:,0].mean():.2f}")


if __name__ == "__main__":
    main()
