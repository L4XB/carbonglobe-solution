#!/usr/bin/env python
"""How large is a probe's public site-noise projection for a given sigma? Compare with the observed scatter of the
null-ish cluster probes (~0.06 in lambda, of which ~0.05 is readout noise) to bound sigma."""
import numpy as np
z = np.load("sim_probes.npz", allow_pickle=True)
D = z["D"].astype(np.float64); names = z["names"]; site = z["site"]; dev = z["dev"].astype(np.float64)
K, N, _ = D.shape; Df = D.reshape(K, -1); cell_site = np.repeat(site, 2); n_sites = int(site.max()) + 1
scen = np.repeat(np.arange(N) // (N // 2), 2)
oa = np.zeros((N, 2)); oa[:, 1] = np.abs(dev[:, 1]); oa = oa.ravel()
es = np.array([np.bincount(cell_site, weights=Df[k] ** 2, minlength=n_sites) for k in range(K)])
n_eff = es.sum(1) ** 2 / np.maximum((es ** 2).sum(1), 1e-30)
conc = n_eff < 300
rng = np.random.default_rng(1)
for sigma in (0.3, 0.5, 0.8, 1.2):
    lam = []
    for it in range(40):
        P = (rng.random(n_sites) < 0.5)[cell_site]
        e = sigma * rng.normal(size=(2, n_sites))[scen, cell_site] * oa
        lam.append((Df[:, P] @ e[P]) / (Df[:, P] ** 2).sum(1))
    lam = np.array(lam)
    print(f"sigma {sigma}: site-noise lambda sd, concentrated probes median {np.median(lam[:, conc].std(0)):.3f}, global probes median {np.median(lam[:, ~conc].std(0)):.3f}")
