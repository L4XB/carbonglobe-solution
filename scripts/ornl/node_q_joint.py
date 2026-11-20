#!/usr/bin/env python
"""Joint node fit: the spatial field from ALL existing-forest pixels, the state level from the mature ones.

At sites where mature (relaxed old-growth) pixels are rare near the node, node_q.py can only offer the all-pixel value, and that population responds
differently (all - relaxed: +0.05, sd 0.12 at fire-possible sites: a STATE effect), while the v4 value takes the few mature pixels wherever they sit
(a LOCATION error, rms 0.06-0.09). Here both are separated in one linear fit over the window node +- 0.5 deg, in log space:

    log(SSP / REF)_p = sum_k w_pk lambda_k + beta x [pixel p is not mature]          (9 node values, one state offset; two robust passes)

lambda_centre is the node's effect for mature forest. Mature = REF 2021 > 0.5 kgC and REF growth 2021 -> 2060 in (-0.3, 0.2) (proxy 4 of q_proxies.py).
Writes work/ornl2522/q_node_joint.npz: sites, lam (n, 2), se (n, 2), beta (n, 2), n_mat (n, 2), n_all (n, 2), neff (n, 2), res (n, 2).
    python scripts/ornl/node_q_joint.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import node_q as nq  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "work" / "ornl2522"


def fit(w: np.ndarray, nonmat: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    X = np.column_stack([w, nonmat.astype(float)]); lam = 1e-3 * len(y); med = float(np.median(y[~nonmat])) if (~nonmat).any() else float(np.median(y))
    P = lam * np.eye(10); P[9, 9] = 1e-6 * len(y); prior = np.r_[np.full(9, med), 0.0]
    k = np.ones(len(y), bool); s = 1.0
    for _ in range(2):
        A = X[k].T @ X[k] + P; b = X[k].T @ y[k] + P @ prior
        th = np.linalg.solve(A, b); res = y - X @ th
        s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-6; k = np.abs(res) < 3 * s
        if k.sum() < 30:
            break
    cov = s ** 2 * np.linalg.inv(A)
    return float(th[4]), float(np.sqrt(cov[4, 4])), float(th[9]), float(s)


def main() -> None:
    sites = np.load(D / "node_win_sites.npy"); n = len(sites)
    L = {k: np.load(D / f"node_win_{k}.npy", mmap_mode="r") for k in ("r0", "r1", "s126", "s585")}
    f = lambda a: np.where(a == -32768, np.nan, a.astype(np.float32) * np.float32(0.01))
    out = {k: np.full((n, 2), np.nan) for k in ("lam", "se", "beta", "neff", "res")}; n_mat = np.zeros((n, 2), int); n_all = np.zeros((n, 2), int)
    t0 = time.time()
    for i in range(n):
        r0 = f(L["r0"][i]); r1 = f(L["r1"][i]); okx = np.isfinite(r0) & np.isfinite(r1) & (r0 > 0.3) & (r1 > 0)
        if okx.sum() < 30:
            continue
        g = np.where(okx, np.log(np.maximum(r1, 1e-6) / np.maximum(r0, 1e-6)), np.nan); mat = okx & (r0 > 0.5) & (g < 0.2) & (g > -0.3)
        for j, sc in enumerate(("s126", "s585")):
            s1 = f(L[sc][i]); m = okx & np.isfinite(s1) & (s1 > 0)
            n_all[i, j] = int(m.sum()); n_mat[i, j] = int((m & mat).sum())
            if m.sum() < 30 or (m & mat).sum() < 12:
                continue
            w = nq.W9[:, m].T; y = np.log(s1[m] / r1[m]); nm = ~mat[m]
            out["neff"][i, j] = float(np.sum(w[:, 4] ** 2))
            out["lam"][i, j], out["se"][i, j], out["beta"][i, j], out["res"][i, j] = fit(w, nm, y)
    np.savez(D / "q_node_joint.npz", sites=sites, n_mat=n_mat, n_all=n_all, **out)
    ok = np.isfinite(out["lam"][:, 1]); print(f"joint fits: {int(ok.sum())} sites (ssp585) after {time.time() - t0:.0f} s; median beta {np.nanmedian(out['beta']):+.3f}, sd {np.nanstd(out['beta']):.3f}; median se {np.nanmedian(out['se']):.4f}")


if __name__ == "__main__":
    main()
