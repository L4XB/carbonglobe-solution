#!/usr/bin/env python
"""NODE-centred ORNL effect: undo the bilinear interpolation of the 0.01-deg product between its 0.5-deg lookup-table nodes.

Finding of 11-11 04:00 UTC (grid_edges.py, node_profiles.py): log(SSP / REF) of the ORNL DAAC 2522 pixels has kinks exactly at the centres of the
competition's cells and is bilinear in between, i.e. every pixel is a mix of the tables of the four nodes around it. The competition's truth is the
node's own table, so the cell median used so far (q_old of rows v1-v5) is a smoothed value: own node 56 %, the eight neighbours 44 % - and the
board's "amplified effect" (truth = 1.4-2 x ORNL's effect around a pivot of +0.05) is this smoothing seen from the other side.

Per exact site and scenario, on the 100 x 100 pixel window node_windows.py wrote (node +- 0.5 deg = the support of the node's tent weight):

    cell     median log ratio of the pixels of the own cell (the old estimator)
    central  median over the pixels within +-0.05 deg of the node (own weight >= 0.81)
    plain    centre value of a 9-node tent fit in ratio space, R_p = sum_k w_pk rho_k
    rat      centre value of the biomass-weighted tent fit R_p = sum_k w_pk L_k rho_k / sum_k w_pk L_k (the product interpolates BIOMASS, so a rich
             neighbour outweighs a poor node even close to the node); log L_k (relative to the centre) shared by both scenarios, robust loss

for the strict old-growth pixels of the existing forest (REF 2021 > 1.0 kgC, REF growth 2021 -> 2060 in (-0.2, 0.12)), for all existing-forest pixels
(REF 2021 > 0.3) and for the new forest (REF 2060 > 0.05).

Writes work/ornl2522/q_node.npz.   python scripts/ornl/node_q.py [first_site last_site]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "work" / "ornl2522"
N = 100
SIG = 0.03            # pixel noise of the log ratio (sets the strength of the priors)
GROUPS = tuple(os.environ.get("NODE_GROUPS", "old,all,new").split(","))      # NODE_GROUPS=mat NODE_RAT=0: only the relaxed old-growth group, no rational fit
RAT = os.environ.get("NODE_RAT", "1") == "1"


def tent() -> np.ndarray:
    d = (np.arange(N) + 0.5 - N / 2) / 50.0
    t = np.stack([np.maximum(-d, 0.0), 1.0 - np.abs(d), np.maximum(d, 0.0)], 0)             # (3, N): node -1, 0, +1 along one axis
    return np.einsum("ir,jc->ijrc", t, t).reshape(9, N, N)                                 # node k = 3 i + j, centre = 4


W9 = tent()
DCEN = np.maximum(np.abs((np.arange(N) + 0.5 - N / 2) / 50.0)[:, None], np.abs((np.arange(N) + 0.5 - N / 2) / 50.0)[None, :])   # Chebyshev distance, node spacings


def fit_plain(w: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, float, float]:
    """9-node tent fit in ratio space, two robust passes. w (m, 9), r (m,) -> rho (9,), se of rho_centre, robust residual sd."""
    lam = 1e-3 * len(r)
    med = float(np.median(r))
    A = w.T @ w + lam * np.eye(9); b = w.T @ r + lam * med
    rho = np.linalg.solve(A, b); res = r - w @ rho
    s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-6
    k = np.abs(res) < 3 * s
    if k.sum() >= 20:
        A = w[k].T @ w[k] + lam * np.eye(9); b = w[k].T @ r[k] + lam * med
        rho = np.linalg.solve(A, b); res = r - w @ rho
        s = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-6
    cov = s ** 2 * np.linalg.inv(A)
    return rho, float(np.sqrt(cov[4, 4])), float(s)


def fit_rat(ws: list[np.ndarray], lqs: list[np.ndarray], lam0: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray], list[float], float]:
    """Biomass-weighted tent fit, scenarios jointly. ws[s] (m_s, 9), lqs[s] (m_s,) log ratios, lam0[s] start values of log rho (9,).
    Returns ell (9,; centre 0), [log rho (9,)] per scenario, [se of the centre log rho] per scenario, robust residual sd."""
    ns = len(ws)

    def unpack(th: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        ell = np.zeros(9); ell[[0, 1, 2, 3, 5, 6, 7, 8]] = th[:8]
        return ell, [th[8 + 9 * s:17 + 9 * s] for s in range(ns)]

    def resid(th: np.ndarray) -> np.ndarray:
        ell, lams = unpack(th); el = np.exp(ell); out = []
        for s in range(ns):
            den = ws[s] @ el; num = ws[s] @ (el * np.exp(lams[s]))
            out.append(lqs[s] - np.log(np.maximum(num, 1e-9) / np.maximum(den, 1e-9)))
            out.append(SIG * (lams[s] - lam0[s]) / 1.0)                                    # weak prior on the node ratios (sd 1 in log)
        out.append(SIG * th[:8] / 1.0)                                                      # weak prior on the relative node levels (sd 1 in log)
        return np.concatenate(out)

    th0 = np.concatenate([np.zeros(8)] + [np.clip(l, -3, 2) for l in lam0])
    sol = least_squares(resid, th0, loss="soft_l1", f_scale=0.06, max_nfev=60, x_scale=1.0)
    ell, lams = unpack(sol.x)
    m = sum(len(l) for l in lqs); res = sol.fun
    s = 1.4826 * np.median(np.abs(res[:max(m, 1)])) + 1e-6
    try:
        cov = s ** 2 * np.linalg.inv(sol.jac.T @ sol.jac + 1e-9 * np.eye(len(sol.x)))
        ses = [float(np.sqrt(max(cov[8 + 9 * k + 4, 8 + 9 * k + 4], 0.0))) for k in range(ns)]
    except np.linalg.LinAlgError:
        ses = [float("nan")] * ns
    return ell, [np.array(l) for l in lams], ses, float(s)


def estimate(mask: np.ndarray, lq2: list[np.ndarray], do_rat: bool = True) -> dict:
    """mask (N, N) pixel selection, lq2 = [log ratio (N, N) per scenario] -> estimators per scenario."""
    out = dict(cell=[np.nan, np.nan], central=[np.nan, np.nan], plain=[np.nan, np.nan], plain_se=[np.nan, np.nan], rat=[np.nan, np.nan], rat_se=[np.nan, np.nan],
               n=[0, 0], n_cell=[0, 0], n_cen=[0, 0], neff=[0.0, 0.0], res_plain=[np.nan, np.nan], res_rat=np.nan, ell=np.full(9, np.nan),
               rat_nodes=np.full((2, 9), np.nan))
    ws, lqs, lam0 = [], [], []
    for j in range(2):
        m = mask & np.isfinite(lq2[j]); out["n"][j] = int(m.sum())
        mc = m & (DCEN <= 0.5); out["n_cell"][j] = int(mc.sum())
        if mc.sum() >= 12:
            out["cell"][j] = float(np.median(lq2[j][mc]))
        mz = m & (DCEN <= 0.1); out["n_cen"][j] = int(mz.sum())
        if mz.sum() >= 8:
            out["central"][j] = float(np.median(lq2[j][mz]))
        if m.sum() < 30:
            ws.append(None); lqs.append(None); lam0.append(None); continue
        w = W9[:, m].T; y = lq2[j][m]
        out["neff"][j] = float(np.sum(w[:, 4] ** 2))
        rho, se, s = fit_plain(w, np.exp(y))
        out["plain"][j] = float(np.log(max(rho[4], 1e-3))); out["plain_se"][j] = se / max(rho[4], 1e-3); out["res_plain"][j] = s
        ws.append(w); lqs.append(y); lam0.append(np.log(np.maximum(rho, 1e-3)))
    if do_rat and all(w is not None for w in ws) and min(out["neff"]) >= 2.0:
        ell, lams, ses, s = fit_rat(ws, lqs, lam0)
        out["ell"] = ell; out["res_rat"] = s
        for j in range(2):
            out["rat"][j] = float(lams[j][4]); out["rat_se"][j] = ses[j]; out["rat_nodes"][j] = lams[j]
    return out


def main() -> None:
    sites = np.load(D / "node_win_sites.npy")
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, len(sites))
    L = {k: np.load(D / f"node_win_{k}.npy", mmap_mode="r") for k in ("r0", "r1", "s126", "s585", "n1", "n126", "n585")}
    f = lambda a: np.where(a == -32768, np.nan, a.astype(np.float32) * np.float32(0.01))
    keys = ("cell", "central", "plain", "plain_se", "rat", "rat_se", "n", "n_cell", "n_cen", "neff", "res_plain")
    res = {grp: {k: np.full((len(sites), 2), np.nan) for k in keys} for grp in GROUPS}
    extra = {grp: dict(ell=np.full((len(sites), 9), np.nan), res_rat=np.full(len(sites), np.nan), rat_nodes=np.full((len(sites), 2, 9), np.nan)) for grp in res}
    t0 = time.time()
    for i in range(lo, hi):
        r0 = f(L["r0"][i]); r1 = f(L["r1"][i]); n1 = f(L["n1"][i])
        okx = np.isfinite(r0) & np.isfinite(r1) & (r0 > 0.3) & (r1 > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(okx, np.log(np.maximum(r1, 1e-6) / np.maximum(r0, 1e-6)), np.nan)
            lq_e = [np.where(okx & (f(L[sc][i]) > 0), np.log(np.maximum(f(L[sc][i]), 1e-6) / np.maximum(r1, 1e-6)), np.nan) for sc in ("s126", "s585")]
            okn = np.isfinite(n1) & (n1 > 0.05)
            lq_n = [np.where(okn & (f(L[sc][i]) > 0), np.log(np.maximum(f(L[sc][i]), 1e-6) / np.maximum(n1, 1e-6)), np.nan) for sc in ("n126", "n585")]
        old = okx & (r0 > 1.0) & (g < 0.12) & (g > -0.2)
        mat = okx & (r0 > 0.5) & (g < 0.2) & (g > -0.3)            # relaxed old-growth pixels (proxy 4 of q_proxies.py)
        for grp, mask, lq2 in (("old", old, lq_e), ("all", okx, lq_e), ("new", okn, lq_n), ("mat", mat, lq_e)):
            if grp not in GROUPS:
                continue
            e = estimate(mask, lq2, do_rat=RAT)
            for k in keys:
                res[grp][k][i] = e[k]
            extra[grp]["ell"][i] = e["ell"]; extra[grp]["res_rat"][i] = e["res_rat"]; extra[grp]["rat_nodes"][i] = e["rat_nodes"]
        if (i - lo) % 500 == 499:
            print(f"{i + 1 - lo} sites after {time.time() - t0:.0f} s", flush=True)
    tag = ("" if GROUPS == ("old", "all", "new") else "_" + "_".join(GROUPS)) + ("" if (lo, hi) == (0, len(sites)) else f"_{lo}_{hi}")
    np.savez(D / f"q_node{tag}.npz", sites=sites, **{f"{grp}_{k}": v for grp in res for k, v in res[grp].items()},
             **{f"{grp}_{k}": v for grp in extra for k, v in extra[grp].items()})
    o = res[GROUPS[0]]
    print(f"wrote q_node{tag}.npz after {time.time() - t0:.0f} s; old-growth estimates: cell {int(np.isfinite(o['cell'][lo:hi, 0]).sum())}, central {int(np.isfinite(o['central'][lo:hi, 0]).sum())}, "
          f"plain {int(np.isfinite(o['plain'][lo:hi, 0]).sum())}, rational {int(np.isfinite(o['rat'][lo:hi, 0]).sum())}")


if __name__ == "__main__":
    main()
