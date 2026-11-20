#!/usr/bin/env python
"""Cluster keep/collapse assignment probe (strategist recipe D2).

Per collapse-cluster site: logistic regression of sign(board field f) on site covariates (log stand size, driest-month
precipitation normal, window drought score, scenario LAI gap, hour-invariant first-year GPP drop, the stack's own
keep ratio), leave-one-site-out. Each cluster mid/old agb row is then moved a fraction toward 0.9 x 0.876 R (P(keep) >
0.5) or 0.32 x 0.876 R (collapse), weighted by |P - 0.5|. Writes work/dir_cl0_assign.npy for dir_probe.py.

    ./.venv/bin/python scripts/cluster_classifier_probe.py --frac 0.3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cgb import io, shift  # noqa: E402
from cgb import mortality as mort  # noqa: E402


def logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 200) -> np.ndarray:
    """Newton-Raphson L2 logistic regression; X includes an intercept column."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        step = np.linalg.solve(H, g); w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frac", type=float, default=0.3)
    args = ap.parse_args()
    from mort_trigger import build_indices
    rd = lambda p: shift.read_predictions(p).astype(np.float64)
    B = rd(ROOT / "results/stack_20261010.csv"); R = rd(ROOT / "results/latent_ref_y0_m114114.csv")
    keys = shift.template_keys(); site = keys["site"].to_numpy(); age = keys["age"].to_numpy()
    s5 = (keys["scenario"] == "ssp585").to_numpy(); key = site * 2 + s5
    cl = np.load(ROOT / "work/collapse_cluster_labels.npy") == 0
    f = np.load(ROOT / "work/cluster0_field_f.npy")
    csites = np.where(cl)[0]
    # site-scenario covariates
    m_old = cl[site] & (age >= 150) & (R[:, 1] > 0.05)
    def smean(v, mm):
        num = np.bincount(key[mm], weights=v, minlength=12342); cnt = np.bincount(key[mm], minlength=12342)
        return np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)
    keep_ratio = smean(B[m_old, 1] / R[m_old, 1], m_old)
    size = smean(np.log(R[m_old, 1]), m_old)
    z = np.load(ROOT / "work/strategist/flux_diseq.npz"); zd = np.load(ROOT / "work/strategist/d1_hour_invariant_all.npz")
    feats = {"keep": keep_ratio, "size": size}
    pmin, drought, lai_gap, d1 = (np.full(12342, np.nan) for _ in range(4))
    for si, s in enumerate(io.SCENARIOS):
        sea = np.load(ROOT / f"work/cache/test_{s}_season.npy")
        pmin[np.arange(6171) * 2 + si] = np.asarray(sea[:, 1, :]).min(1)
        ix = build_indices(np.asarray(mort.open_monthly(s)))
        sc_ = np.zeros(6171)
        for k in ("spi6", "spi1"):
            v = np.asarray(ix[k]); sc_ += np.nanmean(v, 1) / np.nanstd(np.nanmean(v, 1)[cl])
        drought[np.arange(6171) * 2 + si] = sc_
        lai_gap[np.arange(6171) * 2 + si] = z["lai_gap"][:, 14]
        d1[np.arange(6171) * 2 + si] = zd[f"d1_{s}"][:, 14]
    feats.update(pmin=np.log1p(pmin), drought=drought, lai_gap=lai_gap, d1=d1)
    ks = np.concatenate([csites * 2, csites * 2 + 1])
    X = np.column_stack([feats[k][ks] for k in feats])
    X = np.nan_to_num((X - np.nanmean(X, 0)) / np.nanstd(X, 0))
    X = np.column_stack([np.ones(len(ks)), X])
    y = (f[ks // 2] < 0).astype(float)          # field_f < 0: the board lowers the collapse -> keep
    p = np.zeros(len(ks))
    for s_ in csites:                            # leave-one-site-out (both scenarios of the site)
        te = (ks // 2) == s_
        w = logistic_fit(X[~te], y[~te])
        p[te] = 1 / (1 + np.exp(-X[te] @ w))
    acc = np.mean((p > 0.5) == (y > 0.5))
    print(f"leave-one-site-out accuracy {acc:.3f} (base rate {max(y.mean(), 1 - y.mean()):.3f}); features {list(feats)}")
    w = logistic_fit(X, y); print("coefficients", dict(zip(["const"] + list(feats), np.round(w, 2))))
    P = np.full(12342, np.nan); P[ks] = p
    D = np.zeros_like(B)
    m = cl[site] & (age >= 50) & (R[:, 1] > 0.05)
    pk = P[key[m]]
    tgt = np.where(pk > 0.5, 0.9, 0.32) * 0.876 * R[m, 1]
    D[m, 1] = args.frac * np.abs(2 * pk - 1) * (tgt - B[m, 1])
    np.save(ROOT / "work/dir_cl0_assign.npy", D)
    print(f"wrote work/dir_cl0_assign.npy: rows {m.sum()}, raw E {np.mean(D ** 2):.5f}")


if __name__ == "__main__":
    main()
