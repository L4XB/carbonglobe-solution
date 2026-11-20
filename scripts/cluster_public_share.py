#!/usr/bin/env python
"""Predict the public-energy ratio r = E_pub[D^2] / (rho E_all[D^2]) of every un-mirrored stack-centred probe whose energy
sits partly on the collapse cluster, from the mirrored cluster pairs.

The public/private split is by site (cluster mirror pairs show r 0.89-1.34, impossible for a row-level split). Each
mirrored probe k observes the public share of its cluster energy, sum_s p_s w_ks (w = energy shares over the 204 cluster
sites, p_s = public indicator). A Gaussian approximation of the Bernoulli(q) prior gives the posterior mean of p, and
with it r for every other probe (the non-cluster part of a probe's energy is taken at r = 1). Leave-one-pair-out rmse of
r: 0.089 vs 0.155 for the mean ratio and 0.207 for r = 1 (2026-11-04). Writes work/cluster_ratio_pred.json, read by
spatial_correction.py / local_eval.py (--ratio-file).

    ./.venv/bin/python scripts/cluster_public_share.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

Q_PUBLIC = 0.5
NOISE_SD = 0.03


def main() -> None:
    sc = json.load(open(ROOT / "results" / "board_scores.json")); centers = json.load(open(ROOT / "work" / "spat_centers.json"))
    rd = lambda p: shift.read_predictions(p).astype(np.float64)
    site = shift.template_keys()["site"].to_numpy()
    clsite = np.load(ROOT / "work" / "collapse_cluster_labels.npy") == 0
    C0 = sc["zeros.csv"]; ps = rd(ROOT / "results" / "persistence_shifted.csv")
    rho = 2 * (sc["persistence_shifted.csv"] - 2 * sc["persistence_s05.csv"] + C0) / float(np.mean(ps ** 2))
    names, share, r, W = [], [], [], []
    for n in sorted(sc):
        if not n.startswith("spat_") or n.endswith("_mirror.csv"):
            continue
        c = centers.get(n, "R")
        isR = c == "R" or c.endswith("latent_ref_y0_m114114.csv")
        Cp = rd(ROOT / "results" / "latent_ref_y0_m114114.csv") if isR else rd(ROOT / c)
        D = rd(ROOT / "results" / n) - Cp
        es = np.bincount(site, weights=(D * D).sum(1), minlength=6171)
        tot = es.sum()
        if tot <= 0:
            continue
        mir = n.replace(".csv", "_mirror.csv"); rk = np.nan
        if mir in sc and not isR:
            clip = rd(ROOT / "results" / mir) - (Cp - D)
            e_all = float(np.mean(D * D)); e_clip = float(np.mean((D - clip) ** 2))
            rk = ((sc[n] + sc[mir] - 2 * sc[Path(c).name]) / 2) / (rho * (e_all + e_clip) / 2)
        names.append(n); share.append(es[clsite].sum() / tot); r.append(rk)
        W.append(es[clsite] / max(es[clsite].sum(), 1e-30))
    share, r, W = np.array(share), np.array(r), np.stack(W)
    mir = np.isfinite(r) & (share > 0.5)
    sp = Q_PUBLIC * rho
    A, b = W[mir], r[mir] * sp
    m0 = np.full(W.shape[1], Q_PUBLIC); S0 = Q_PUBLIC * (1 - Q_PUBLIC)
    Sig = S0 * A @ A.T + np.diag(np.full(mir.sum(), (NOISE_SD * sp) ** 2))
    K = np.linalg.solve(Sig, A).T                               # (n_sites, n_mir)
    post = m0 + S0 * K @ (b - A @ m0)
    r_cl = (W @ post) / sp
    var_cl = (S0 * np.einsum("ks,ks->k", W, W) - S0 ** 2 * np.einsum("ks,sj,kj->k", W, A.T @ np.linalg.solve(Sig, np.eye(mir.sum())) @ A, W)) / sp ** 2
    out = {}
    for i, n in enumerate(names):
        if share[i] > 0.3 and not np.isfinite(r[i]):
            out[n] = {"r": float(share[i] * r_cl[i] + (1 - share[i])), "sd": float(share[i] * np.sqrt(max(var_cl[i], 0.0))), "cluster_share": float(share[i])}
    # probes whose public-energy ratio was derived from split-conditioned simulations (e.g. the level probes)
    static = ROOT / "work" / "split_ratio_static.json"
    if static.exists():
        out.update({k: {"r": v["r"], "sd": v["sd"], "cluster_share": 0.0} for k, v in json.loads(static.read_text()).items()})
    (ROOT / "work" / "cluster_ratio_pred.json").write_text(json.dumps(out, indent=1))
    vals = np.array([v["r"] for v in out.values()])
    print(f"{mir.sum()} mirrored cluster pairs; predicted r for {len(out)} un-mirrored probes: mean {vals.mean():.3f}, range {vals.min():.3f}-{vals.max():.3f}")
    print(f"cluster public fraction (posterior mean over the 204 sites) {post.mean():.3f}")


if __name__ == "__main__":
    main()
