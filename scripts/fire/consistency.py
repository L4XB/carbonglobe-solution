#!/usr/bin/env python
"""Consistency of a hypothesis direction h (truth - C = kappa * h) with ALL scored probes centred on C.
For probe p with D_p = file - C: observed b_p = (E_pub,p - (s_p - s_C)) / 2 = <T - C, D_p>_pub; under the hypothesis
b_p = kappa <h, D_p>_pub. Public inner products use the site-share model of build_ed_probes (rho, cluster posterior).
    python scripts/fire/consistency.py work/fire/dir_clim40_both.npy [more .npy]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

S_C = 0.0438
C_PATH = "results/spatcorr_stack_20261106_s08.csv"


def main() -> None:
    k = shift.template_keys(); site = k.site.to_numpy()
    rd = lambda p: shift.read_predictions(ROOT / p).astype(np.float64)
    C = rd(C_PATH)
    sc = json.load(open(ROOT / "results/board_scores.json")); cen = json.load(open(ROOT / "work/spat_centers.json"))
    clsite = np.load(ROOT / "work/collapse_cluster_labels.npy") == 0
    ps = rd("results/persistence_shifted.csv")
    rho = 2 * (sc["persistence_shifted.csv"] - 2 * sc["persistence_s05.csv"] + sc["zeros.csv"]) / float(np.mean(ps ** 2))
    # cluster public-share posterior (same as the builder)
    W_, r_ = [], []
    for n_ in sorted(sc):
        if not n_.startswith("spat_") or n_.endswith("_mirror.csv"): continue
        c_ = cen.get(n_, "R"); isR = c_ == "R" or c_.endswith("latent_ref_y0_m114114.csv"); mir = n_.replace(".csv", "_mirror.csv")
        if not (mir in sc and not isR): continue
        Cp = rd(c_); Dk = rd("results/" + n_) - Cp; es = np.bincount(site, weights=(Dk * Dk).sum(1), minlength=6171)
        if es[clsite].sum() / es.sum() <= 0.5: continue
        clip_ = rd("results/" + mir) - (Cp - Dk); ea = float(np.mean(Dk * Dk)); ec = float(np.mean((Dk - clip_) ** 2))
        r_.append(((sc[n_] + sc[mir] - 2 * sc[Path(c_).name]) / 2) / (rho * (ea + ec) / 2)); W_.append(es[clsite] / es[clsite].sum())
    W_, r_ = np.stack(W_), np.array(r_); sp = 0.5 * rho; S0 = 0.25
    Sig = S0 * W_ @ W_.T + np.diag(np.full(len(r_), (0.03 * sp) ** 2)); K = np.linalg.solve(Sig, W_).T
    post = 0.5 + S0 * K @ (r_ * sp - W_ @ np.full(W_.shape[1], 0.5))
    share = np.full(6171, 0.5); share[np.where(clsite)[0]] = np.clip(post, 0.02, 0.98)
    wrow = (share / 0.5)[site]                                               # row weights relative to the mean public share

    def ip(a: np.ndarray, b: np.ndarray) -> float:
        return rho * float(np.sum((a * b).sum(1) * wrow)) / a.size

    probes = [n for n in sorted(sc) if cen.get(n, "").endswith(Path(C_PATH).name) and (ROOT / "results" / n).exists()]
    D = {n: rd("results/" + n) - C for n in probes}
    b_obs = {n: (ip(D[n], D[n]) - (sc[n] - S_C)) / 2 for n in probes}
    sd_b = 0.00042 / 2                                                       # 3-decimal board rounding and the error of s_C
    print(f"{len(probes)} scored probes centred on C; rho {rho:.3f}")
    for path in sys.argv[1:]:
        h = np.load(ROOT / path).astype(np.float64)
        x = np.array([ip(h, D[n]) for n in probes]); y = np.array([b_obs[n] for n in probes])
        kap = float(x @ y / max(x @ x, 1e-30)); se = sd_b / np.sqrt(max(x @ x, 1e-30))
        chi0 = float(np.sum((y / sd_b) ** 2)); chi1 = float(np.sum(((y - x) / sd_b) ** 2)); chik = float(np.sum(((y - kap * x) / sd_b) ** 2))
        print(f"== {Path(path).name}: E_pub(h) {ip(h, h):.5f}; kappa from past probes {kap:+.3f} +- {se:.3f}; chi2 (truth = C) {chi0:.1f}, (truth = C + h) {chi1:.1f}, (C + kappa h) {chik:.1f}")
        order = np.argsort(-np.abs(x))[:10]
        for j in order:
            n = probes[j]; e = ip(D[n], D[n])
            print(f"     {n[:44]:44s} E_pub {e:.5f}  observed lambda {y[j] / e:+.3f}  predicted under h {x[j] / e:+.3f}   (cos {x[j] / np.sqrt(e * ip(h, h)):+.3f})")


if __name__ == "__main__":
    main()
