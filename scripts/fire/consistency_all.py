#!/usr/bin/env python
"""Screen a hypothesis direction h (truth = C + kappa h) against ALL scored board files, whatever their centre.
For a scored file p with centre c_p (work/spat_centers.json, score of the centre known): D_p = file - c_p,
  b_p = (E_pub,p - (s_p - s_c)) / 2 = <T - c_p, D_p>_pub ,   resid_p = b_p - <C - c_p, D_p>_pub = <T - C, D_p>_pub ,
and the hypothesis predicts resid_p = kappa <h, D_p>_pub. Public inner products: site-share model of build_ed_probes.
The stacked D_p are cached in work/fire/probe_cache.npz (rebuilt when the set of scored files changes).
    python scripts/fire/consistency_all.py work/fire/dir_mer_gs.npy [more .npy] [--top 8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

C_PATH = "results/spatcorr_stack_20261106_s08.csv"
S_C = 0.0438


def load() -> dict:
    k = shift.template_keys(); site = k.site.to_numpy()
    rd = lambda p: shift.read_predictions(ROOT / p).astype(np.float64)
    # CA_SCORES = alternative score file (synthetic boards of scripts/sim/synth_board.py); the probe cache depends only on the file names
    sc = json.load(open(ROOT / __import__("os").environ.get("CA_SCORES", "results/board_scores.json"))); cen = json.load(open(ROOT / "work/spat_centers.json"))
    sc[Path(C_PATH).name] = sc.get(Path(C_PATH).name, S_C)
    clsite = np.load(ROOT / "work/collapse_cluster_labels.npy") == 0
    ps = rd("results/persistence_shifted.csv")
    rho = 2 * (sc["persistence_shifted.csv"] - 2 * sc["persistence_s05.csv"] + sc["zeros.csv"]) / float(np.mean(ps ** 2))
    cache_f = ROOT / "work/fire/probe_cache.npz"
    names = []
    for n in sorted(sc):
        c = cen.get(n)
        if c is None or c == "R" or not (ROOT / "results" / n).exists():
            continue
        cn = Path(c).name
        if cn not in sc or not (ROOT / c).exists():
            continue
        names.append(n)
    if cache_f.exists():
        z = np.load(cache_f, allow_pickle=True)
        if list(z["names"]) == names:
            return dict(site=site, rho=rho, clsite=clsite, sc=sc, cen=cen, names=names, D=z["D"], CC=z["CC"], rd=rd)
    C = rd(C_PATH); cents: dict[str, np.ndarray] = {}
    D = np.zeros((len(names), C.shape[0], 2), dtype=np.float32); CC = np.zeros_like(D)
    for j, n in enumerate(names):
        c = cen[n]
        if c not in cents: cents[c] = rd(c)
        D[j] = rd("results/" + n) - cents[c]; CC[j] = C - cents[c]
    np.savez(cache_f, names=np.array(names, dtype=object), D=D, CC=CC)
    return dict(site=site, rho=rho, clsite=clsite, sc=sc, cen=cen, names=names, D=D, CC=CC, rd=rd)


def row_weights(L: dict) -> np.ndarray:
    """Public weight per row (share / 0.5): cluster sites get their public-share posterior from the mirror probes."""
    site, rho, clsite, sc, cen, names, D, rd = (L[k] for k in ("site", "rho", "clsite", "sc", "cen", "names", "D", "rd"))
    # cluster public-share posterior (mirror probes), as in build_ed_probes
    W_, r_ = [], []
    for n_ in sorted(sc):
        if not n_.startswith("spat_") or n_.endswith("_mirror.csv"): continue
        c_ = cen.get(n_, "R"); isR = c_ == "R" or c_.endswith("latent_ref_y0_m114114.csv"); mir = n_.replace(".csv", "_mirror.csv")
        if not (mir in sc and not isR) or n_ not in names: continue
        j = names.index(n_); Dk = D[j].astype(np.float64); es = np.bincount(site, weights=(Dk * Dk).sum(1), minlength=6171)
        if es[clsite].sum() / es.sum() <= 0.5: continue
        Cp = rd(c_); clip_ = rd("results/" + mir) - (Cp - Dk); ea = float(np.mean(Dk * Dk)); ec = float(np.mean((Dk - clip_) ** 2))
        r_.append(((sc[n_] + sc[mir] - 2 * sc[Path(c_).name]) / 2) / (rho * (ea + ec) / 2)); W_.append(es[clsite] / es[clsite].sum())
    W_, r_ = np.stack(W_), np.array(r_); sp = 0.5 * rho; S0 = 0.25
    Sig = S0 * W_ @ W_.T + np.diag(np.full(len(r_), (0.03 * sp) ** 2)); K = np.linalg.solve(Sig, W_).T
    post = 0.5 + S0 * K @ (r_ * sp - W_ @ np.full(W_.shape[1], 0.5))
    share = np.full(6171, 0.5); share[np.where(clsite)[0]] = np.clip(post, 0.02, 0.98)
    wrow = ((share / 0.5)[site]).astype(np.float32)[:, None]
    return wrow


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    L = load(); site, rho, clsite, sc, cen, names, D, CC, rd = (L[k] for k in ("site", "rho", "clsite", "sc", "cen", "names", "D", "CC", "rd"))
    wrow = row_weights(L)
    nrm = rho / (D.shape[1] * 2)
    DW = D * wrow                                                        # (n, rows, 2) weighted probe directions
    E = np.einsum("nrc,nrc->n", DW, D) * nrm
    base = np.einsum("nrc,nrc->n", DW, CC) * nrm
    s_p = np.array([sc[n] for n in names]); s_c = np.array([sc[Path(cen[n]).name] for n in names])
    b = (E - (s_p - s_c)) / 2; resid = b - base
    sd = 0.00041                                                         # two 3-decimal board scores
    ok = E > 2e-5
    print(f"{int(ok.sum())} scored files with a scored centre; chi2 of the residuals under truth = C: {float(np.sum((resid[ok] / sd) ** 2)):.1f}")
    for path in args.dirs:
        h = np.load(ROOT / path).astype(np.float32)
        x = np.einsum("nrc,rc->n", DW, h) * nrm; eh = float(np.sum(h.astype(np.float64) ** 2 * wrow) * nrm)
        xo, ro = x[ok], resid[ok]
        kap = float(xo @ ro / max(xo @ xo, 1e-30)); se = sd / np.sqrt(max(xo @ xo, 1e-30))
        chi0 = float(np.sum((ro / sd) ** 2)); chi1 = float(np.sum(((ro - xo) / sd) ** 2)); chik = float(np.sum(((ro - kap * xo) / sd) ** 2))
        print(f"== {Path(path).name}: E_pub(h) {eh:.5f}; kappa {kap:+.3f} +- {se:.3f}; chi2 truth = C {chi0:.1f} | C + h {chi1:.1f} | C + kappa h {chik:.1f}  (gain at kappa {max(kap, 0) ** 2 * eh if kap < 1 else (2 * kap - 1) * eh:.5f} if h were followed by min(kappa, 1))")
        order = np.argsort(-np.abs(x * ok))[:args.top]
        for j in order:
            print(f"     {names[j][:46]:46s} centre {Path(cen[names[j]]).name[:28]:28s} E {E[j]:.5f}  resid/E {resid[j] / E[j]:+.3f}  predicted {x[j] / E[j]:+.3f}  (cos {x[j] / np.sqrt(E[j] * eh):+.2f})")


if __name__ == "__main__":
    main()
