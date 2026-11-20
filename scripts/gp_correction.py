#!/usr/bin/env python
"""Structured-prior (GP) correction of the 10-10 stack from every scored probe readout.

Model over all cells: r = T - B = Z beta + eps, beta ~ N(0, tau_g^2) per feature group, eps a white cell prior (the
ridge-in-span correction of `spatial_correction.py` is the special case Z = 0). Probe k measures
y_k = E_pub[r D_k] = rho_k mean(D_k r) + noise_k, so with A_kf = rho_k mean(D_k Z_f) and G the probes' Gram matrix

    C = A T A' + v2 diag(rho) G diag(rho) + S,   w = C^-1 y,   r_hat = Z T A' w + v2 sum_k rho_k D_k w_k.

Features are block one-hots (band x column x in/out of the collapse cluster) times a scale variable (R, the stack's
change B - R, the chronodyn-58 pull CD - B, the age-shift-46 pull A46 - B) times a site/row covariate (constant,
smooth graph modes, climate, soil, profile, scenario sign, log age, flux disequilibrium d1/e/LAI gap, log stand size,
cluster modes). Hyperparameters (one tau per scale x covariate family, and v2) by type-II maximum likelihood; the
leave-one-probe-out predictions of y_k show whether the features extrapolate to directions no other probe measured.

    python scripts/gp_correction.py --out results/gpcorr_x.csv --scale 0.8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cgb import io, shift  # noqa: E402
from local_eval import readouts  # noqa: E402

SCALES = ("R", "dev", "cd", "a46")
EXTRA_COV = False   # gp_screen.py sets True to add the spin-up mortality-load covariates
FAMILIES = ("const", "modes", "env", "scen", "age", "flux", "size", "cmodes")


def probe_noise(sc: dict, names: list[str], Dm: np.ndarray, rho: float, fac: np.ndarray, site: np.ndarray) -> np.ndarray:
    """Readout noise per probe, as in spatial_correction.py (rounding, public-energy mismatch, s_R error)."""
    centers = json.loads((ROOT / "work" / "spat_centers.json").read_text())
    sig = []
    for k, n in enumerate(names):
        c = centers.get(n, "R"); isR = c == "R" or c.endswith("latent_ref_y0_m114114.csv")
        d = Dm[:, k].reshape(-1, 2)
        e2 = rho * fac[k] * float(np.mean(d * d))
        es = np.bincount(site, weights=(d * d).sum(1), minlength=site.max() + 1)
        mis = max(0.02, float(np.sqrt(np.sum(es ** 2)) / np.sum(es)))
        if n.replace(".csv", "_mirror.csv") in sc and not isR:
            s = 0.0007 / 4
        elif isR:
            s = np.sqrt((0.00029 / 2) ** 2 + (mis * e2 / 2) ** 2 + (0.0003 / 2) ** 2)
        else:
            s = np.sqrt(2 * (0.00029 / 2) ** 2 + (mis * e2 / 2) ** 2)
        sig.append(s)
    return np.array(sig)


def covariates(keys, R: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Standardized row covariates (185130, P) and their family labels."""
    site = keys["site"].to_numpy(); age = keys["age"].to_numpy(); scen = keys["scenario"].to_numpy()
    s5 = scen == "ssp585"
    cols, fam = [], []

    def add(v: np.ndarray, f: str) -> None:
        v = np.nan_to_num(np.asarray(v, dtype=np.float64), nan=np.nanmean(v))
        sd = v.std()
        if sd > 1e-12:
            cols.append((v - v.mean()) / sd); fam.append(f)
    cols.append(np.ones(len(site))); fam.append("const")
    modes = np.load(ROOT / "work" / "site_graph_modes240.npy")[:, 1:13].astype(np.float64)
    for j in range(modes.shape[1]):
        add(modes[site, j], "modes")
    env = {}
    for s in io.SCENARIOS:
        t = np.load(ROOT / "work" / "cache" / f"test_{s}_annual.npy", mmap_mode="r")
        env[s] = np.c_[np.asarray(t[:, :, 0]).mean(1), np.log1p(np.asarray(t[:, :, 1]).mean(1)), np.asarray(t[:, :, 0]).std(1),
                       np.log1p(np.asarray(t[:, :, 1])).std(1), np.asarray(t[:, 0, 129:136])]
    E = np.where(s5[:, None], env["ssp585"][site], env["ssp126"][site])
    for j in range(E.shape[1]):
        add(E[:, j], "env")
    prof = np.zeros((2, 6171, 2)); cnt = np.zeros((2, 6171))
    np.add.at(prof, (s5.astype(int), site), R); np.add.at(cnt, (s5.astype(int), site), 1)
    prof /= np.maximum(cnt, 1)[..., None]
    for j in range(2):
        add(prof[s5.astype(int), site, j], "env")
    add(np.where(s5, 1.0, -1.0), "scen")
    band = shift.band_of(age); la = np.log(age.astype(np.float64))
    for b in np.unique(band):
        m = band == b; la[m] -= la[m].mean()
    add(la, "age")
    z = np.load(ROOT / "work" / "strategist" / "flux_diseq.npz")
    aidx = np.searchsorted(np.asarray(io.SEED_AGES), age)
    for key in ("d1", "e"):
        v = np.where(s5, z[f"{key}_ssp585"][site, aidx], z[f"{key}_ssp126"][site, aidx])
        add(v, "flux")
    add(z["lai_gap"][site, aidx], "flux")
    add(np.log1p(R[:, 1] * 3.39567 * 1.14114), "size")
    cm = np.load(ROOT / "work" / "cluster0_modes.npy")
    for j in range(1, 11):
        add(cm[site, j], "cmodes")
    if EXTRA_COV:
        # spin-up mortality load (summed relative agb drops along the site's own profile) and its dip count
        ages = np.asarray(io.SEED_AGES, float); ai = np.searchsorted(ages, age)
        prof = np.zeros((2, 6171, 15)); prof[s5.astype(int), site, ai] = R[:, 1]
        rel = prof[:, :, 1:] / np.maximum(prof[:, :, :-1], 1e-6)
        load = (np.clip(1 - rel, 0, 1) * (prof[:, :, :-1] > 0.05)).sum(2)
        add(load[s5.astype(int), site], "load")
        add(((rel < 0.9) & (prof[:, :, :-1] > 0.05)).sum(2)[s5.astype(int), site], "load")
    return np.stack(cols, 1), fam


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="")
    ap.add_argument("--scale", type=float, default=0.8)
    ap.add_argument("--groups", default="scale_family", choices=["scale_family", "family", "scale"])
    ap.add_argument("--no-features", action="store_true", help="span-only prior (the ridge analogue) for comparison")
    ap.add_argument("--hcap", type=float, default=35.0002 / (1.14114 * 10.90645))
    args = ap.parse_args()
    B, rho, Dm, y, fac, sc = readouts()
    names = sorted(n for n in sc if (n.startswith("spat_") or n.startswith("perp_spat_")) and not n.endswith("_mirror.csv"))
    K = len(y); N = B.size
    keys = shift.template_keys(); site = keys["site"].to_numpy(); age = keys["age"].to_numpy()
    sig = probe_noise(sc, names, Dm, rho, fac, site)
    rk = rho * fac
    G = Dm.T @ Dm / N
    rd = lambda p: shift.read_predictions(p).astype(np.float64)
    R = rd(ROOT / "results" / "latent_ref_y0_m114114.csv")
    CD = rd(ROOT / "results" / "chronodyn1g58_g58.csv")
    A46 = 0.876 * np.load(ROOT / "work" / "ageshift46_dec.npy")
    svars = {"R": R, "dev": B - R, "cd": CD - B, "a46": A46 - B}
    Zc, fam = covariates(keys, R)
    fam = np.array(fam)
    cl = (np.load(ROOT / "work" / "collapse_cluster_labels.npy") == 0)[site]
    band = shift.band_of(age)
    blocks = [(b, c, inside) for b in ("young", "mid", "old") for c in (0, 1) for inside in (False, True)]
    A_parts, gid, meta = [], [], []
    group_names = []

    def group_of(s: str, f: str) -> int:
        g = {"scale_family": f"{s}:{f}", "family": f, "scale": s}[args.groups]
        if g not in group_names:
            group_names.append(g)
        return group_names.index(g)
    for b, c, inside in blocks:
        rows = np.where((band == b) & (cl == inside))[0]
        if len(rows) == 0:
            continue
        cells = rows * 2 + c
        Zb = Zc[rows]
        for s in SCALES:
            v = svars[s][rows, c]
            norm = np.sqrt(np.mean((v[:, None] * Zb) ** 2, axis=0)) + 1e-12      # unit RMS per feature on its block
            W = Dm[cells] * v[:, None]                                            # (n_b, K)
            A_parts.append(rk[:, None] * (W.T @ Zb) / N / norm[None, :])
            for j in range(Zb.shape[1]):
                gid.append(group_of(s, fam[j])); meta.append((rows, c, s, j, norm[j]))
    A = np.concatenate(A_parts, 1); gid = np.array(gid); ng = len(group_names)
    print(f"{K} probes, {A.shape[1]} features in {ng} groups; |y| median {np.median(np.abs(y)):.2e}, sig median {np.median(sig):.2e}")
    RGR = (rk[:, None] * G * rk[None, :])
    S = np.diag(sig ** 2)
    # per-group A T A' pieces
    AAg = np.stack([A[:, gid == g] @ A[:, gid == g].T for g in range(ng)])

    def cov(theta: np.ndarray) -> np.ndarray:
        v2 = np.exp(theta[0]); C = v2 * RGR + S
        if not args.no_features:
            C = C + np.tensordot(np.exp(theta[1:]), AAg, axes=1)
        return C

    def nll(theta: np.ndarray) -> float:
        C = cov(theta)
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            return 1e10
        a = np.linalg.solve(L, y)
        return float(0.5 * a @ a + np.sum(np.log(np.diag(L))))
    th0 = np.r_[np.log(4e-4), np.full(ng, np.log(1e-4))]
    if args.no_features:
        th0 = th0[:1]
    best = None
    for start in (0.0, -3.0, 3.0):
        r = minimize(nll, th0 + np.r_[0.0, np.full(len(th0) - 1, start)], method="L-BFGS-B",
                     bounds=[(-25, 5)] * len(th0))
        if best is None or r.fun < best.fun:
            best = r
    th = best.x
    C = cov(th); Ci = np.linalg.inv(C); w = Ci @ y
    loo = y - w / np.diag(Ci)
    e2 = rk * np.diag(G)
    lam, lam_loo = y / e2, loo / e2
    print(f"neg log marginal likelihood {best.fun:.2f}; v2 {np.exp(th[0]):.2e}")
    if not args.no_features:
        for g in np.argsort(-th[1:]):
            print(f"  tau^2 {np.exp(th[1 + g]):.2e}  {group_names[g]}")
    print(f"LOO: corr(y, loo) {np.corrcoef(y, loo)[0, 1]:+.3f}; corr(lambda, lambda_loo) {np.corrcoef(lam, lam_loo)[0, 1]:+.3f}; "
          f"explained {1 - np.sum((y - loo) ** 2) / np.sum(y ** 2):+.3f}; z-rms {np.sqrt(np.mean((y - loo) ** 2 * np.diag(Ci) ** 1)):.2f}")
    # posterior mean correction
    v2 = np.exp(th[0])
    corr = (v2 * (Dm @ (rk * w))).reshape(B.shape)
    if not args.no_features:
        Tdiag = np.exp(th[1:])[gid]
        beta = Tdiag * (A.T @ w)
        for f, (rows, c, s, j, norm) in enumerate(meta):
            if beta[f] != 0.0:
                corr[rows, c] += beta[f] * svars[s][rows, c] * Zc[rows, j] / norm
    # how much of the correction the probes measure: least-squares projection onto the probe span
    cf = corr.ravel(); coef = np.linalg.lstsq(Dm, cf, rcond=None)[0]; inspan = Dm @ coef
    share = float(np.sum(inspan ** 2) / max(np.sum(cf ** 2), 1e-30))
    meas_gain = float(2 * coef @ y - rho * np.mean(cf ** 2))
    print(f"correction: E[c^2] {np.mean(cf ** 2):.5f}, {share:.1%} inside the probe span; measured public gain at scale 1 "
          f"(in-span part, rho {rho:.3f}) {2 * coef @ y - rho * np.mean(inspan ** 2):.5f}; prior-implied total {rho * np.mean(cf ** 2):.5f}")
    for b in ("young", "mid", "old"):
        for c in (0, 1):
            m = band == b
            print(f"  {b:5s} {'h' if c == 0 else 'a'}: rms correction {np.sqrt(np.mean(corr[m, c] ** 2)):.4f}")
    if args.out:
        P = np.clip(B + args.scale * corr, 0, None)
        if args.hcap > 0:
            P[:, 0] = np.minimum(P[:, 0], args.hcap)
        shift.write_predictions(P, ROOT / args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
