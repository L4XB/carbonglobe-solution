#!/usr/bin/env python
"""Realization-specific weather effects of the latent models on the test.

Each latent group was rolled out with its actual forcing and with K bootstrap weather realizations
(`predict_latent.py --shuffle-years k --bootstrap`: per site, residual years around the linear trend drawn with
replacement). M = mean over the realizations is the model's expected outcome for the site's climate; W = actual - M is
the part of the prediction that belongs to this particular weather sequence. The two scenarios have independent weather,
so the truth's realization-specific part is independent between them while the y0 and the climate are nearly shared.

    ./.venv/bin/python scripts/weather_realizations.py [--tag boot]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

GROUPS = {"latelat2": "results/latent_latelat2_actual.csv", "ss20L3": "results/latent_ss20L3_actual.csv",
          "noco2": "work/shuffle/noco2_actual.csv", "v1x2": "work/shuffle/v1x2_actual.csv"}


def contrast(F: np.ndarray, o126: np.ndarray, o585: np.ndarray) -> np.ndarray:
    D = np.zeros_like(F); d = (F[o585] - F[o126]) / 2
    D[o585] = d; D[o126] = -d
    return D


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="boot")
    ap.add_argument("--blend", default="results/fsolve_rho_r2e4_20261005.csv")
    args = ap.parse_args()
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    B = shift.read_predictions(ROOT / args.blend).astype(np.float64)
    keys = shift.template_keys(); masks = shift.band_masks(keys["age"].to_numpy())
    o126, o585 = np.load(ROOT / "work" / "contrast_index.npy")
    N2 = 2 * len(R)
    W = {}
    for g, act in GROUPS.items():
        files = sorted((ROOT / "work" / "shuffle").glob(f"{g}_{args.tag}*.csv"))
        if not files or not (ROOT / act).exists():
            print(f"{g}: missing"); continue
        S = np.stack([shift.read_predictions(f).astype(np.float64) for f in files])
        A = shift.read_predictions(ROOT / act).astype(np.float64)
        Mn = S.mean(0); sd = S.std(0)
        W[g] = A - Mn
        np.save(ROOT / "work" / "shuffle" / f"{g}_{args.tag}_mean.npy", Mn)
        print(f"{g}: K={len(files)}  E[(A-R)^2] {np.sum((A-R)**2)/N2:.4f}  E[W^2] {np.sum(W[g]**2)/N2:.4f}  "
              f"E[sd_k^2] {np.sum(sd**2)/N2:.4f}  mean W h {W[g][:,0].mean():+.4f} a {W[g][:,1].mean():+.4f}")
        for b in ("young", "mid", "old"):
            m = masks[b]
            print(f"    {b:5s} W energy h {np.sum(W[g][m,0]**2)/N2:.5f} a {np.sum(W[g][m,1]**2)/N2:.5f} | "
                  f"realization sd^2 h {np.sum(sd[m,0]**2)/N2:.5f} a {np.sum(sd[m,1]**2)/N2:.5f}")
        cA = contrast(A - R, o126, o585); cM = contrast(Mn - R, o126, o585)
        print(f"    contrast energy: actual {np.sum(cA**2)/N2:.5f}, expected-weather {np.sum(cM**2)/N2:.5f}")
    # the blend's loading on the realization-specific parts
    names = list(W)
    X = np.stack([W[g].ravel() for g in names], 1)
    y = (B - R).ravel()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = (X @ beta).reshape(R.shape)
    print("blend deviation regressed on the W's:", dict(zip(names, np.round(beta, 3))), f"fitted energy {np.sum(fit**2)/N2:.5f}")
    Bc = contrast(B, o126, o585)
    for g in names:
        cW = contrast(W[g], o126, o585)
        print(f"  corr(blend contrast, W_{g} contrast) {np.sum(Bc*cW)/np.sqrt(np.sum(Bc**2)*np.sum(cW**2)):+.3f}")
    np.save(ROOT / "work" / "shuffle" / f"blend_W_fit_{args.tag}.npy", fit)


if __name__ == "__main__":
    main()
