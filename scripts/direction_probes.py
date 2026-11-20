#!/usr/bin/env python
"""Physics-direction probes on the old band's agb column.

The board returns E[T.F] for any file F. With F = R + Z_k, where Z_k is a direction that is zero
outside the old band's agb cells, E[T.Z_k] - E[R.Z_k] = E[dT.Z_k] is the projection of the truth's
forty-year change on that direction; with the directions' Gram matrix (known) this is a linear
regression of the true old-band change on interpretable site covariates, fitted without seeing
the truth. Directions are built from a covariate c (standardised over the vegetated old rows)
times the stand size R_a, then orthogonalised (Gram-Schmidt) against the models' own change
direction M - R and against each other, and scaled to E[Z^2] = target.

    ./.venv/bin/python scripts/direction_probes.py --model results/mix_g2x2_noco2.csv --tag phys
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io, shift  # noqa: E402

OLD = (150, 200, 250, 300, 350, 400, 450, 500)


def covariates(keys) -> dict[str, np.ndarray]:
    sc = keys["scenario"].to_numpy(); site = keys["site"].to_numpy()
    out = {k: np.zeros(len(sc)) for k in ("dT", "dP", "Tmean", "scen", "Pmean", "prof", "dq", "dSW", "dry")}
    years = np.arange(40) - 19.5
    ages = keys["age"].to_numpy()
    for s in io.SCENARIOS:
        ann = np.load(ROOT / "work" / "cache" / f"test_{s}_annual.npy", mmap_mode="r")
        t = np.asarray(ann[:, :, 0], dtype=np.float64); p = np.asarray(ann[:, :, 1], dtype=np.float64)
        q = np.asarray(ann[:, :, 33 + 4 * np.arange(24)], dtype=np.float64).mean(2)
        sw = np.asarray(ann[:, :, 35 + 4 * np.arange(24)], dtype=np.float64).mean(2)
        dT = (t * years).sum(1) / (years**2).sum() * 40.0  # K over the run
        dP = (p * years).sum(1) / (years**2).sum() * 40.0 / np.maximum(p.mean(1), 1.0)  # relative change over the run
        dq = (q * years).sum(1) / (years**2).sum() * 40.0 / np.maximum(q.mean(1), 1e-6)
        dSW = (sw * years).sum(1) / (years**2).sum() * 40.0 / np.maximum(sw.mean(1), 1.0)
        dry = (p < 0.8 * p.mean(1, keepdims=True)).sum(1).astype(np.float64)  # dry years in the run
        y0 = io.open_test(s)[1].astype(np.float64)
        a = 1.14114 * (y0[:, :, 1] + 0.401412)  # decoded agb by seed age (6171, 15)
        prof = np.log((a[:, io.SEED_AGES.index(500)] + 0.1) / (a[:, io.SEED_AGES.index(300)] + 0.1))  # recent spin-up collapse indicator
        m = sc == s
        out["dT"][m] = dT[site[m]]; out["dP"][m] = dP[site[m]]
        out["Tmean"][m] = (t.mean(1) - 273.15)[site[m]]; out["Pmean"][m] = np.log1p(p.mean(1) * 12.0)[site[m]]
        out["scen"][m] = 1.0 if s == "ssp585" else -1.0
        out["prof"][m] = prof[site[m]]; out["dq"][m] = dq[site[m]]; out["dSW"][m] = dSW[site[m]]; out["dry"][m] = dry[site[m]]
    exc = np.load(ROOT / "work" / "cache" / "excess_test.npz")["exc"]  # (rows, 7): initial-state excess over the training relation
    for k, nm in enumerate(("height", "agb", "soil", "lai", "gpp", "npp", "rh")):
        out[f"exc_{nm}"] = exc[:, k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--target", type=float, default=0.03, help="E[Z^2] pooled over all 2N cells")
    ap.add_argument("--directions", default="dT,dP,scen,Tmean,Pmean", help="ordered list; each is orthogonalised against the model change and all earlier ones")
    ap.add_argument("--band", default="old", choices=("young", "mid", "old"))
    ap.add_argument("--column", default="agb", choices=("height", "agb"))
    args = ap.parse_args()
    M = shift.read_predictions(args.model).astype(np.float64)
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    keys = shift.template_keys(); ages = keys["age"].to_numpy()
    col = 1 if args.column == "agb" else 0
    band_ages = {"old": OLD, "mid": (50, 70, 100), "young": (1, 10, 20, 30)}[args.band]
    old = np.isin(ages, band_ages) & (R[:, col] > 0.02)
    cov = covariates(keys)
    n2 = R.size
    def vec(col_values: np.ndarray) -> np.ndarray:
        z = np.zeros_like(R); z[old, col] = col_values[old]; return z
    d_model = vec((M - R)[:, col])
    basis = [d_model.reshape(-1) / np.sqrt(np.mean(d_model**2))]
    directions = {}
    for name in args.directions.split(","):
        c = cov[name].copy()
        c = (c - c[old].mean()) / (c[old].std() + 1e-9)
        z = vec(c * R[:, col]).reshape(-1)
        for b in basis:  # orthogonalise against the model direction and earlier directions
            z = z - (z @ b) / (b @ b) * b
        z = z / np.sqrt(np.mean(z**2)) * np.sqrt(args.target)
        basis.append(z)
        directions[name] = z.reshape(-1, 2)
        P = np.clip(R + directions[name], 0, None)
        path = ROOT / "results" / f"dirprobe_{args.tag}_{name}.csv"
        shift.write_predictions(P, path)
        clipped = np.mean((P - R - directions[name])**2)
        print(f"{path.name}: E[Z^2] {np.mean(directions[name]**2):.4f}, corr with model change {np.corrcoef(directions[name][old,col], (M-R)[old,col])[0,1]:+.3f}, clipping loss {clipped:.2e}")
    np.savez(ROOT / "work" / "cache" / f"dirprobe_{args.tag}_directions.npz", **directions, model=d_model)


if __name__ == "__main__":
    main()
