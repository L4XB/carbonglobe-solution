#!/usr/bin/env python
"""Local public-score estimate for any candidate built on the 10-10 stack, from every scored probe.

s(P) = s_B - 2 a.y + a' G a + E[r^2]: P - B is split into its least-squares projection a on the scored probe
directions D_k (readouts y_k = E_pub[(T - B) D_k], public Gram G with the measured mirror ratios) and a remainder r
whose truth alignment is unknown (taken as 0, so it only costs its energy). The raw estimate is optimistic for
cluster-concentrated corrections; `--calibrate` fits realised = a0 + a1 * predicted gain on every scored corrected
stack (spatcorr_*) and applies it.

    ./.venv/bin/python scripts/local_eval.py --calibrate results/x.csv [results/y.csv ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

S_B = 0.0499          # stack_20261010: convex-stack prediction (board 0.050)
SITE = shift.template_keys()["site"].to_numpy()
CLUSTER = np.load(ROOT / "work" / "cluster0_modes.npy")[:, 0] != 0


def readouts(rho_conc: float = 1.0, ratio_file: str = ""):
    sc = json.load(open(ROOT / "results" / "board_scores.json"))
    rd = lambda p: shift.read_predictions(p).astype(np.float64)
    B = rd(ROOT / "results" / "stack_20261010.csv"); R = rd(ROOT / "results" / "latent_ref_y0_m114114.csv")
    C = sc["zeros.csv"]; ps = rd(ROOT / "results" / "persistence_shifted.csv")
    rho = 2 * (sc["persistence_shifted.csv"] - 2 * sc["persistence_s05.csv"] + C) / float(np.mean(ps ** 2))
    centers = json.loads((ROOT / "work" / "spat_centers.json").read_text())
    names = sorted(n for n in sc if (n.startswith("spat_") or n.startswith("perp_spat_")) and not n.endswith("_mirror.csv"))
    s_R = 0.25171
    ratios = json.loads((ROOT / ratio_file).read_text()) if ratio_file else {}
    D, y, fac = [], [], []
    for n in names:
        c = centers.get(n, "R"); isR = c == "R" or c.endswith("latent_ref_y0_m114114.csv")
        Cp = R if isR else rd(ROOT / c)
        d = rd(ROOT / "results" / n) - Cp
        e_all = float(np.mean(d * d)); f = 1.0
        mir = n.replace(".csv", "_mirror.csv")
        if mir in sc and not isR:
            clip = rd(ROOT / "results" / mir) - (Cp - d)                      # >= 0 where the mirror was clipped at zero
            e_clip = float(np.mean((d - clip) ** 2))
            e_pub = (sc[n] + sc[mir] - 2 * sc[Path(c).name]) / 2; f = e_pub / (rho * (e_all + e_clip) / 2)
            xr = (sc[n] - sc[mir]) / 4 - f * rho * (e_all - e_clip) / 4
        elif isR:
            f = ratios[n]["r"] if n in ratios else 1.0
            xr = (sc[n] - s_R - rho * f * e_all) / 2
        else:
            # un-mirrored probe: public energy rho * f * E_all, f = rho_conc for probes whose energy sits mostly on the
            # collapse cluster (two mirror pairs measured 1.28 / 1.34 there)
            es = np.bincount(SITE, weights=(d * d).sum(axis=1), minlength=SITE.max() + 1)
            if n in ratios:
                f = ratios[n]["r"]
            elif rho_conc != 1.0 and float(np.sum(es[CLUSTER]) / max(np.sum(es), 1e-30)) > 0.5:
                f = rho_conc
            xr = (sc[n] - sc[Path(c).name] - rho * f * e_all) / 2
        yb = -xr - rho * float(np.mean((B - Cp) * d))
        D.append(d.ravel()); y.append(yb); fac.append(f)
    return B, rho, np.stack(D, 1), np.array(y), np.array(fac), sc


def terms(P, B, rho, Dm, y, fac):
    """(2 a.y, a'Ga + rho E[r^2]) of a candidate: the linear (readout) and quadratic (public energy) parts of its gain."""
    w = np.sqrt(fac)
    G = rho * (Dm * w).T @ (Dm * w) / Dm.shape[0]
    d = (P - B).ravel()
    a, *_ = np.linalg.lstsq(Dm, d, rcond=None)
    r = d - Dm @ a
    return 2 * a @ y, a @ G @ a + rho * float(np.mean(r * r)), float(np.mean(r * r)) / max(float(np.mean(d * d)), 1e-30)


def predict(P, B, rho, Dm, y, fac):
    lin, quad, rem = terms(P, B, rho, Dm, y, fac)
    return lin - quad, rem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--rho-conc", type=float, default=1.0,
                    help="public-energy factor for un-mirrored cluster-concentrated probes (readout and Gram)")
    ap.add_argument("--ratio-file", default="", help="predicted public-energy ratios of un-mirrored cluster probes (json)")
    ap.add_argument("--structural", action="store_true",
                    help="calibrate realised = alpha (2 a.y) - beta (a'Ga) on the scored stacks (separates readout optimism "
                         "from public-energy error, so it extrapolates across scales and constructions)")
    args = ap.parse_args()
    B, rho, Dm, y, fac, sc = readouts(args.rho_conc, args.ratio_file)
    rd = lambda p: shift.read_predictions(p).astype(np.float64)
    a0, a1 = 0.0, 1.0
    if args.calibrate:
        pts = []
        for n in sorted(sc):
            if n.startswith("spatcorr_stack_"):
                g, _ = predict(rd(ROOT / "results" / n), B, rho, Dm, y, fac)
                pts.append((g, S_B - sc[n], n))
        G_ = np.array([p[0] for p in pts]); Rz = np.array([p[1] for p in pts])
        a1, a0 = np.polyfit(G_, Rz, 1)
        for g, r, n in pts:
            print(f"  calib {n:38s} predicted gain {g:.5f}  realised {r:.4f} (+-0.0005)")
        print(f"  realised ~ {a0:+.5f} + {a1:.3f} x predicted")
    alpha, beta = 1.0, 1.0
    if args.structural:
        L, Q, Z = [], [], []
        for n in sorted(sc):
            if n.startswith("spatcorr_stack_"):
                lin, quad, _ = terms(rd(ROOT / "results" / n), B, rho, Dm, y, fac)
                L.append(lin); Q.append(quad); Z.append(S_B - sc[n])
        (alpha, beta), *_ = np.linalg.lstsq(np.column_stack([L, -np.array(Q)]), np.array(Z), rcond=None)
        res = np.array(Z) - (alpha * np.array(L) - beta * np.array(Q))
        print(f"  structural: realised ~ {alpha:.3f} x (2 a.y) - {beta:.3f} x (a'Ga); rms residual {np.sqrt(np.mean(res**2)):.5f} "
              f"over {len(Z)} stacks; optimal scale of a fixed correction ~ {alpha / beta:.2f} x its own optimum")
    for f in args.files:
        lin, quad, rem = terms(rd(f), B, rho, Dm, y, fac)
        g = lin - quad
        line = f"{Path(f).name:40s} predicted gain {g:.5f} (remainder {rem:.1%} of the change)  -> score {S_B - (a0 + a1 * g):.4f}"
        if args.structural:
            line += f" | structural {S_B - (alpha * lin - beta * quad):.4f}"
        print(line)


if __name__ == "__main__":
    main()
