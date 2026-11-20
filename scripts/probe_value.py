#!/usr/bin/env python
"""Leave-one-out value of every scored file for the board blend.

Solves the sum-to-one, rho-scaled blend (``file_solve.py --rho --sum1``, fixed ridge) with all scored files and
again without each one; the difference of the point estimates is what that file contributes given all the
others. Families of probes that paid off are the ones to extend.

    ./.venv/bin/python scripts/probe_value.py --ridge 3e-4
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

EXCLUDE = ("persistence_s05", "colprobe_latent_v1x2_level090", "persistence_raw", "zeros", "fsolve_", "blend_gnr")


def solve(G: np.ndarray, b: np.ndarray, ridge: float, w0: np.ndarray) -> np.ndarray:
    A = G + ridge * np.eye(len(b))
    ai_r = np.linalg.solve(A, b + ridge * w0)
    ai_1 = np.linalg.solve(A, np.ones(len(b)))
    mu = (ai_r.sum() - 1.0) / ai_1.sum()
    return ai_r - mu * ai_1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ridge", type=float, default=3e-4)
    args = ap.parse_args()
    sc = json.load(open(ROOT / "results" / "board_scores.json"))
    names = [n for n in sc if (ROOT / "results" / n).exists() and not any(e in n for e in EXCLUDE)]
    F = np.stack([shift.read_predictions(ROOT / "results" / n).astype(np.float64).reshape(-1) for n in names], axis=1)
    s = np.array([sc[n] for n in names])
    C = sc["zeros.csv"]
    ps = shift.read_predictions(ROOT / "results" / "persistence_shifted.csv").astype(np.float64)
    rho = 2 * (sc["persistence_shifted.csv"] - 2 * sc["persistence_s05.csv"] + C) / float(np.mean(ps ** 2))
    G = rho * (F.T @ F / F.shape[0])
    b = (np.diag(G) + C - s) / 2
    best = int(np.argmin(s))

    def point(idx: list[int]) -> float:
        w0 = np.zeros(len(idx))
        if best in idx:
            w0[idx.index(best)] = 1.0
        else:
            w0[int(np.argmin(s[idx]))] = 1.0
        Gi, bi = G[np.ix_(idx, idx)], b[idx]
        w = solve(Gi, bi, args.ridge, w0)
        return float(w @ Gi @ w - 2 * w @ bi + C)

    full = list(range(len(names)))
    p_all = point(full)
    print(f"all {len(names)} files: point {p_all:.4f} (ridge {args.ridge:g})")
    rows = []
    for k in full:
        if k == best:
            continue
        idx = [i for i in full if i != k]
        rows.append((point(idx) - p_all, names[k]))
    for d, n in sorted(rows, reverse=True):
        print(f"  {d:+.4f}  {n}")
    groups = {"phys directions": "dirprobe_phys", "cell probes": "cellprobe_", "relax probes": "relaxprobe_", "young gbdt2 probes": "bandprobe_gbdt2_young",
              "old-agb model probes": ("probe_detector", "bandprobe_gw4latex2", "bandprobe_state3", "bandprobe_noco2_old", "probe_fillin"),
              "R-slope probes": "bandprobe_Rx080", "column probes": "colprobe_", "full model files": ("latent_", "direct_")}
    print("group leave-out values:")
    for g, pat in groups.items():
        pats = pat if isinstance(pat, tuple) else (pat,)
        idx = [i for i in full if not any(p in names[i] for p in pats) or i == best]
        print(f"  {point(idx) - p_all:+.4f}  {g} ({len(full) - len(idx)} files)")


if __name__ == "__main__":
    main()
