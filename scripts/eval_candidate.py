#!/usr/bin/env python
"""Predicted public score of an arbitrary candidate file from the board functionals.

The candidate P is split into its projection on the span of the scored files (whose truth functionals are known
from the scores, rho-scaled Gram as in file_solve.py) and the remainder; the remainder's truth functional is
unknown and is reported as a bound: score = C - 2 E[T.P_span] - 2 E[T.P_res] + E[P^2], with E[T.P_res] set to 0
(the remainder uncorrelated with the truth) and to +-sqrt(E[P_res^2] * resid) for a residual energy resid.

    ./.venv/bin/python scripts/eval_candidate.py results/x.csv [results/y.csv ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

EXCLUDE = ("persistence_s05", "colprobe_latent_v1x2_level090", "persistence_raw", "zeros")


def main() -> None:
    sc = json.load(open(ROOT / "results" / "board_scores.json"))
    names = [n for n in sc if (ROOT / "results" / n).exists() and not any(e in n for e in EXCLUDE)]
    F = np.stack([shift.read_predictions(ROOT / "results" / n).astype(np.float64).reshape(-1) for n in names], axis=1)
    s = np.array([sc[n] for n in names])
    C = sc["zeros.csv"]
    ps = shift.read_predictions(ROOT / "results" / "persistence_shifted.csv").astype(np.float64)
    rho = 2 * (sc["persistence_shifted.csv"] - 2 * sc["persistence_s05.csv"] + C) / float(np.mean(ps ** 2))
    G = rho * (F.T @ F / F.shape[0])
    b = (np.diag(G) + C - s) / 2          # E_pub[T F_k]
    for path in sys.argv[1:]:
        P = shift.read_predictions(path).astype(np.float64).reshape(-1)
        coef, *_ = np.linalg.lstsq(F, P, rcond=None)
        Pspan = F @ coef; Pres = P - Pspan
        e_tp_span = float(coef @ b)
        e_p2 = rho * float(np.mean(P ** 2))
        e_res2 = rho * float(np.mean(Pres ** 2))
        point = C - 2 * e_tp_span + e_p2
        print(f"{Path(path).name:42s} predicted public score {point:.4f} (out-of-span energy {e_res2:.4f}; truth on the remainder unknown, "
              f"range +-{2 * np.sqrt(e_res2 * 0.05):.4f} for a 0.05 residual)")


if __name__ == "__main__":
    main()
