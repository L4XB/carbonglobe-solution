#!/usr/bin/env python
"""Read out the board's verdict on blend variants P = B + D against their base B (both scored).

s(P) - s(B) = E_pub[D^2] - 2 E_pub[(T - B) D], so the truth's coefficient on D, lambda = E[(T - B) D] / E[D^2], is
(E[D^2] - ds) / (2 E[D^2]); lambda = 1 means the step was exactly right in that direction, 0 that it was noise, and the
best step is lambda x D (gain lambda^2 E[D^2]). Two roundings make ds uncertain by +-0.001.

    ./.venv/bin/python scripts/read_variants.py results/fsolve_rho_r2e4_20261005.csv results/wshrink_blend_20261005.csv ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402


def main() -> None:
    sc = json.load(open(ROOT / "results" / "board_scores.json"))
    C = sc["zeros.csv"]; s1, s05 = sc["persistence_shifted.csv"], sc["persistence_s05.csv"]
    ps = shift.read_predictions(ROOT / "results" / "persistence_shifted.csv").astype(np.float64)
    rho = 2 * (s1 - 2 * s05 + C) / float(np.mean(ps ** 2))
    base = sys.argv[1]
    B = shift.read_predictions(ROOT / base).astype(np.float64)
    sB = sc[Path(base).name]
    for f in sys.argv[2:]:
        name = Path(f).name
        if name not in sc:
            print(f"{name}: not scored yet"); continue
        D = shift.read_predictions(ROOT / f).astype(np.float64) - B
        e2 = rho * float(np.mean(D ** 2))
        ds = sc[name] - sB
        lam = (e2 - ds) / (2 * e2)
        dl = 0.001 / (2 * e2)
        print(f"{name:40s} score {sc[name]:.3f} vs base {sB:.3f}: E[D^2] {e2:.5f}  lambda {lam:+.2f} (+-{dl:.2f})  "
              f"best step gain {max(lam, 0) ** 2 * e2:.5f} at {lam:+.2f} x D")


if __name__ == "__main__":
    main()
