#!/usr/bin/env python
"""Read a batch of contrast files against their base: kappa of every contrast from the score difference, with the 3-decimal rounding of the board.

A contrast file F = base + D (D on its own cells). If the truth is base + kappa D_full on those cells and the file applied the step s, the gain of F
against the base is (2 kappa s - s^2) E_full. Here every file is treated as ONE direction at step 1 with E = sum (F - base)^2 (mixed steps inside a file
are part of its direction): gain = (2 kappa - 1) E, kappa = (gain / E + 1) / 2. Scores are rounded to 0.001 = 370.26 squared units: the true gain lies
within +-1 quantum of the observed difference (triangular), so the kappa interval is printed for gain -1 / observed / +1 quantum.

    python scripts/readout_kappa.py results/cand48_A.csv results/cand48_CY.csv results/cand48_CM.csv results/cand48_B1.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

Q = 370.26
sc = json.load(open(ROOT / "results/board_scores.json"))
base = Path(sys.argv[1]); B = shift.read_predictions(ROOT / base).astype(np.float64); sb = sc.get(base.name)
print(f"base {base.name}: score {sb}")
for f in sys.argv[2:]:
    f = Path(f); s = sc.get(f.name)
    D = shift.read_predictions(ROOT / f).astype(np.float64) - B; E = float(np.sum(D ** 2))
    if s is None or sb is None:
        print(f"   {f.name:20s}: no score yet; energy {E:.0f} squared units = {E / Q:.2f} quanta"); continue
    g = (sb - s) * 1000.0                     # gain in quanta (positive = better than the base)
    ks = [((gg * Q) / E + 1.0) / 2.0 for gg in (g - 1.0, g - 0.5, g, g + 0.5, g + 1.0)]
    print(f"   {f.name:20s}: score {s} -> gain {g:+.0f} quanta; energy {E:.0f} ({E / Q:.2f} quanta); kappa {ks[2]:+.2f}  (68 % range {ks[1]:+.2f} .. {ks[3]:+.2f}, hard limits {ks[0]:+.2f} .. {ks[4]:+.2f})")
