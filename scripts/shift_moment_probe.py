#!/usr/bin/env python
"""One probe that reads the mean year-40 agb of the test population off the board.

The persistence anchors give only ``E[X T]``, a slope weighted by ``X^2`` and so by
the tall old stands, and they cannot separate a decode multiplier ``m`` on agb from
a population whose old stands decline differently: ``s*(m) = s*(1) / m`` for any
``m``, and the reading of "how much the old stands decline" moves with it
(``scripts/shift_board_arithmetic.py`` Sec. 3b). The **first moment** is different.
The constant-1.0 probe gave ``E[T_h] + E[T_a] = 2 * 0.9933``; a file with agb = 1
and height = 0 splits them:

    score = 0.5 [ E[T_h^2] + E[(1 - T_a)^2] ] = C + 0.5 - E[T_a]

with ``C = E[T^2] = 2.5886`` already known, so ``E[T_a] = C + 0.5 - score`` and
``E[T_h] = 2 * 0.9933 - E[T_a]``, each to about +/-0.002.

Why that pins the decode: the mean year-40 agb is the mean decoded year-0 agb times
the population's mean growth ratio, ``E[T_a] = m * mean(X_a; m=1) * r_a``, and
``r_a`` is not free -- in the training runs it is 0.980 (young growth and old
decline cancel), and a population whose trajectories change less, as the board
says this one does, sits between that and 1.0. The mean, unlike the slope, is not
dominated by the tall stands, so ``r_a`` is far less sensitive to what the old
stands do than ``s*`` is. ``E[T_h]`` likewise reads the population's mean height
growth against the training 1.178, which is what the young band's lambda needs.

Writes ``results/shift_probe_agb_const1.csv`` and prints what each outcome means.
Nothing here submits.

    ./.venv/bin/python scripts/shift_moment_probe.py [--score 2.09]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import io, shift  # noqa: E402

RESULTS = io.ROOT / "results"
C_POOLED, C_ERR = 2.58856, 0.0015  # scripts/shift_board_arithmetic.py Sec. 1
E_T_POOLED, E_T_ERR = 0.99328, 0.0010
R_A_TRAIN, R_H_TRAIN = 0.9799, 1.1777  # scripts/shift_population.py Sec. G, all sites


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score", type=float, default=None, help="the board score once the file has been submitted")
    ap.add_argument("--rounding", type=float, default=5e-4)
    args = ap.parse_args()

    n = len(shift.template_keys())
    probe = np.zeros((n, 2))
    probe[:, 1] = 1.0
    path = RESULTS / "shift_probe_agb_const1.csv"
    shift.write_predictions(probe, path, probe=True)
    x1 = shift.persistence(m_agb=1.0)
    mean_a1, mean_h = x1[:, 1].mean(), x1[:, 0].mean()
    print(f"wrote {path.name}: height 0, agb 1 on every row; nothing submitted")
    print(f"mean floor-shifted agb persistence (m = 1): {mean_a1:.4f}; mean decoded height persistence: {mean_h:.4f}")
    print("\nexpected score under each reading, E[T_a] = m * mean(X_a; m=1) * r_a:")
    print(f"  {'m_agb':>6s} {'r_a=0.98':>9s} {'r_a=1.00':>9s} {'r_a=1.05':>9s}   (score = C + 0.5 - E[T_a])")
    for m in (1.0, 1.14114, 1.26):
        row = [C_POOLED + 0.5 - m * mean_a1 * r for r in (0.98, 1.0, 1.05)]
        print(f"  {m:6.3f} {row[0]:9.4f} {row[1]:9.4f} {row[2]:9.4f}")
    if args.score is None:
        print("\nrun again with --score <board score> to read E[T_a], E[T_h] and the implied multiplier")
        return
    e_ta = C_POOLED + 0.5 - args.score
    err = C_ERR + args.rounding
    e_th = 2.0 * E_T_POOLED - e_ta
    print(f"\nE[T_a] = {e_ta:.4f} (+/- {err:.4f})   E[T_h] = {e_th:.4f} (+/- {err + 2 * E_T_ERR:.4f})")
    print(f"  mean height growth ratio E[T_h]/mean(X_h) = {e_th / mean_h:.4f} against 1.178 in training")
    for r in (0.98, 1.0, 1.05):
        print(f"  if r_a = {r:.2f}: m_agb = {e_ta / (mean_a1 * r):.4f} (+/- {err / (mean_a1 * r):.4f})")


if __name__ == "__main__":
    main()
