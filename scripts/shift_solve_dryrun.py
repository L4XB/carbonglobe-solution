#!/usr/bin/env python
"""Before the slots are spent: does the board's rounding swallow the joint solve's gain?

Tomorrow's plan puts the model and three probes on the board and then submits the
candidate at the joint optimum of ``(c_old,h, c_old,a, lambda_old,a)``
(``scripts/shift_solve.py`` with three ``--b``). Four scores enter, each rounded to
three decimals, and ``cgb.shift.quadratic_search`` bounds the solution over all
sixteen rounding corners. The half-widths on the parameters depend only on the
probe files (``delta t = -H^-1 delta g``), so they are known now; the half-width on
the score at the optimum depends on how far the optimum sits from the model, so it
is shown for a few hypothetical optima, each converted to the four scores the
board would display and rounded the way the board rounds.

The question answered per scenario: is the candidate at the solved optimum better
than the model even at the worst rounding corner, and by how much more than the
best single-direction file would be.

    ./.venv/bin/python scripts/shift_solve_dryrun.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import io, shift  # noqa: E402

RESULTS = io.ROOT / "results"
REFERENCE = "latent_v1x2_m114.csv"
PROBES = {  # name -> (step on its parameter, parameter name)
    "shift_latent_v1x2_scale_old150_h085.csv": (0.15, "c_h"),
    "shift_latent_v1x2_scale_old150_a090.csv": (0.10, "c_a"),
    "shift_latent_v1x2_lam_old150_a05.csv": (0.5, "lambda_a"),
}
ROUNDING = 5e-4
SCENARIOS = [
    ("A expected", 0.070, (0.90, 1.00, 0.60)),
    ("B large shift", 0.070, (0.85, 1.05, 0.50)),
    ("C small shift", 0.070, (0.95, 1.00, 0.80)),
    ("D model already good", 0.040, (0.93, 1.00, 0.70)),
    ("E nothing to fix", 0.040, (1.00, 1.00, 1.00)),
]


def main() -> None:
    p = shift.read_predictions(RESULTS / REFERENCE)
    names = list(PROBES)
    bs = [shift.read_predictions(RESULTS / n) for n in names]
    steps = np.array([PROBES[n][0] for n in names])
    labels = [PROBES[n][1] for n in names]
    v = [b - p for b in bs]
    h = np.array([[float(np.mean(vi * vj)) for vj in v] for vi in v])
    h_inv = np.linalg.inv(h)
    print("H = E[V_i V_j] from the files:")
    print(np.array2string(h, precision=6, suppress_small=True))
    t_half = np.abs(h_inv) @ np.full(len(names), ROUNDING)
    print("worst-case half-width, score-independent: " + "  ".join(f"{l} {t * s:.4f}" for l, t, s in zip(labels, t_half, steps)))

    for label, s_model, optimum in SCENARIOS:
        t_true = (1.0 - np.array(optimum)) / steps
        g = -h @ t_true
        exact = [s_model + 2.0 * g[i] + h[i, i] for i in range(len(names))]
        shown_a, shown_b = round(s_model, 3), [round(x, 3) for x in exact]
        qs = shift.quadratic_search(p, bs, shown_a, shown_b, rounding=ROUNDING)
        solved = 1.0 - steps * qs.t_star
        lo = 1.0 - steps * qs.t_hi
        hi = 1.0 - steps * qs.t_lo
        guaranteed = qs.score_hi < shown_a - ROUNDING
        best = min(
            (shift.line_search(p, b, shown_a, sb, rounding=ROUNDING) for b, sb in zip(bs, shown_b)),
            key=lambda ls: ls.score_star,
        )
        print(f"\n{label}: model {s_model:.3f}, true optimum {optimum}, true gain {float(g @ h_inv @ g):.4f}")
        print(f"  board shows {shown_a:.3f} / " + " / ".join(f"{x:.3f}" for x in shown_b))
        print("  solved " + "  ".join(f"{l} {s:.4f} [{a:.4f}, {b:.4f}]" for l, s, a, b in zip(labels, solved, lo, hi)))
        print(f"  score at optimum {qs.score_star:.4f} [{qs.score_lo:.4f}, {qs.score_hi:.4f}] vs model {shown_a:.3f}; "
              f"better at the worst corner: {guaranteed}; best single line {best.score_star:.4f} +/- {best.score_err:.4f}")


if __name__ == "__main__":
    main()
