#!/usr/bin/env python
"""Read the optimal step between two submitted files off their two board scores.

Any two files ``a`` and ``b`` lie on a line ``a + t (b - a)``; the score along it is
an exact parabola whose curvature ``E[(b - a)^2]`` we compute ourselves, so the two
board scores fix the optimal ``t`` and the score there (``cgb.shift.line_search``).
With ``--out`` the file at the optimum is written as the next candidate.

If ``b`` was made from ``a`` by ``scripts/shift_apply.py`` with ``lambda_d`` on one
band, then ``lambda* = 1 + t* (lambda_d - 1)`` for that band. If ``b`` is a per-band
persistence probe against its reference ``a``, ``--band`` prints the ``B_band`` /
``s*_band`` reading of ``notes/board-anchors.md`` as well.

    ./.venv/bin/python scripts/shift_solve.py --a results/latent_v1.csv --score-a 0.061 \
        --b results/latent_v1_lam_old05.csv --score-b 0.055 --lam-d 0.5 \
        --out results/latent_v1_lam_old_star.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import shift  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--score-a", type=float, required=True)
    ap.add_argument("--b", type=Path, required=True, action="append",
                    help="probe file(s); give --b/--score-b once per file to solve the k-parameter family jointly")
    ap.add_argument("--score-b", type=float, required=True, action="append")
    ap.add_argument("--rounding", type=float, default=5e-4, help="half the board's last shown digit")
    ap.add_argument("--lam-d", type=float, default=None, help="the lambda used to make b from a, if any")
    ap.add_argument("--band", action="store_true", help="also read b as a per-band persistence probe of a")
    ap.add_argument("--out", type=Path, default=None, help="write the file at the optimal step")
    args = ap.parse_args()

    if len(args.b) != len(args.score_b):
        raise SystemExit("give one --score-b per --b")
    a = shift.read_predictions(args.a)
    bs = [shift.read_predictions(p) for p in args.b]

    if len(bs) == 1:
        b, score_b = bs[0], args.score_b[0]
        ls = shift.line_search(a, b, args.score_a, score_b, rounding=args.rounding)
        print(f"line from {args.a.name} (t=0) to {args.b[0].name} (t=1)")
        print(ls.render())
        if args.lam_d is not None:
            lam_star = 1.0 + ls.t_star * (args.lam_d - 1.0)
            lam_err = ls.t_err * abs(args.lam_d - 1.0)
            print(f"  lambda* on the damped band = {lam_star:.4f} (+/- {lam_err:.4f})")
        if args.band:
            keys = shift.template_keys()
            bp = shift.band_probe(a, b, args.score_a, score_b, keys["age"].to_numpy(), rounding=args.rounding)
            print(bp.render())
        best = a + ls.t_star * (b - a)
        t_note = f"t = {ls.t_star:.4f}"
    else:
        qs = shift.quadratic_search(a, bs, args.score_a, args.score_b, rounding=args.rounding)
        print(f"quadratic on the span of {len(bs)} probes around {args.a.name}; t_i = 1 is probe i")
        print(qs.render([p.name for p in args.b]))
        best = a + sum(t * (b - a) for t, b in zip(qs.t_star, bs))
        t_note = "t = " + ", ".join(f"{t:.4f}" for t in qs.t_star)

    if args.out is not None:
        negative = int((best < -1e-9).sum())
        if negative:
            print(f"WARNING {negative} cells negative at the optimum; clipped to 0")
        report = shift.write_predictions(np.clip(best, 0.0, None), args.out)
        print(report.render())
        print(f"wrote {args.out} at {t_note}; nothing submitted")


if __name__ == "__main__":
    main()
