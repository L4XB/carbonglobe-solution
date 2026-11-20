#!/usr/bin/env python
"""Damp a prediction file's forty-year change per seed-age band and write the candidate.

    corrected = y0 + lambda_band * (pred - y0)

``y0`` is decoded persistence in scaled units under the same decode the model used
(height 1.14114; agb multiplier on the command line), so ``pred - y0`` is exactly the
model's predicted change. Bands not named keep ``lambda = 1``. The output differs
from the input in nothing else, which is what makes a board pair of the two readable
(``scripts/shift_solve.py``).

Also printed: ``E_band[D^2]`` of the change being damped, and the precision a pair of
scores (input, output) would give on the band's optimal lambda -- so the lead can
see whether the slot buys a measurement before spending it.

    ./.venv/bin/python scripts/shift_apply.py --pred results/latent_v1.csv \
        --lam old=0.5 --out results/latent_v1_lam_old05.csv
    ./.venv/bin/python scripts/shift_apply.py --pred results/latent_v1.csv \
        --lam old=0.5:1.0 --lam mid=0.8 --out ...        # height:agb per band
    ./.venv/bin/python scripts/shift_apply.py --pred results/latent_v1.csv \
        --scale old=0.85:1.0 --out ...                   # rescale the old band's height level

Two families, one per call, because a board pair reads one line. ``--lam`` damps
the predicted change and cannot reverse its sign; ``--scale`` rescales the level
and can, which is what the board's height slope of 0.960 (old stands losing height
where every present-day model grows them) calls for.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import shift  # noqa: E402


def parse_lam(items: list[str]) -> dict[str, float | tuple[float, float]]:
    out: dict[str, float | tuple[float, float]] = {}
    for item in items:
        band, _, value = item.partition("=")
        if band not in shift.BANDS:
            raise SystemExit(f"unknown band {band!r}; bands are {list(shift.BANDS)}")
        if ":" in value:
            h, a = value.split(":")
            out[band] = (float(h), float(a))
        else:
            out[band] = float(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", type=Path, required=True, help="prediction file in template order")
    ap.add_argument("--lam", action="append", default=[], help="band=lambda or band=lambda_h:lambda_a (damp the change)")
    ap.add_argument("--scale", action="append", default=[], help="band=c or band=c_h:c_a (rescale the level)")
    ap.add_argument("--m-agb", type=float, default=shift.M_HEIGHT, help="agb decode multiplier the model used")
    ap.add_argument("--base", type=Path, default=None,
                    help="the model's own year-0 reference file (persistence with the model's rules, e.g. "
                         "results/latent_ref_y0_m114114.csv); default: decoded persistence without rules. "
                         "With the model's own base, pred - base is exactly its predicted change and a rule "
                         "like bare-stays-bare is not mistaken for a change to be damped")
    ap.add_argument("--rounding", type=float, default=5e-4)
    ap.add_argument("--board-slopes", type=str, default="0.9602,0.9334",
                    help="the board's whole-population persistence slopes, height,agb, on the decoded columns "
                         "(scripts/shift_board_arithmetic.py; the agb value assumes the 1.14114 decode)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    board_h, board_a = (float(v) for v in args.board_slopes.split(","))
    lam = parse_lam(args.lam)
    factor = parse_lam(args.scale)
    if bool(lam) == bool(factor):
        raise SystemExit("give exactly one of --lam (damp the change) or --scale (rescale the level), "
                         "so the output differs from the input along one line")

    keys = shift.template_keys()
    ages = keys["age"].to_numpy()
    pred = shift.read_predictions(args.pred)
    base = shift.read_predictions(args.base) if args.base is not None else shift.persistence(m_agb=args.m_agb)
    out = shift.damp(pred, base, lam, ages) if lam else shift.scale(pred, factor, ages)

    d = pred - base
    v = out - pred
    masks = shift.band_masks(ages)
    print(f"input  {args.pred}")
    print(f"base   {args.base if args.base is not None else f'decoded persistence, height {shift.M_HEIGHT} agb {args.m_agb}'}")
    print(f"lambda {lam}" if lam else f"level factor {factor}")
    print(f"{'band':6s} {'f_band':>7s} {'E_band[D_h^2]':>14s} {'E_band[D_a^2]':>14s} {'mean D_h':>9s} {'mean D_a':>9s} "
          f"{'s*_model h':>11s} {'s*_model a':>11s}")
    for band, m in masks.items():
        print(f"{band:6s} {m.mean():7.4f} {np.mean(d[m, 0] ** 2):14.6f} {np.mean(d[m, 1] ** 2):14.6f} "
              f"{d[m, 0].mean():+9.4f} {d[m, 1].mean():+9.4f} "
              f"{shift.slope(base[m, 0], pred[m, 0]):11.4f} {shift.slope(base[m, 1], pred[m, 1]):11.4f}")
    s_h, s_a = shift.slope(base[:, 0], pred[:, 0]), shift.slope(base[:, 1], pred[:, 1])
    print(f"\nwhole population: s*_model height {s_h:.4f} agb {s_a:.4f} against the board's {board_h:.4f} / {board_a:.4f}")
    try:
        lam_h = shift.moment_matched_lambda(board_h, s_h)
        lam_a = shift.moment_matched_lambda(board_a, s_a)
        print(f"  moment-matched lambda (X-projection, dominated by the tall old stands): height {lam_h:.3f}  agb {lam_a:.3f}")
        print("  this needs no slot; it is exact for the projection on y0 and only an estimate of the MSE optimum")
    except ZeroDivisionError as exc:
        print(f"  moment matching undefined: {exc}")
    negative = int((out < -1e-9).sum())
    if negative:
        print(f"WARNING {negative} cells went negative (a factor outside [0, 1]); clipping them to 0 breaks exact linearity")
        out = np.clip(out, 0.0, None)
    e_v2 = float(np.mean(v**2))
    print(f"\nE[V^2] of the change this file makes (pooled over all cells): {e_v2:.6f}")
    if e_v2 > 0:
        print(f"a board pair (input, output) resolves the optimal step to +/-{args.rounding / e_v2:.4f} "
              f"(in units where the output is step 1) at {args.rounding} rounding")
    report = shift.write_predictions(out, args.out)
    print(report.render())
    print(f"wrote {args.out}; nothing submitted")


if __name__ == "__main__":
    main()
