#!/usr/bin/env python
"""Three probe files around a scored model, for a joint solve on the board.

The model scored 0.128 against a predicted 0.025, so its forty-year change is at best
partly right. Each file here moves one direction of the family
``height = R_h + l_h (L_h - R_h)``, ``agb = R_a + l_a (L_a - R_a)``, times a common
level ``c``, where ``L`` is the model and ``R`` its own decoded initial state. The
family is linear in prediction space, so the three board scores plus the model's
give the exact optimum over ``(l_h, l_a, c)`` via ``cgb.shift.quadratic_search``
(``scripts/shift_solve.py``, one ``--b`` per file).

    ./.venv/bin/python scripts/column_probes.py --model results/latent_v1x2_m114.csv \
        --ref results/latent_ref_y0_m114114.csv --tag latent_v1x2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgb import shift  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True, help="the model's decoded initial state")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lam", type=float, default=0.5, help="change multiplier for the two column probes")
    ap.add_argument("--level", type=float, default=0.9, help="level multiplier for the common-level probe")
    args = ap.parse_args()

    model = shift.read_predictions(args.model)  # (N, 2): height, agb
    ref = shift.read_predictions(args.ref)

    probes: dict[str, np.ndarray] = {}
    for col, name in ((0, "h"), (1, "a")):
        pred = model.copy()
        pred[:, col] = ref[:, col] + args.lam * (model[:, col] - ref[:, col])
        probes[f"colprobe_{args.tag}_lam{name}{int(round(args.lam * 100)):03d}"] = pred
    probes[f"colprobe_{args.tag}_level{int(round(args.level * 100)):03d}"] = model * args.level

    for stem, pred in probes.items():
        path = ROOT / "results" / f"{stem}.csv"
        shift.write_predictions(pred, path)
        moved = np.abs(pred - model).max(axis=1) > 0
        print(f"{path.name}: {int(moved.sum())} rows moved, E[(p - model)^2] = "
              f"{float(np.mean((pred - model) ** 2)):.6f}")


if __name__ == "__main__":
    main()
