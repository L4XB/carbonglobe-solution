#!/usr/bin/env python
"""Per-band, per-column persistence probes rebased on a reference that is on the board.

The sibling's ``persistence_mh1141_agb114_band_*_s05.csv`` scale both columns of one
band against ``persistence_mh1141_agb114.csv``, which is not on the board (its score
is predicted, not measured, by ``scripts/shift_board_arithmetic.py``). These files
instead take ``persistence_mh1141_agb100.csv`` -- measured at 0.247 -- as the
reference and scale **one column** of one band by ``s``, so a single slot returns
that band's ``B`` for that column alone:

    delta = f_cells [(s^2 - 1) A - 2 (s - 1) B],   A = E_band,col[X^2] (known)

where ``f_cells`` is the band-column's share of the ``2 N`` scored cells. The old
band's agb column is where 56% of the persistence loss sits and where the shift was
measured, so that is the file worth a slot first. Nothing here submits.

    ./.venv/bin/python scripts/shift_band_probes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import io, shift  # noqa: E402

RESULTS = io.ROOT / "results"
REFERENCE = RESULTS / "persistence_mh1141_agb100.csv"
S = 0.5


def main() -> None:
    keys = shift.template_keys()
    ages = keys["age"].to_numpy()
    ref = shift.read_predictions(REFERENCE)
    masks = shift.band_masks(ages)
    print(f"reference {REFERENCE.name} (board 0.247); scale s = {S}")
    print(f"{'file':44s} {'f_cells':>8s} {'A':>9s} {'s* err':>7s}  {'delta if s*=1':>13s} {'if s*=0.9':>10s}")
    for band in ("old", "mid", "young"):
        for k, col in enumerate(shift.COLUMNS):
            m = masks[band]
            probe = ref.copy()
            probe[m, k] *= S
            f = m.sum() / ref.size
            a = float(np.mean(ref[m, k] ** 2))
            err = 5e-4 / ((1 - S) * f * a)
            path = RESULTS / f"shift_band_{band}_{col}_s05.csv"
            shift.write_predictions(probe, path)
            bp = shift.band_probe(ref, probe, 0.247, 0.247 + f * ((S * S - 1) * a - 2 * (S - 1) * a), ages)
            assert bp.band == band and bp.columns == col and abs(bp.s_star - 1.0) < 1e-9
            print(f"{path.name:44s} {f:8.4f} {a:9.6f} {err:7.4f}  {f * (-0.75 * a + 1.0 * a):+13.4f} "
                  f"{f * (-0.75 * a + 0.9 * a):+10.4f}")
    print("nothing submitted")


if __name__ == "__main__":
    main()
