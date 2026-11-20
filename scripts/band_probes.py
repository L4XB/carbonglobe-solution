#!/usr/bin/env python
"""Block-replacement probes: the model on one seed-age band, the reference elsewhere.

With ``S_ref`` the reference file's board score and ``S_b`` the score of the file
that carries the model only on band ``b``, ``S_b - S_ref = f_b (M_b - Ref_b)`` where
``f_b`` is the band's share of the scored cells and ``M_b``/``Ref_b`` the two files'
MSE inside the band: each probe measures the model's gain over the reference in that
band outright, and the three bands plus the full file are one consistency check
(the three deltas must sum to the full file's delta). Files are written in template
order and validated on re-read; nothing is submitted.

    ./.venv/bin/python scripts/band_probes.py --model results/latent_noco2x2.csv \
        --ref results/latent_ref_y0_m114114.csv --tag noco2x2 [--bands young,mid,old]
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
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bands", default="young,mid,old")
    ap.add_argument("--scenario", default="", help="restrict the replaced block to one scenario (ssp126|ssp585)")
    ap.add_argument("--column", default="", help="restrict the replaced block to one column (height|agb)")
    args = ap.parse_args()

    model = shift.read_predictions(args.model).astype(np.float64)
    ref = shift.read_predictions(args.ref).astype(np.float64)
    keys = shift.template_keys()
    ages = keys["age"].to_numpy()
    masks = shift.band_masks(ages)
    full_delta = float(np.mean((model - ref) ** 2))
    print(f"E[(model - ref)^2] over all cells: {full_delta:.6f}")
    for band in args.bands.split(","):
        m = masks[band]
        if args.scenario:
            m = m & (keys["scenario"].to_numpy() == args.scenario)
        pred = ref.copy()
        if args.column:
            k = shift.COLUMNS.index(args.column)
            pred[m, k] = model[m, k]
        else:
            pred[m] = model[m]
        f_b = m.sum() / len(m)
        a_b = float(np.mean((pred[m] - ref[m]) ** 2))
        stem = f"bandprobe_{args.tag}_{band}{'_' + args.scenario if args.scenario else ''}{'_' + args.column if args.column else ''}"
        path = ROOT / "results" / f"{stem}.csv"
        shift.write_predictions(pred, path)
        print(f"{path.name}: band {band:5s} rows {int(m.sum()):6d} (share {f_b:.4f}); E_band[(model-ref)^2] {a_b:.5f}; "
              f"contribution to E[(p-ref)^2] {f_b * a_b:.5f}")


if __name__ == "__main__":
    main()
