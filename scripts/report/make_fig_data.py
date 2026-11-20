#!/usr/bin/env python
"""Curves and counts for the technical report, read from a finished submission file.

    python scripts/report/make_fig_data.py results/<final>.csv [--site 3138]

Writes
    report/figures/data/site_curve.dat   one example site: released initial states (15 seed ages) and the file's year-40 values
    report/figures/data/auto_macros.tex  counts taken from the data and the score registry (scored files, sites, rows, the example site)
Biomass is shown in physical units: physical = 1.14114 x (stored - floor), the decode of the released test states
(src/cgb/shift.py: M_HEIGHT, FLOORS); a submission value v is stored = 3.39567 v.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io, shift  # noqa: E402

M = 1.14114
DIV_A = 3.39567


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("final")
    ap.add_argument("--site", type=int, default=3138)
    args = ap.parse_args()
    out = ROOT / "report/figures/data"; out.mkdir(parents=True, exist_ok=True)
    keys = shift.template_keys(); scen = keys.scenario.to_numpy(); site = keys.site.to_numpy()
    n_sites = int(site.max()) + 1; ages = np.array(io.SEED_AGES)
    P = shift.read_predictions(ROOT / args.final).astype(np.float64)                      # (rows, 2): height, agb in submission units
    fl_a = float(shift.FLOORS[1])
    pred = {}
    for sc in io.SCENARIOS:
        m = scen == sc
        assert (site[m].reshape(n_sites, len(ages))[:, 0] == np.arange(n_sites)).all()
        pred[sc] = M * (P[m, 1].reshape(n_sites, len(ages)) * DIV_A - fl_a)               # (sites, 15) kg C m-2
    y0 = M * (np.asarray(io.open_test("ssp585")[1], dtype=np.float64)[:, :, 1] - fl_a)    # (sites, 15) kg C m-2
    s = args.site
    with open(out / "site_curve.dat", "w") as fh:
        fh.write("age y0 age40 p585 p126\n")
        for i, a in enumerate(ages):
            fh.write(f"{a} {y0[s, i]:.3f} {a + 40} {pred['ssp585'][s, i]:.3f} {pred['ssp126'][s, i]:.3f}\n")
    ex = np.load(ROOT / "work/test_cell_exact.npz"); lat, lon = float(ex["lat"][s]), float(ex["lon"][s])
    scores = json.load(open(ROOT / "results/board_scores.json"))
    n_scored = sum(1 for v in scores.values() if isinstance(v, (int, float)))
    n_cells = int((np.asarray(ex["cell"]) >= 0).sum())
    lat_s = f"{abs(lat):.2f}$^\\circ$\\,{'N' if lat >= 0 else 'S'}"; lon_s = f"{abs(lon):.2f}$^\\circ$\\,{'E' if lon >= 0 else 'W'}"
    lines = [
        "% written by scripts/report/make_fig_data.py - do not edit by hand",
        f"\\newcommand{{\\cgNscored}}{{{n_scored}}}                 % scored files in results/board_scores.json",
        f"\\newcommand{{\\cgNsites}}{{{n_sites:,}}}".replace(",", "{,}"),
        f"\\newcommand{{\\cgNrows}}{{{len(keys):,}}}".replace(",", "{,}"),
        f"\\newcommand{{\\cgNmatched}}{{{n_cells:,}}}".replace(",", "{,}") + "              % test sites with exactly one matching training cell",
        f"\\newcommand{{\\cgExSite}}{{{s}}}",
        f"\\newcommand{{\\cgExWhere}}{{{lat_s}, {lon_s}}}",
        f"\\newcommand{{\\cgFinalFile}}{{{Path(args.final).name.replace('_', chr(92) + '_')}}}",
    ]
    (out / "auto_macros.tex").write_text("\n".join(lines) + "\n")
    print((out / "site_curve.dat").read_text())
    print((out / "auto_macros.tex").read_text())


if __name__ == "__main__":
    main()
