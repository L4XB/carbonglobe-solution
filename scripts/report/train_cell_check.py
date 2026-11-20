#!/usr/bin/env python
"""ED-based targets on TRAINING cells, where the organisers' year-40 states are known (numbers for the technical report).

    python scripts/report/train_cell_check.py [work/edv3/train_hv2]

Input: <root>/height_eval_rows.csv (scripts/ed/train_height_eval.py): per cell x seed age the truth (y0, y40) and our ED run of the same cell
(e0 = first record, e40 = year 40) under the test configuration (cycled 40-year spin-up at 415 ppm, windows = the cell's own forcing years).
Predictions compared with the truth of year 40:
    chronosequence   the cell's own y0 profile at age + 40 (no model)
    ED target        that profile x ED's change relative to ED's own profile (the decode of scripts/ed/dec2.py)
Rows are gated like the submission: |log(y0 / e0)| below --gate for the variable. Printed: rms of log(prediction / truth) and the mean squared
error in scored units (height / 10.90645, biomass / 3.39567).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

AGES = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]); LA = np.log(AGES)
EPS = {"h": 0.1, "a": 0.01}; DIV = {"h": 10.90645, "a": 3.39567}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("root", nargs="?", default="work/edv3/train_hv2"); ap.add_argument("--gate", type=float, default=0.12)
    args = ap.parse_args()
    d = pd.read_csv(Path(args.root) / "height_eval_rows.csv"); x = np.log(np.minimum(AGES + 40, 500)); out = []
    for cell, g in d.groupby("cell"):
        g = g.set_index("age").reindex(AGES)
        for c in ("h", "a"):
            e = EPS[c]; y0 = g[f"y0_{c}"].to_numpy(); e0 = np.maximum(g[f"e0_{c}"].to_numpy(), 0); e40 = np.maximum(g[f"e40_{c}"].to_numpy(), 0)
            y40 = g[f"y40_{c}"].to_numpy()
            yc = np.exp(np.interp(x, LA, np.log(y0 + e))); ec = np.exp(np.interp(x, LA, np.log(e0 + e)))      # both chronosequences at age + 40
            for i, a in enumerate(AGES):
                out.append(dict(cell=cell, zone=g.zone.iloc[i], age=a, col=c, truth=y40[i], chrono=yc[i] - e, ed=yc[i] * (e40[i] + e) / ec[i] - e,
                                raw=e40[i], mis=abs(np.log((y0[i] + e) / (e0[i] + e)))))
    D = pd.DataFrame(out).dropna()
    print(f"{args.root}: {D.cell.nunique()} training cells, zones {D.groupby('zone').cell.nunique().to_dict()}, gate {args.gate}")
    for c, cn in (("h", "height"), ("a", "biomass")):
        g = D[D.col == c]; ok = g[g.mis < args.gate]; e = EPS[c]
        rl = lambda p, t: float(np.sqrt(np.mean(np.log((p + e) / (t + e)) ** 2)))
        ms = lambda p, t: float(np.mean(((p - t) / DIV[c]) ** 2))
        print(f"   {cn:8s} rows passing the gate {len(ok)} of {len(g)} ({100 * len(ok) / len(g):.0f} %)")
        for nm, k in (("chronosequence alone", "chrono"), ("ED target", "ed"), ("ED year 40, raw", "raw")):
            print(f"      {nm:22s} rms log error {rl(ok[k], ok.truth):.3f}   MSE in scored units {ms(ok[k], ok.truth):.5f}   (all rows: {rl(g[k], g.truth):.3f} / {ms(g[k], g.truth):.5f})")
    ok = D[D.mis < args.gate]
    for k in ("chrono", "ed", "raw"):
        m = np.mean([np.mean(((ok[ok.col == c][k] - ok[ok.col == c].truth) / DIV[c]) ** 2) for c in ("h", "a")])
        print(f"   mean of the two columns, gated rows: {k:7s} MSE {m:.5f}")


if __name__ == "__main__":
    main()
