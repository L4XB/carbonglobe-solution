#!/usr/bin/env python
"""Which ED readout form fits the organisers' training trajectories best? Gated MSE per root x column x band.

Forms on the rows that pass the offset-log gate |log((y0 + eps) / (e0 + eps))| <= ln 2 (other rows: persistence):
    hyb<a>  x40 * clip((y0 / e0)^a, 0.5, 2)       (a = 0 is raw x40, a = 1 the test pipeline's hybrid)
    add     y0 + (x40 - e0)
Both scenarios are scored against the same truth row (as in train_gate_eval.py), normalized units.
    python scripts/ed/train_form_eval.py work/edv3/train_fwo_fmax10_trop ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_eval import load  # noqa: E402

DIV = {"h": 10.90645, "a": 3.39567}
EPS = {"h": 0.1, "a": 0.01}
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def main() -> None:
    pd.set_option("display.width", 220)
    for r in sys.argv[1:]:
        d = load(Path(r))
        band = np.where(d.age <= 30, "young", np.where(d.age <= 150, "mid", "old"))
        rows = []
        for c in ("h", "a"):
            y0, y40 = d[f"y0_{c}"].to_numpy(), d[f"y40_{c}"].to_numpy()
            preds = {f: [] for f in [f"hyb{a:g}" for a in ALPHAS] + ["add"]}
            gates = []
            for sx in ("", "b"):
                e0, x40 = np.maximum(d[f"e0{sx}_{c}"].to_numpy(), 0.0), d[f"x40{sx}_{c}"].to_numpy()
                g = np.abs(np.log((y0 + EPS[c]) / (e0 + EPS[c]))) <= np.log(2.0)
                gates.append(g)
                for a in ALPHAS:
                    fac = np.where(e0 > 0, np.clip((y0 / np.maximum(e0, 1e-12)) ** a, 0.5, 2.0), 1.0)
                    p = np.where(y0 > 0, x40 * fac, 0.0)
                    preds[f"hyb{a:g}"].append(np.where(g, p, y0))
                preds["add"].append(np.where(g, np.maximum(y0 + x40 - e0, 0.0), y0))
            for b in ("young", "mid", "old", "all"):
                m = (band == b) if b != "all" else np.ones(len(d), bool)
                row = {"col": c, "band": b, "n": int(m.sum()), "kept": float(np.mean(np.concatenate([g[m] for g in gates]))),
                       "pers": float(np.mean(((y0 - y40)[m] / DIV[c]) ** 2))}
                for f, ps in preds.items():
                    row[f] = float(np.mean(np.concatenate([((p - y40)[m] / DIV[c]) ** 2 for p in ps])))
                rows.append(row)
        print(f"== {Path(r).name}: {d.site.nunique()} sites")
        print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
