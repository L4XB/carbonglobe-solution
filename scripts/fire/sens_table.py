#!/usr/bin/env python
"""Old / mid / young agb outcome of several ED roots at the same sites, next to C and y0 (driver sensitivity tests).
    python scripts/fire/sens_table.py name=root [name=root ...] [--sites 4916,5194]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "fire"))
from cgb import shift  # noqa: E402
import build_dir as bd  # noqa: E402

DIV = bd.DIV; M = bd.M
AG = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500])


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("roots", nargs="+"); ap.add_argument("--sites", default=""); ap.add_argument("--band", default="old")
    args = ap.parse_args()
    roots = dict(r.split("=", 1) for r in args.roots)
    k = shift.template_keys(); site = k.site.to_numpy(); scen = k.scenario.to_numpy()
    C = shift.read_predictions(bd.C_PATH).astype(float) * DIV * M
    R = shift.read_predictions("results/latent_ref_y0_m114114.csv").astype(float) * DIV
    g = bd.groups(); ex = np.load(ROOT / "work/test_cell_exact.npz")
    bm = {"old": AG >= 150, "mid": (AG >= 50) & (AG < 150), "young": (AG >= 10) & (AG < 50)}[args.band]
    done = {n: set(int(f.name.split("_")[0]) for f in (ROOT / r / "results").glob("*_ssp585.csv")) for n, r in roots.items()}
    sites = [int(s) for s in args.sites.split(",") if s] or sorted(set.union(*done.values()))
    print(f"{args.band} agb: log(x40 / e0) ssp126 / ssp585 [seed log(e0 / y0)] per root; C = log(C / y0)")
    print("site   cl  lat    lon   y0   |      C      | " + " | ".join(f"{n:^22s}" for n in roots))
    acc = {n: [] for n in roots}; accc = []
    for s in sites:
        cells = []
        c = [np.log(max(C[(site == s) & (scen == sc), 1][bm].mean(), 0.02) / max(R[(site == s) & (scen == sc), 1][bm].mean(), 0.02)) for sc in ("ssp126", "ssp585")]
        for n, r in roots.items():
            if s not in done[n]:
                cells.append(" " * 22); continue
            v = []; sd = 0.0
            for sc in ("ssp126", "ssp585"):
                e40 = pd.read_csv(ROOT / r / "results" / f"{s}_{sc}.csv").set_index("age").loc[AG].agb_kgC.to_numpy()
                e0 = pd.read_csv(ROOT / r / "results" / f"{s}_{sc}_y0.csv").set_index("age").loc[AG].agb_kgC.to_numpy()
                v.append(np.log(max(e40[bm].mean(), 0.02) / max(e0[bm].mean(), 0.02)))
                sd = np.log(max(e0[bm].mean(), 0.02) / max(R[(site == s) & (scen == sc), 1][bm].mean(), 0.02))
            cells.append(f"{v[0]:+.2f} / {v[1]:+.2f} [{sd:+.2f}]"); acc[n].append((s, v[0], v[1], sd))
        accc.append(c)
        print(f"{s:5d}  {int(g['cl'][s])} {ex['lat'][s]:6.1f} {ex['lon'][s]:6.1f} {R[(site == s) & (scen == 'ssp126'), 1][bm].mean():5.1f} | {c[0]:+.2f} / {c[1]:+.2f} | " + " | ".join(f"{x:^22s}" for x in cells))


if __name__ == "__main__":
    main()
