#!/usr/bin/env python
"""Old-stand agb outcome (ages >= 150, mean) of ED on training cells vs the organisers' truth, per fire variant root.
    python scripts/ed/train_fire_table.py work/edv3/train_trf6 [more roots]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

roots = [Path(r) for r in sys.argv[1:]]
cells = sorted(int(f.stem) for f in (roots[0] / "truth").glob("*.csv"))
print("cell    truth y0 -> y40 (ratio) | " + " | ".join(f"{r.name[6:]:>22s}" for r in roots) + "   (ED e0 -> x40, ratio; two identical-input runs averaged)")
acc = {r.name: [] for r in roots}
for c in cells:
    t = pd.read_csv(roots[0] / "truth" / f"{c}.csv"); o = t.age >= 150
    t0, t40 = t.y0_agb[o].mean(), t.y40_agb[o].mean(); y = t.age.isin([10, 20, 30]); ty0, ty40 = t.y0_agb[y].mean(), t.y40_agb[y].mean()
    cells_txt = []
    for r in roots:
        try:
            e40 = np.mean([pd.read_csv(r / "results" / f"{c}_{s}.csv").set_index("age").agb_kgC for s in ("ssp126", "ssp585")], axis=0)
            e0 = np.mean([pd.read_csv(r / "results" / f"{c}_{s}_y0.csv").set_index("age").agb_kgC for s in ("ssp126", "ssp585")], axis=0)
            oo = o.to_numpy(); yy = y.to_numpy()
            cells_txt.append(f"{e0[oo].mean():5.1f}->{e40[oo].mean():5.1f} ({e40[oo].mean() / max(e0[oo].mean(), 1e-6):4.2f})")
            acc[r.name].append((t40 / max(t0, 1e-6), e40[oo].mean() / max(e0[oo].mean(), 1e-6), ty40 / max(ty0, 1e-6), e40[yy].mean() / max(e0[yy].mean(), 1e-6)))
        except FileNotFoundError:
            cells_txt.append("          -          ")
    print(f"{c:6d}  {t0:5.1f} -> {t40:5.1f} ({t40 / max(t0, 1e-6):4.2f})  | " + " | ".join(cells_txt))
for r in roots:
    a = np.array(acc[r.name])
    if len(a):
        print(f"{r.name}: n {len(a)}  old ratio rms(log ED/truth) {np.sqrt(np.mean(np.log(np.maximum(a[:, 1], 0.02) / np.maximum(a[:, 0], 0.02)) ** 2)):.3f}  bias {np.mean(np.log(np.maximum(a[:, 1], 0.02) / np.maximum(a[:, 0], 0.02))):+.3f} | young ratio rms {np.sqrt(np.mean(np.log(a[:, 3] / a[:, 2]) ** 2)):.3f} bias {np.mean(np.log(a[:, 3] / a[:, 2])):+.3f}")
