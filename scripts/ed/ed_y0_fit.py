#!/usr/bin/env python
"""How well do an ED run's seed states (results/<site>_<scen>_y0.csv) match the decoded test y0? Per band and column:
median |log(ED / y0)|, share within 10%, corr(log); optional site list (file of ids) to compare runs on the same sites.
    python scripts/ed/ed_y0_fit.py <results dir> [site list file]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

DIV = np.array([10.90645, 3.39567])
d = Path(sys.argv[1])
only = set(int(l) for l in open(sys.argv[2]) if l.strip()) if len(sys.argv) > 2 else None
keys = shift.template_keys()
R = shift.read_predictions(ROOT / "results/latent_ref_y0_m114114.csv").astype(np.float64) * DIV
ref = pd.DataFrame({"site": keys["site"], "age": keys["age"], "scen": keys["scenario"], "h": R[:, 0], "a": R[:, 1]})
ref = ref[ref.scen == "ssp126"].set_index(["site", "age"])
rows = []
for f in sorted(d.glob("*_ssp126_y0.csv")):
    s = int(f.name.split("_")[0])
    if only is not None and s not in only:
        continue
    e = pd.read_csv(f)
    for _, r in e.iterrows():
        rows.append((s, int(r["age"]), r["height_m"], r["agb_kgC"]))
df = pd.DataFrame(rows, columns=["site", "age", "eh", "ea"]).set_index(["site", "age"]).join(ref, how="inner")
df["band"] = shift.band_of(df.index.get_level_values("age").to_numpy())
print(f"{d.parent.name}: {df.index.get_level_values('site').nunique()} sites")
for b in ("young", "mid", "old"):
    x = df[df.band == b]
    out = []
    for e, t, nm in (("ea", "a", "agb"), ("eh", "h", "h")):
        g = (x[e] > 0.05) & (x[t] > 0.05)
        lr = np.log(x.loc[g, e]) - np.log(x.loc[g, t])
        out.append(f"{nm}: med|log| {np.median(np.abs(lr)):.3f} within10% {np.mean(np.abs(lr) < 0.0953):.2f} bias {np.median(lr):+.3f} corr {np.corrcoef(np.log(x.loc[g, e]), np.log(x.loc[g, t]))[0, 1]:+.2f}")
    print(f"  {b:5s} " + " | ".join(out))
