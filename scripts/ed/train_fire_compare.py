#!/usr/bin/env python
"""Fire on vs fire off against the organisers' training trajectories (roots of train_site.py on the same sites).

Per seed age: truth ratios y40/y0 (height, agb) next to ED's x40/e0 for each root, the median ratios by band and the
gated hybrid MSE against persistence (scripts/ed/train_gate_eval.py logic, alpha 1).
    python scripts/ed/train_fire_compare.py work/edv3/train_c400_s30_trop work/edv3/train_c400f_s30_trop
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_eval import load  # noqa: E402


def main() -> None:
    roots = [Path(r) for r in sys.argv[1:]]
    frames = {r.name: load(r) for r in roots}
    common = set.intersection(*[set(f.site) for f in frames.values()])
    out = None
    for name, d in frames.items():
        d = d[d.site.isin(common) & (d.y0_a > 0.5) & (d.y0_h > 0.5)].copy()
        d[f"h_{name}"] = d.x40_h / d.e0_h.clip(lower=1e-6); d[f"a_{name}"] = d.x40_a / d.e0_a.clip(lower=1e-6)
        keep = d[["site", "age", "y0_h", "y40_h", "y0_a", "y40_a", f"h_{name}", f"a_{name}"]]
        out = keep if out is None else out.merge(keep[["site", "age", f"h_{name}", f"a_{name}"]], on=["site", "age"])
    out["t_h"] = out.y40_h / out.y0_h; out["t_a"] = out.y40_a / out.y0_a
    out["band"] = np.where(out.age <= 30, "young", np.where(out.age <= 150, "mid", "old"))
    print(f"{len(common)} common sites, {len(out)} rows")
    cols = ["t_h"] + [f"h_{n}" for n in frames] + ["t_a"] + [f"a_{n}" for n in frames]
    print("median ratios by band (truth t_, ED per root):")
    print(out.groupby("band")[cols].median().round(3).to_string())
    print("correlation with the truth ratio, all rows:")
    for n in frames:
        print(f"  {n}: h {np.corrcoef(out.t_h, out[f'h_{n}'])[0, 1]:+.2f}  agb {np.corrcoef(out.t_a, out[f'a_{n}'])[0, 1]:+.2f}  "
              f"| rms log-ratio error h {np.sqrt(np.mean(np.log(out[f'h_{n}'] / out.t_h) ** 2)):.3f} agb {np.sqrt(np.mean(np.log(out[f'a_{n}'].clip(lower=1e-3) / out.t_a) ** 2)):.3f}")
    print("per site medians (old band):")
    print(out[out.band == "old"].groupby("site")[cols].median().round(2).to_string())


if __name__ == "__main__":
    main()
