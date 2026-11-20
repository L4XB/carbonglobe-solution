#!/usr/bin/env python
"""MSE-optimal fire cap on training sites by log-cap interpolation between runs at different fire_max_disturbance_rate.

Each argument is cap=root (train_site.py roots on the same sites). Per row the ED ratio log(x40 / e0) is interpolated
piecewise-linearly in log(cap) (linear extrapolation outside the run caps, at most one octave); the hybrid prediction
x40 * clip(y0 / e0, 0.5, 2) is scored against the truth y40 in normalized units (DIV), both scenarios pooled.
    python scripts/ed/cap_interp_eval.py 0.2=work/edv3/train_c400fwo_s30_trop \
        0.1=work/edv3/train_fwo_fmax10_trop 0.05=work/edv3/train_fwo_fmax05_trop
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_eval import load  # noqa: E402

DIV = {"h": 10.90645, "a": 3.39567}


def main() -> None:
    runs = sorted(((float(a.split("=")[0]), Path(a.split("=")[1])) for a in sys.argv[1:]), key=lambda t: t[0])
    caps = np.array([c for c, _ in runs]); lc = np.log(caps)
    frames = [load(r) for _, r in runs]
    keys = set.intersection(*[set(zip(f.site, f.age)) for f in frames])
    frames = [f[[k in keys for k in zip(f.site, f.age)]].sort_values(["site", "age"]).reset_index(drop=True) for f in frames]
    base = frames[0]
    band = np.where(base.age <= 30, "young", np.where(base.age <= 150, "mid", "old"))
    print(f"{base.site.nunique()} sites, {len(base)} rows; run caps {caps.tolist()}")
    grid = np.exp(np.linspace(lc[0] - np.log(2.0), lc[-1] + np.log(2.0), 41))
    for c in ("h", "a"):
        y0, y40 = base[f"y0_{c}"].to_numpy(), base[f"y40_{c}"].to_numpy()
        # per run, per scenario: log ratio x40 / e0 (both scenarios against the same truth row, as in train_gate_eval)
        res = {}
        for scen, sx in (("126", ""), ("585", "b")):
            lr = np.stack([np.log((f[f"x40{sx}_{c}"].to_numpy() + 1e-6) / (f[f"e0{sx}_{c}"].to_numpy() + 1e-6)) for f in frames])
            e0 = np.stack([f[f"e0{sx}_{c}"].to_numpy() for f in frames]).mean(axis=0)
            res[scen] = (lr, e0)
        print(f"== {'height' if c == 'h' else 'agb'}: MSE (normalized) by cap; persistence per band in brackets")
        for b in ("young", "mid", "old", "all"):
            m = (band == b) if b != "all" else np.ones(len(base), bool)
            m &= y0 > 0
            pers = np.mean(((y0 - y40)[m] / DIV[c]) ** 2)
            out = []
            for g in grid:
                lg = np.log(g); errs = []
                for scen in ("126", "585"):
                    lr, e0 = res[scen]
                    if lg <= lc[0]:
                        w = (lg - lc[0]) / (lc[1] - lc[0]); v = lr[0] + w * (lr[1] - lr[0])
                    elif lg >= lc[-1]:
                        w = (lg - lc[-2]) / (lc[-1] - lc[-2]); v = lr[-2] + w * (lr[-1] - lr[-2])
                    else:
                        j = np.searchsorted(lc, lg) - 1; w = (lg - lc[j]) / (lc[j + 1] - lc[j]); v = lr[j] + w * (lr[j + 1] - lr[j])
                    x40 = np.exp(v) * (e0 + 1e-6)
                    hyb = x40 * np.clip(y0 / np.maximum(e0, 1e-12), 0.5, 2.0)
                    errs.append(((hyb - y40)[m] / DIV[c]) ** 2)
                out.append(np.mean(np.concatenate(errs)))
            out = np.array(out); k = int(np.argmin(out))
            at = {cap: out[int(np.argmin(np.abs(np.log(grid) - np.log(cap))))] for cap in caps}
            print(f"  {b:5s} n {int(m.sum()):3d} [pers {pers:.4f}]  best cap {grid[k]:.3f} MSE {out[k]:.4f} | "
                  + " ".join(f"cap {cap:g}: {v:.4f}" for cap, v in at.items()))


if __name__ == "__main__":
    main()
