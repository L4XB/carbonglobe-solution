#!/usr/bin/env python
"""Distance of raw / repaired model predictions to the board-calibrated stack C, per band x column (truth coding:
height x 1/1.14114), and the repair directions (P_tag - P_raw, level-free per age x scenario x column).
    python scripts/geo/compare_repairs.py tag1 [tag2 ...]
writes work/dir_rep_<tag>_avg3_lf.npy (mean over the three families, truth coding)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

SC = np.array([1.0 / 1.14114, 1.0])
FAMS = ("gbdt2", "monthL2", "latelat2")
keys = shift.template_keys(); site = keys["site"].to_numpy(); age = keys["age"].to_numpy(); scen = keys["scenario"].to_numpy(); band = shift.band_of(age)
C = shift.read_predictions(ROOT / "results/spatcorr_stack_20261106_s08.csv").astype(float)
B = shift.read_predictions(ROOT / "results/stack_20261010.csv").astype(float)
rd = lambda f, t: shift.read_predictions(ROOT / f"work/repair/{f}_{t}.csv").astype(float) * SC
blocks = [(b, c) for b in ("young", "mid", "old") for c in (0, 1)]


def lf(D: np.ndarray) -> np.ndarray:
    out = D.copy()
    for a in np.unique(age):
        for s in ("ssp126", "ssp585"):
            m = (age == a) & (scen == s); out[m] -= out[m].mean(0)
    return out


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b) + 1e-30))


tags = sys.argv[1:]
raw = {f: rd(f, "raw") for f in FAMS}
print("E[(P - C)^2] per block (yh ya mh ma oh oa | total)")
for t in ["raw"] + tags:
    for f in FAMS:
        p = raw[f] if t == "raw" else rd(f, t)
        e = [np.mean((p[band == b, c] - C[band == b, c]) ** 2) for b, c in blocks]
        print(f"  {t:8s} {f:9s} " + " ".join(f"{v:.4f}" for v in e) + f" | {np.mean((p - C) ** 2):.4f}")
CB = lf(C - B)
dirs = {}
for t in tags:
    Ds = [lf(rd(f, t) - raw[f]) for f in FAMS]
    D = np.mean(Ds, axis=0); dirs[t] = D
    np.save(ROOT / f"work/dir_rep_{t}_avg3_lf.npy", D)
    e = [np.mean(D[band == b, c] ** 2) * np.mean(band == b) / 2 for b, c in blocks]
    print(f"direction {t}: E_all {np.mean(D ** 2):.5f}; per family {[round(float(np.mean(d ** 2)), 4) for d in Ds]}; cross-family cos {cos(Ds[0], Ds[1]):+.2f} {cos(Ds[0], Ds[2]):+.2f} {cos(Ds[1], Ds[2]):+.2f}; "
          f"block shares {[round(v / np.mean(D ** 2), 2) for v in e]}; cos with C - B {cos(D, CB):+.3f}; cos with (C - mean raw) {cos(D, lf(C - np.mean([raw[f] for f in FAMS], 0))):+.3f}")
for i, a in enumerate(tags):
    for b in tags[i + 1:]:
        print(f"cos({a}, {b}) = {cos(dirs[a], dirs[b]):+.3f}")
