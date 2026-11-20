#!/usr/bin/env python
"""Model-free predictors from the exact training twin of each test site (work/test_cell_exact.npz) and their distance
to the board-calibrated stack C, per band x column (scaled truth units; height truth = physical / 1.14114).

    twin_add   = R + (Y40 - Y0)_twin            (the twin's 40-year change added to the test's own start state)
    twin_ratio = R x (Y40 / Y0)_twin            (relative change; rows with Y0_twin < eps fall back to twin_add)
    twin_abs   = Y40_twin                       (the twin's outcome as is)
R = decoded test y0 (results/latent_ref_y0_m114114.csv, physical / y_mean). Writes work/twin/{name}.npy ((rows, 2),
template order, scaled truth units; unmatched sites = C) and prints E[(P - C)^2] and corr of the change.
    python scripts/geo/twin_predictors.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

DIV = np.array([10.90645, 3.39567])
AGES = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500])
HSCALE = 1.0 / 1.14114            # truth height = physical / 1.14114 (board, 10-03)
import os
ASCALE = float(os.environ.get("ASCALE", "1.0"))   # truth agb = physical x ASCALE (1.0 = the old reading, 0.8763 = coded like height)


def main() -> None:
    keys = shift.template_keys(); site = keys["site"].to_numpy(); age = keys["age"].to_numpy(); scen = keys["scenario"].to_numpy()
    band = shift.band_of(age); ai = np.searchsorted(AGES, age)
    C = shift.read_predictions(ROOT / "results/spatcorr_stack_20261106_s08.csv").astype(float)
    R = shift.read_predictions(ROOT / "results/latent_ref_y0_m114114.csv").astype(float)        # physical / y_mean
    G = shift.read_predictions(ROOT / "work/repair/gbdt2_raw.csv").astype(float)
    tw = np.load(ROOT / "work/train_twin_states.npz"); tsites = tw["sites"]
    y0 = tw["tr0"][:, :, 0, :2].astype(float) / DIV; y40 = tw["tr40"][:, :, :, :2].astype(float).mean(2) / DIV      # (n, 15, 2) physical / y_mean
    idx = np.full(site.max() + 1, -1); idx[tsites] = np.arange(len(tsites))
    k = idx[site]; ok = k >= 0
    Y0 = np.zeros_like(R); Y40 = np.zeros_like(R)
    Y0[ok] = y0[k[ok], ai[ok]]; Y40[ok] = y40[k[ok], ai[ok]]
    out = ROOT / "work/twin"; out.mkdir(exist_ok=True)
    np.save(out / "Y0_twin.npy", np.where(ok[:, None], Y0, np.nan)); np.save(out / "Y40_twin.npy", np.where(ok[:, None], Y40, np.nan))
    sc = np.array([HSCALE, ASCALE])                # physical/y_mean -> truth coding
    Rt = R * sc
    P = {}
    P["twin_add"] = np.clip(R + (Y40 - Y0), 0, None) * sc
    ratio = np.where(Y0 > 0.02, Y40 / np.maximum(Y0, 1e-9), np.nan)
    pr = np.where(np.isfinite(ratio), R * np.clip(ratio, 0.0, 5.0), np.clip(R + (Y40 - Y0), 0, None)) * sc
    P["twin_ratio"] = pr
    P["twin_abs"] = Y40 * sc
    P["gbdt2_raw_h0876"] = G * sc
    P["persistence"] = Rt
    for nm, p in P.items():
        p = np.where(ok[:, None], p, C); P[nm] = p
        if nm.startswith("twin"):
            np.save(out / f"{nm}.npy", p)
    print(f"matched rows {ok.sum()} of {len(ok)}")
    print(f"{'predictor':18s} " + " ".join(f"{b[:1]}{c}" .rjust(15) for b in ("young", "mid", "old") for c in ("h", "a")) + "   total E[(P-C)^2]")
    for nm, p in P.items():
        cells = []
        for b in ("young", "mid", "old"):
            for c in (0, 1):
                m = ok & (band == b)
                e = np.mean((p[m, c] - C[m, c]) ** 2)
                dc = C[m, c] - Rt[m, c]; dp = p[m, c] - Rt[m, c]
                r = np.corrcoef(dc, dp)[0, 1] if dp.std() > 0 else np.nan
                cells.append(f"{e:.4f} ({r:+.2f})")
        print(f"{nm:18s} " + " ".join(s.rjust(15) for s in cells) + f"   {np.mean((p - C) ** 2):.4f}")


if __name__ == "__main__":
    main()
