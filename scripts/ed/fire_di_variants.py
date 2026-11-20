#!/usr/bin/env python
"""Variants of ED's yearly dryness index for the fire calibration (truth = organisers' training trajectories).
    di_variants(T, P, b) -> dict name -> (years,) DI_avg
T (years, 12) deg C, P (years, 12) mm/month, b = Thornthwaite day-length factors (fire_proxy_calib.b_factors).
  wrap : shipped code with January reset (a dry January collects the dry months that end the same year)
  own  : a dry January starts from its own deficit (no backward collection)
  zero : the January index is always 0 (a dry February starts from 0)
"""
from __future__ import annotations

import numpy as np


def monthly_terms(T: np.ndarray, P: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Tc = np.maximum(T, 0.0001)
    I = np.sum((Tc / 5.0) ** 1.51, axis=1, keepdims=True)
    a = 67.5e-8 * I ** 3 - 77.1e-6 * I ** 2 + 0.0179 * I + 0.492
    pet = 1.62 * b[None, :] * (10.0 * Tc / I) ** a * 120.0          # mm/yr
    pr = np.maximum(P * 12.0, 1e-9)
    return pet / pr > 1.0, 30.0 * (pet - pr)


def di_variants(T: np.ndarray, P: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    dry, inc = monthly_terms(T, P, b)
    ny = T.shape[0]
    out = {k: np.zeros(ny) for k in ("wrap", "own", "zero")}
    mon = {k: np.zeros((ny, 12)) for k in out}
    for y in range(ny):
        for name in out:
            di = np.zeros(12)
            if dry[y, 0] and name != "zero":
                d0 = inc[y, 0]
                if name == "wrap":
                    for m2 in range(11, 0, -1):
                        if dry[y, m2]:
                            d0 += inc[y, m2]
                        else:
                            break
                di[0] = d0
            for m in range(1, 12):
                di[m] = di[m - 1] + inc[y, m] if dry[y, m] else 0.0
            out[name][y] = di.mean(); mon[name][y] = di
    out["_monthly"] = mon
    return out
