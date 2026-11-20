#!/usr/bin/env python
"""Fast emulator of the organisers' fire regime (old-stand agb), calibrated on their TRAINING trajectories.

Findings behind it (notes/board-solve.md, 2026-11-09/10):
  * the yearly loss follows ED's ignition  fuel x fp1 x (DI / 30000)^10  capped per year,
  * DI = ED's Thornthwaite dryness index, but the January index is always 0 (no backward collection) and
  * the 12-slot climate array is ROLLING: in month m of year k it holds months <= m of year k and months > m of
    year k-1; the burn seen from December of year k-1 to December of year k uses the mean over the 12 months of year k.

    xroll(T, P, lat)         -> (years, 12) monthly ignition factor (DI_roll / 30000)^10
    simulate(a0, W, params)  -> (sites, years + 1) agb trajectory for old stands
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ed"))
import fire_proxy_calib as fp  # noqa: E402

PARAMS = dict(c=5.0e-6, cap=0.150, rho=0.045, power=10.0)


def _di_zero(T: np.ndarray, P: np.ndarray, b: np.ndarray) -> np.ndarray:
    """T, P (..., 12) deg C and mm/month -> DI_avg with the January index fixed at 0."""
    Tc = np.maximum(T, 0.0001)
    I = np.sum((Tc / 5.0) ** 1.51, axis=-1, keepdims=True)
    a = 67.5e-8 * I ** 3 - 77.1e-6 * I ** 2 + 0.0179 * I + 0.492
    pet = 1.62 * b * (10.0 * Tc / I) ** a * 120.0
    pr = np.maximum(P * 12.0, 1e-9)
    dry = pet > pr
    inc = 30.0 * (pet - pr)
    di = np.zeros(T.shape)
    for m in range(1, 12):
        di[..., m] = np.where(dry[..., m], di[..., m - 1] + inc[..., m], 0.0)
    return di.mean(axis=-1)


def xroll(T: np.ndarray, P: np.ndarray, lat: float, T_prev: np.ndarray | None = None, P_prev: np.ndarray | None = None, power: float = 10.0) -> np.ndarray:
    """T, P (years, 12) -> (years, 12) ignition factor of each month with the rolling 12-slot array.
    T_prev / P_prev (12,) fill the array before the first year (default: the first year itself)."""
    b = fp.b_factors(float(lat))
    ny = T.shape[0]
    Tp = np.concatenate([(T[:1] if T_prev is None else T_prev[None]), T], axis=0)       # (ny + 1, 12)
    Pp = np.concatenate([(P[:1] if P_prev is None else P_prev[None]), P], axis=0)
    out = np.zeros((ny, 12))
    for m in range(12):
        Ta = T.copy(); Pa = P.copy()
        Ta[:, m + 1:] = Tp[:-1, m + 1:]; Pa[:, m + 1:] = Pp[:-1, m + 1:]
        out[:, m] = (np.maximum(_di_zero(Ta, Pa, b), 1.0) / 30000.0) ** power
    return out


# median yearly regrowth of old stands in quiet years of the organisers' training runs, by deficit log(a_eq / a)
REGROW_D = np.array([-0.5, 0.0, 0.065, 0.15, 0.30, 0.55, 0.95, 2.0, 4.0])
REGROW_G = np.array([-0.03, 0.0, 0.001, 0.004, 0.010, 0.024, 0.046, 0.064, 0.070])


def regrowth(d: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return scale * np.interp(d, REGROW_D, REGROW_G)


def simulate(a0: np.ndarray, W: np.ndarray, c: float = PARAMS["c"], cap: float = PARAMS["cap"], rho: float = PARAMS["rho"], aeq: np.ndarray | None = None,
             teacher: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """a0 (n,) agb in kgC/m2 at the window start, W (n, years) yearly mean ignition factor.
    Returns (trajectory (n, years + 1), burn rate (n, years)). teacher: truth trajectory used as the fuel (one-step mode)."""
    n, ny = W.shape
    aeq = a0 if aeq is None else aeq
    a = np.zeros((n, ny + 1)); a[:, 0] = a0; r = np.zeros((n, ny))
    for k in range(ny):
        cur = a[:, k] if teacher is None else teacher[:, k]
        r[:, k] = np.minimum(cap, c * cur * W[:, k])
        d = np.log(np.maximum(aeq, 1e-6) / np.maximum(cur, 1e-6))
        grow = np.clip(rho * d, -0.05, 0.12) if rho > 0 else regrowth(d)
        a[:, k + 1] = cur * np.exp(-r[:, k] + grow)
    return a, r
