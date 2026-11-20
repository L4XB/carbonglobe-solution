#!/usr/bin/env python
"""Yearly ignition factors X[k] (window year k = 0..39) for ED's ED_FIRE_X_FILE (see scripts/fire/emulator.py).
    series(mode, site, scenario, train=False) -> (40,)
modes
  roll      training cells: the cell's own MERRA-2 years, rolling 12-slot array (validation against the truth)
  truthx    training cells: factors read off the cell's own truth trajectory (old stands)
  twinx     test sites: the exact twin's truth-derived factors, years k -> k
  mer       test sites: the exact twin's MERRA-2 years k -> k (rolling)
  mer:<map> test sites: twin years in another order: rev | first20 | last20 | shiftN
  clim40    test sites: the twin's 40-year mean climatology, constant (clim10 / clim20: last 10 / 20 years)
  fut       test sites: the window's own monthly climate, release transform undone (rolling)
  rel       test sites: the window's monthly climate exactly as released (rolling)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "fire")); sys.path.insert(0, str(ROOT / "scripts" / "ed"))
from cgb import io  # noqa: E402
import emulator as em  # noqa: E402
import fire_proxy_calib as fp  # noqa: E402

_C: dict = {}


def _train_tp(cell: int) -> tuple[np.ndarray, np.ndarray]:
    tw = np.asarray(io.open_train_x()[int(cell), :, :, :2], dtype=np.float64)
    return tw[:, :, 0] - 273.15, tw[:, :, 1]


FP1 = 1.64e-5; SURV = 0.7; CAP_B = 0.127          # ED: rate = fuel x fp1 x X, agb loss fraction = 0.7 (1 - exp(-rate)), cap 0.2 -> 0.127
M = 1.14114


def truth_x(cell: int, x_emul: np.ndarray) -> np.ndarray:
    """Ignition factors read off the organisers' own trajectory of training cell `cell` (old stands, ages 150-500):
    yearly loss b = -ln(agb ratio) + regrowth(deficit); unsaturated years give X = rate / (fuel x fp1) exactly, saturated
    years keep at least the factor that saturates (the emulator's X when larger), quiet years give 0."""
    A = np.mean([np.asarray(io.open_train_y(a)[int(cell), :, 0, 1], dtype=np.float64) for a in (150, 200, 250, 300, 350, 400, 450, 500)], axis=0) * M
    x = np.zeros(40)
    aeq = A[0]
    for k in range(40):
        fuel = max(A[k], 1e-3)
        loss = 1.0 - (A[k + 1] / fuel) * float(np.exp(-em.regrowth(np.log(max(aeq, 1e-6) / fuel))))      # agb fraction lost to fire
        if loss < 0.012 or A[k + 1] > 0.99 * fuel:     # a burn year shows a net decline (the regrowth curve alone must not create burns)
            continue
        if fuel * FP1 * float(x_emul[max(k - 1, 0):k + 2].max()) < 1e-3:   # no dryness around this year: the loss is not fire (cold / drought die-off)
            continue
        if loss >= 0.9 * CAP_B:
            x[k] = max(x_emul[k], 3.0 * 0.2 / (fuel * FP1))
        else:
            rate = -np.log(1.0 - loss / SURV)
            x[k] = rate / (fuel * FP1)
    return x


def year_map(name: str) -> np.ndarray:
    """Window year j (0 = 2021) -> index of the MERRA-2 training year (0 = 1981).
    cyc35: the shipped ED constants MERRA2_START 1981 / MERRA2_END 2015 used cyclically: 2021 + j -> 1981 + (40 + j) mod 35."""
    idx = np.arange(40)
    if name == "cyc35":
        return (idx + 5) % 35
    if name.startswith("shift"):
        return (idx + int(name[5:])) % 40
    return {"kk": idx, "rev": idx[::-1], "first20": np.r_[idx[:20], idx[:20]], "last20": np.r_[idx[20:], idx[20:]]}[name]


def series(mode: str, site: int, scenario: str, train: bool = False) -> np.ndarray:
    if train:
        if "tll" not in _C: _C["tll"] = np.load(ROOT / "work/edv3/train_latlon.npy")
        T, P = _train_tp(site); lat = float(_C["tll"][site, 0])
        if mode == "roll":
            return em.xroll(T, P, lat).mean(1)
        if mode == "truthx":
            return truth_x(site, em.xroll(T, P, lat).mean(1))
        if mode.startswith("merm") or mode.startswith("trm"):
            # test-style structures on a training cell (forced-fire validation, scripts/ed/forced_fire.py): 480 monthly factors of the cell's own MERRA-2
            # years with the optional wet-month rule. merm = lookup (the array before 1981 holds the cycle's last year), trm = window (no wrap)
            xr = em.xroll(T, P, lat, T[-1], P[-1]) if mode.startswith("merm") else em.xroll(T, P, lat)
            for o in mode.split(":")[1:]:
                if o.startswith("rule"):
                    xr = xr * (P < float(o[4:]))
            return xr.reshape(-1)
        raise KeyError(f"training mode {mode}")
    if "ex" not in _C: _C["ex"] = np.load(ROOT / "work/test_cell_exact.npz")
    ex = _C["ex"]; cell = int(ex["cell"][site]); lat = float(ex["lat"][site]) if np.isfinite(ex["lat"][site]) else 0.0
    base = mode.split(":", 1)[0]
    if base in ("fut", "rel", "relp"):
        # fut: release transform undone; rel: monthly climate exactly as delivered. Optional ":rule<mm>" = the shipped
        # wet-month rule (no ignition in months with more than <mm> of rain; 8.33 mm = precip * 12 > 100) and ":own" =
        # the array before the first year holds that year itself (default: the twin's last MERRA year).
        if "cal" not in _C:
            z = np.load(ROOT / "work/cal/cal_maps_cal.npz"); _C["cal"] = (z["alpha"][:2].astype(float), z["beta"][:2].astype(float))
        al, be = _C["cal"]
        opts = mode.split(":")[1:]
        x = np.asarray(io.open_test(scenario)[0][site, :, :, :2], dtype=np.float64)
        # relp (11-10): temperature real, precipitation as released (0.868 x real + 6.35 mm)
        T = ((x[:, :, 0] - be[0]) / al[0] if base in ("fut", "relp") else x[:, :, 0]) - 273.15
        P = np.clip((x[:, :, 1] - be[1]) / al[1], 0.0, None) if base == "fut" else x[:, :, 1]
        prev = _train_tp(cell) if (cell >= 0 and "own" not in opts) else None
        xr = em.xroll(T, P, lat, None if prev is None else prev[0][-1], None if prev is None else prev[1][-1])
        for o in opts:
            if o.startswith("rule"):
                xr = xr * (P < float(o[4:]))
        if "monthly" in opts:                         # 480 monthly factors: ED caps every month separately (build_fire)
            return xr.reshape(-1)
        return xr.mean(1)
    if cell < 0:
        return np.zeros(40)
    T, P = _train_tp(cell)
    if mode.startswith("merm"):                       # lookup on the twin's cycled MERRA-2 years: 40 x 12 monthly factors, optional ":rule<mm>"
        # mermad / mermadp (11-10): the fire module sees the MERRA months through the release transform (released = alpha x real + beta):
        # mermad = T and P, mermadp = P only; the rule then acts on the transformed P (8.33 mm = 2.28 mm real)
        if base in ("mermad", "mermadp"):
            if "cal" not in _C:
                z = np.load(ROOT / "work/cal/cal_maps_cal.npz"); _C["cal"] = (z["alpha"][:2].astype(float), z["beta"][:2].astype(float))
            al, be = _C["cal"]
            P = al[1] * P + be[1]
            if base == "mermad":
                T = al[0] * (T + 273.15) + be[0] - 273.15
        if base in ("mermw", "merms"):                # 11-10: shipped January handling instead of "January index = 0"
            import lookup_fit2 as lf2                 # mermw: a dry January collects the dry months before it; merms: a wet January also keeps its stale index
            di = lf2.xroll(T, P, lat, True) if base == "mermw" else lf2.xroll_stale(T, P, lat)
            xr = (np.maximum(di, 1.0) / 30000.0) ** 10
        else:
            xr = em.xroll(T, P, lat, T[-1], P[-1])    # the array before 1981 holds the cycle's last year
        for o in mode.split(":")[1:]:
            if o.startswith("rule"):
                xr = xr * (P < float(o[4:]))
        return xr.reshape(-1)
    if mode.startswith("climm"):                      # lookup with the shipped wet-month rule: 12 climatological monthly factors x 30 (monthly mode of build_fire)
        opts = mode.split(":"); n = int(opts[0][5:])
        Tc, Pc = T[40 - n:].mean(0), P[40 - n:].mean(0)
        xr = em.xroll(Tc[None], Pc[None], lat)[0]
        for o in opts[1:]:
            if o.startswith("rule"):
                xr = xr * (Pc < float(o[4:]))
        return np.tile(xr, 30)
    if mode.startswith("clim"):
        n = int(mode[4:]); b = fp.b_factors(lat)
        di = em._di_zero(T[40 - n:].mean(0)[None], P[40 - n:].mean(0)[None], b)[0]
        return np.full(40, (max(di, 1.0) / 30000.0) ** 10)
    if mode.startswith("twinx"):                      # the twin's truth-derived factors, years k -> k or twinx:<map>
        x = truth_x(cell, em.xroll(T, P, lat).mean(1))
        return x[year_map(mode.split(":", 1)[1])] if ":" in mode else x
    if mode.startswith("mer"):
        idx = year_map(mode.split(":", 1)[1] if ":" in mode else "kk")
        return em.xroll(T[idx], P[idx], lat).mean(1)
    raise KeyError(mode)
