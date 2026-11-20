#!/usr/bin/env python
"""Fit the TEST world's fire rule on the test_y0 age profiles (the lookup = 500 years on the cycled MERRA-2 years).

State-space emulator of old-stand agb in the lookup of one site:
    A[t+1] = (A[t] + g(A[t])) * (1 - SURV_LOSS * b[t]),     b[t] = min(cap_y, mean_m min(cap_m, A[t] * fp1 * X[k, m] * [P[k, m] < thr]))
with k = t mod 40 the MERRA year, X = (rolling dryness index / 30000)^10 (scripts/fire/emulator.xroll) and g(A) the growth
increment of the site's NO-FIRE lookup (ED, cycled MERRA, from a spin-only run) at the age where that curve passes A.
Compared with the decoded test_y0 agb at the 15 seed ages (log rms over ages >= 30).

    python scripts/fire/lookup_fire_fit.py --nofire work/edv3/sp_cxs_tc40b [--sites file] [--grid]
"""
from __future__ import annotations

import argparse
import glob
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "fire")); sys.path.insert(0, str(ROOT / "scripts" / "ed"))
from cgb import io  # noqa: E402
import emulator as em  # noqa: E402

AGES = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500])
SURV_LOSS = 0.7


def site_data(root: Path, sites: list[int] | None) -> dict[int, dict]:
    ex = np.load(ROOT / "work/test_cell_exact.npz"); X = io.open_train_x()
    out = {}
    for f in sorted(glob.glob(str(root / "csv" / "site*.csv"))):
        t = pd.read_csv(f); s = int(t.site.iloc[0])
        if sites is not None and s not in sites:
            continue
        a = t[t.scenario == "ssp126"].set_index("age").reindex(AGES); b = t[t.scenario == "ssp585"].set_index("age").reindex(AGES)
        c = int(ex["cell"][s])
        if c < 0:
            continue
        T = np.asarray(X[c, :, :, 0], dtype=np.float64) - 273.15; P = np.asarray(X[c, :, :, 1], dtype=np.float64)
        xr = em.xroll(T, P, float(ex["lat"][s]), T[-1], P[-1])
        out[s] = dict(y0=0.5 * (a.y0_agb.to_numpy() + b.y0_agb.to_numpy()), nf=a.spin_agb_kgC.to_numpy(), xr=xr, P=P, lat=float(ex["lat"][s]))
    return out


def growth_fn(nf: np.ndarray):
    """Yearly increment g(A) of the no-fire lookup curve as a function of the state A (monotone envelope of the curve)."""
    t = np.concatenate([[0.0], AGES.astype(float)]); a = np.concatenate([[0.0], np.maximum.accumulate(nf)])
    tt = np.arange(0, 501); aa = np.interp(tt, t, a)
    inc = np.diff(aa, append=aa[-1])
    order = np.argsort(aa)
    def g(A: float) -> float:
        return float(np.interp(A, aa[order], inc[order]))
    return g, float(aa[-1])


def simulate(d: dict, fp1: float, thr: float, cap_m: float, cap_y: float, fuel_pow: float = 1.0) -> np.ndarray:
    g, _ = growth_fn(d["nf"])
    W = d["xr"] * (d["P"] < thr)                      # (40, 12)
    A = 0.0; out = np.zeros(501)
    for t in range(500):
        k = t % 40
        A = A + g(A)
        if A > 0:
            b = min(cap_y, float(np.minimum(cap_m, (A ** fuel_pow) * fp1 * W[k]).mean()))
            A *= (1.0 - SURV_LOSS * b)
        out[t + 1] = A
    return out[AGES]


def loss(sim: np.ndarray, y0: np.ndarray) -> float:
    m = AGES >= 30
    return float(np.sqrt(np.mean(np.log((sim[m] + 0.1) / (y0[m] + 0.1)) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nofire", nargs="+", default=["work/edv3/sp_cxs_tc40b"])
    ap.add_argument("--sites", default="")
    ap.add_argument("--unburned", default="", help="comma list of sites without a no-fire run: their y0 profile is used as the no-fire curve")
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args()
    sites = [int(x) for x in open(ROOT / args.sites).read().split()] if args.sites else None
    data: dict[int, dict] = {}
    for r in args.nofire:
        for s, d in site_data(ROOT / r, sites).items():
            data.setdefault(s, d)
    for s in [int(x) for x in args.unburned.split(",") if x]:
        if s in data:
            data[s]["nf"] = data[s]["y0"].copy()
    names = sorted(data)
    base = {s: loss(simulate(data[s], 0.0, 8.33, 1e9, 1e9), data[s]["y0"]) for s in names}
    print("sites:", names)
    print("no-fire emulator vs y0 (log rms, ages >= 30): " + "  ".join(f"{s}:{base[s]:.2f}" for s in names))
    if not args.grid:
        return
    rows = []
    for fp1, thr, cap_m, cap_y in itertools.product((1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4), (2.3, 5.0, 8.33, 15.0, 1e9), (0.1, 0.25, 0.5, 1.0, 1e9), (0.2, 0.5, 1e9)):
        ls = [loss(simulate(data[s], fp1, thr, cap_m, cap_y), data[s]["y0"]) for s in names]
        rows.append((fp1, thr, cap_m, cap_y, float(np.sqrt(np.mean(np.square(ls)))), *ls))
    r = pd.DataFrame(rows, columns=["fp1", "thr", "cap_m", "cap_y", "rms", *[str(s) for s in names]])
    pd.set_option("display.width", 250)
    print(r.sort_values("rms").head(25).round(3).to_string(index=False))
    r.to_csv(ROOT / "work/fire/lookup_fire_grid.csv", index=False)


if __name__ == "__main__":
    main()
