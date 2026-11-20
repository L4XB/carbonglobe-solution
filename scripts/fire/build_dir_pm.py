#!/usr/bin/env python
"""Old-agb probe directions of the board-preferred fire structure (per-month cap, wet-month rule on the as-delivered
precipitation, as-delivered monthly climate): work/fire/dir_pm_<tag>_{cl,out,both}.npy (D = prediction - C).
    python scripts/fire/build_dir_pm.py --c 30 --thr 8.33 --cap 0.30 --tag a
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "fire"))
from cgb import shift  # noqa: E402
import emulator as em  # noqa: E402
import build_dir as bd  # noqa: E402

DIV = bd.DIV; M = bd.M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=float, default=30.0); ap.add_argument("--thr", type=float, default=8.33); ap.add_argument("--cap", type=float, default=0.30)
    ap.add_argument("--regrow", default="1.1,1.2"); ap.add_argument("--aeq", default="1.05,1.10"); ap.add_argument("--growth", default="0.03,0.05")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    k = shift.template_keys(); site = k.site.to_numpy(); age = k.age.to_numpy(); scen = k.scenario.to_numpy()
    C = shift.read_predictions(bd.C_PATH).astype(float); R = shift.read_predictions("results/latent_ref_y0_m114114.csv").astype(float) * DIV
    g = bd.groups(); a0 = g["a0"]; ns = len(a0)
    z = np.load(ROOT / "work/fire/test_xmonth_rel_forest.npz"); fs = z["sites"]
    rg = list(map(float, args.regrow.split(","))); aq = list(map(float, args.aeq.split(","))); gr = list(map(float, args.growth.split(",")))
    L = {}
    for j, sc in enumerate(("ssp126", "ssp585")):
        xm = z["x126" if sc == "ssp126" else "x585"]; P = z["p126" if sc == "ssp126" else "p585"]
        a = a0[fs].copy(); passm = P < args.thr
        for kk in range(40):
            r = np.minimum(args.cap, bd.PRM["c"] * args.c * a[:, None] * xm[:, kk, :] * passm[:, kk, :]).mean(1)
            a = a * np.exp(-r + em.regrowth(np.log(np.maximum(a0[fs] * aq[j], 1e-6) / np.maximum(a, 1e-6)), rg[j]))
        v = np.full(ns, np.nan); v[fs] = np.log(np.maximum(a, 0.02) / np.maximum(a0[fs], 0.02)) + gr[j]; L[sc] = v
    np.savez(ROOT / f"work/fire/pm_{args.tag}_logratio.npz", ssp126=L["ssp126"], ssp585=L["ssp585"])
    old = age >= 150
    lc = {}
    for sc in L:
        m = old & (scen == sc); v = np.zeros(ns); n = np.zeros(ns)
        np.add.at(v, site[m], np.log(np.maximum(C[m, 1] * DIV[1] * M, 0.02) / np.maximum(R[m, 1], 0.02))); np.add.at(n, site[m], 1); lc[sc] = v / np.maximum(n, 1)
    fire = np.isfinite(L["ssp126"]) & ((np.minimum(L["ssp126"] - gr[0], L["ssp585"] - gr[1]) < -0.03) | (np.minimum(lc["ssp126"], lc["ssp585"]) < -0.05))
    for nm, sel in (("cl", g["cl"] & fire), ("out", ~g["cl"] & fire), ("both", fire)):
        D = np.zeros_like(C)
        for sc in L:
            m = old & (scen == sc) & sel[site]
            D[m, 1] = R[m, 1] * np.exp(L[sc][site[m]]) / (DIV[1] * M) - C[m, 1]
        np.save(ROOT / f"work/fire/dir_pm_{args.tag}_{nm}.npy", D)
        e1, c1 = L["ssp126"][sel], lc["ssp126"][sel]
        print(f"pm_{args.tag}_{nm}: {int(sel.sum())} sites; E_all {np.mean(D ** 2):.5f}; emulated old agb log ratio {e1.mean():+.3f} / {L['ssp585'][sel].mean():+.3f}, C {c1.mean():+.3f} / {lc['ssp585'][sel].mean():+.3f}; corr {np.corrcoef(e1, c1)[0, 1]:+.3f}; raised {int((e1 - c1 > 0.1).sum())}, lowered {int((e1 - c1 < -0.1).sum())}")


if __name__ == "__main__":
    main()
