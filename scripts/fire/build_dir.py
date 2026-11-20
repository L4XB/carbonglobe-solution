#!/usr/bin/env python
"""Probe directions from the fire emulator: old-stand agb of tropical forest sites under a fire-climate hypothesis.
    python scripts/fire/build_dir.py --hyp clim40 --group cluster --tag clim40_cl
  --hyp    clim40 | clim20 | clim10 | first10 | mer (twin MERRA years k -> k) | fut (released monthly climate)
  --group  cluster | prone | both      (prone = tropical forest outside the cluster where emulator, twin or C see fire)
Writes work/fire/dir_<tag>.npy (n_rows, 2) in coded units, D = prediction - C on the touched rows (old ages, agb only).
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

DIV = np.array([10.90645, 3.39567]); M = 1.14114
C_PATH = "results/spatcorr_stack_20261106_s08.csv"
PRM = dict(c=1.15e-5, cap=0.116)                      # fitted on the 1096 tropical forest twins (empirical regrowth)
G_SCEN = {"ssp126": 0.060, "ssp585": 0.112}           # old-agb growth of non-burning tropical forest in C (log)


def load_w(hyp: str) -> dict[str, np.ndarray]:
    z = np.load(ROOT / "work/fire/test_W.npz")
    if hyp == "mer":
        w = np.nan_to_num(z["W_mer"]); return {"ssp126": w, "ssp585": w}
    if hyp == "fut":
        return {"ssp126": z["W_fut"][0], "ssp585": z["W_fut"][1]}
    if hyp.startswith("relrule"):                     # as-delivered monthly climate + shipped wet-month rule (work/fire/test_W_rule.npz)
        z2 = np.load(ROOT / "work/fire/test_W_rule.npz"); thr = hyp[7:] or "8.33"
        return {"ssp126": z2[f"rel_{thr}_ssp126"], "ssp585": z2[f"rel_{thr}_ssp585"]}
    if hyp in ("rel", "relT", "relP"):                # monthly climate as released (no release transform undone) / T only / pr only
        w = np.load(ROOT / "work/fire/test_W_rel.npz")[hyp]; return {"ssp126": w[0], "ssp585": w[1]}
    key = {"clim40": "clim_40_yr", "clim20": "clim_last_20_yr", "clim10": "clim_last_10_yr", "first10": "clim_first_10_yr"}[hyp]
    w = np.repeat(np.load(ROOT / "work/fire/test_W_clim.npz")[key][:, None], 40, axis=1)
    return {"ssp126": w, "ssp585": w}


def groups() -> dict[str, np.ndarray]:
    k = shift.template_keys(); site = k.site.to_numpy(); age = k.age.to_numpy()
    R = shift.read_predictions("results/latent_ref_y0_m114114.csv").astype(float) * DIV
    z = np.load(ROOT / "work/fire/test_W.npz"); cell, lat = z["cell"], z["lat"]; ns = len(cell)
    old = age >= 150
    a0 = np.zeros(ns); n = np.zeros(ns); np.add.at(a0, site[old], R[old, 1]); np.add.at(n, site[old], 1); a0 /= np.maximum(n, 1)
    cl = np.load(ROOT / "work/collapse_cluster_labels.npy") == 0
    forest = (a0 > 5) & (np.abs(lat) < 30) & (cell >= 0)
    return dict(a0=a0, cl=cl & forest, forest=forest, lat=lat, cell=cell)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", default="clim40"); ap.add_argument("--group", default="both"); ap.add_argument("--tag", required=True)
    ap.add_argument("--c-scale", type=float, default=1.0, help="multiplies the ignition constant")
    ap.add_argument("--min-effect", type=float, default=0.05, help="prone = |emulated log ratio| or |C log ratio - growth| above this")
    ap.add_argument("--regrow", default="1,1", help="regrowth multipliers for ssp126,ssp585 (CO2 speeds up the recovery)")
    ap.add_argument("--aeq", default="1,1", help="equilibrium agb multipliers for ssp126,ssp585")
    ap.add_argument("--no-growth-offset", action="store_true", help="do not add G_SCEN (use when --aeq carries the CO2 effect)")
    args = ap.parse_args()
    k = shift.template_keys(); site = k.site.to_numpy(); age = k.age.to_numpy(); scen = k.scenario.to_numpy()
    C = shift.read_predictions(C_PATH).astype(float)
    R = shift.read_predictions("results/latent_ref_y0_m114114.csv").astype(float) * DIV
    g = groups(); a0 = g["a0"]; ns = len(a0)
    W = load_w(args.hyp)
    L = {}
    rg = dict(zip(("ssp126", "ssp585"), map(float, args.regrow.split(",")))); aq = dict(zip(("ssp126", "ssp585"), map(float, args.aeq.split(","))))
    for sc in ("ssp126", "ssp585"):
        a = np.zeros((ns, 41)); a[:, 0] = a0
        for kk in range(40):
            cur = a[:, kk]; r = np.minimum(PRM["cap"], PRM["c"] * args.c_scale * cur * W[sc][:, kk])
            a[:, kk + 1] = cur * np.exp(-r + em.regrowth(np.log(np.maximum(a0 * aq[sc], 1e-6) / np.maximum(cur, 1e-6)), rg[sc]))
        L[sc] = np.log(np.maximum(a[:, -1], 0.02) / np.maximum(a0, 0.02))
    if args.no_growth_offset:
        G_SCEN.update({"ssp126": 0.0, "ssp585": 0.0})
    old = age >= 150
    lc = {}
    for sc in ("ssp126", "ssp585"):
        m = old & (scen == sc); v = np.zeros(ns); n = np.zeros(ns)
        np.add.at(v, site[m], np.log(np.maximum(C[m, 1] * DIV[1] * M, 0.02) / np.maximum(R[m, 1], 0.02))); np.add.at(n, site[m], 1); lc[sc] = v / np.maximum(n, 1)
    fire_c = np.minimum(lc["ssp126"] - G_SCEN["ssp126"], lc["ssp585"] - G_SCEN["ssp585"])
    fire_e = np.minimum(L["ssp126"], L["ssp585"])
    prone = g["forest"] & ~g["cl"] & ((fire_e < -args.min_effect) | (fire_c < -2 * args.min_effect))
    sel = {"cluster": g["cl"], "prone": prone, "both": g["cl"] | prone}[args.group]
    D = np.zeros_like(C)
    for sc in ("ssp126", "ssp585"):
        m = old & (scen == sc) & sel[site]
        pred = R[m, 1] * np.exp(L[sc][site[m]] + G_SCEN[sc]) / (DIV[1] * M)
        D[m, 1] = pred - C[m, 1]
    out = ROOT / "work/fire" / f"dir_{args.tag}.npy"; np.save(out, D)
    le, lcc = L["ssp126"][sel], lc["ssp126"][sel] - G_SCEN["ssp126"]
    print(f"{args.tag}: {int(sel.sum())} sites, {int((np.abs(D).sum(1) > 0).sum())} rows; E_all {np.mean(D ** 2):.5f}; emulated fire log ratio mean {le.mean():+.3f} "
          f"(quartiles {np.percentile(le, [10, 25, 50, 75, 90]).round(2)}), C's {lcc.mean():+.3f} (quartiles {np.percentile(lcc, [10, 25, 50, 75, 90]).round(2)}); corr {np.corrcoef(le, lcc)[0, 1]:+.3f}; "
          f"sites raised {int((le - lcc > 0.1).sum())}, lowered {int((le - lcc < -0.1).sum())}")


if __name__ == "__main__":
    main()
