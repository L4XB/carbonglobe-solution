#!/usr/bin/env python
"""Systematic exact-input ED correction for WET TROPICAL FOREST rows, applied to every such test site.

Site set W (no ED needed): |lat| < 30, decoded y0 old agb (ages >= 150) >= --min-agb kgC, exact twin whose training old agb
ratio y40 / y0 >= 0.95 and whose yearly ED fire dryness index never exceeded --max-di in 1982-2021 (never near burning).
On the W sites already simulated (root, default work/edv3/cl_cx3/results; additionally ED's own old ratio >= 0.95) take per
age x scenario x column the median t of log(ED / (C x 1.14114)) (agb: hybrid x40 * y0 / e0, age 1: additive form; height:
raw x40), minus ED's own bias against the organisers' truth on dense non-burning TRAINING cells (--train-bias, default
on). Direction D = C x (exp(t) - 1) on all W rows (solution scale; C is on the truth scale = physical / 1.14114).
Writes work/dir_cx_sys_<tag>.npy, work/cx_sys_table_<tag>.json and prints the table and energies.
    python scripts/ed/cx_sys_dir.py [--tag wet] [--root work/edv3/cl_cx3/results]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

DIV = np.array([10.90645, 3.39567]); M = 1.14114
AG = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500])
# ED minus the organisers' truth (log) on 6 dense non-burning training cells (11-09): agb ~0 from age 30 on, -0.02 at 10-20;
# height -0.007 .. -0.018. Subtracted from t so that the table estimates truth / C.
TRAIN_BIAS = {"agb": np.array([-0.010, -0.024, -0.020, -0.002, -0.001, 0.003, 0.008, -0.002, -0.004, 0.002, 0.004, -0.005, -0.009, -0.003, 0.000]),
              "height": np.array([-0.008, -0.007, 0.001, 0.001, -0.010, -0.003, -0.011, -0.016, -0.010, -0.004, -0.005, -0.009, -0.014, -0.018, -0.015])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="work/edv3/cl_cx3/results")
    ap.add_argument("--tag", default="wet")
    ap.add_argument("--min-agb", type=float, default=15.0)
    ap.add_argument("--max-di", type=float, default=50000.0)
    ap.add_argument("--no-train-bias", dest="train_bias", action="store_false")
    ap.add_argument("--base", default="results/spatcorr_stack_20261106_s08.csv")
    args = ap.parse_args()
    d = ROOT / args.root
    keys = shift.template_keys(); site = keys["site"].to_numpy(); age = keys["age"].to_numpy(); scen = keys["scenario"].to_numpy()
    C = shift.read_predictions(ROOT / args.base).astype(float)
    R = shift.read_predictions(ROOT / "results/latent_ref_y0_m114114.csv").astype(float) * DIV
    ns = site.max() + 1
    ex = np.load(ROOT / "work/test_cell_exact.npz"); lat = ex["lat"]
    tw = np.load(ROOT / "work/train_twin_states.npz"); tratio = np.zeros(ns)
    tratio[tw["sites"]] = np.maximum(tw["tr40"].mean(2)[:, 7:, 1].mean(1), 0.02) / np.maximum(tw["tr0"][:, 7:, 0, 1].mean(1), 0.02)
    z = np.load(ROOT / "work/cal/fire_di.npz"); dimax = np.full(ns, np.inf); dimax[z["sites"]] = z["di_train"].max(1)
    old = age >= 150
    Ro = np.zeros(ns); n = np.zeros(ns); np.add.at(Ro, site[old], R[old, 1]); np.add.at(n, site[old], 1); Ro /= np.maximum(n, 1)
    W = (np.abs(lat) < 30) & (Ro >= args.min_agb) & (tratio >= 0.95) & (dimax < args.max_di) & ex["exact"]
    done = sorted(set(int(f.name.split("_")[0]) for f in d.glob("*_ssp585.csv")) & set(int(f.name.split("_")[0]) for f in d.glob("*_ssp126.csv")))
    logs = {(sc, c): [] for sc in ("ssp126", "ssp585") for c in ("height", "agb")}
    n_w_done = 0; n_w_burn = 0; n_nonw_noburn = 0
    for s in done:
        if abs(lat[s]) >= 30:
            continue
        rec = {}; own_ok = True
        for sc in ("ssp126", "ssp585"):
            e40 = pd.read_csv(d / f"{s}_{sc}.csv").set_index("age").loc[AG]; e0 = pd.read_csv(d / f"{s}_{sc}_y0.csv").set_index("age").loc[AG]
            m = (site == s) & (scen == sc)
            a0, a40 = e0.agb_kgC.to_numpy(), e40.agb_kgC.to_numpy()
            if a40[AG >= 150].mean() < 0.95 * a0[AG >= 150].mean():
                own_ok = False
            hyb = a40 * np.clip(R[m, 1] / np.maximum(a0, 1e-6), 0.5, 2.0); add = np.maximum(R[m, 1] + a40 - a0, 1e-3)
            pa = np.where(AG == 1, add, hyb)
            rec[sc] = (np.log(np.maximum(e40.height_m.to_numpy(), 1e-3) / np.maximum(C[m, 0] * DIV[0] * M, 1e-3)), np.log(np.maximum(pa, 1e-3) / np.maximum(C[m, 1] * DIV[1] * M, 1e-3)))
        if W[s]:
            n_w_done += 1
            if not own_ok:
                n_w_burn += 1; continue
            for sc in rec:
                logs[(sc, "height")].append(rec[sc][0]); logs[(sc, "agb")].append(rec[sc][1])
        elif own_ok and Ro[s] >= args.min_agb:
            n_nonw_noburn += 1
    print(f"W: {int(W.sum())} wet tropical forest sites; simulated {n_w_done} (ED burns at {n_w_burn} of them); simulated non-W dense sites that do not burn in ED: {n_nonw_noburn}")
    table = {}
    for (sc, c), v in logs.items():
        v = np.array(v); t = np.median(v, axis=0); q1, q3 = np.percentile(v, 25, axis=0), np.percentile(v, 75, axis=0)
        if args.train_bias:
            t = t - TRAIN_BIAS[c]
        table[f"{sc}_{c}"] = dict(t=t.round(4).tolist(), iqr=((q3 - q1) / 2).round(4).tolist(), n=int(len(v)))
    print("age   " + "  ".join(f"{k:>14s}" for k in table))
    for j, a in enumerate(AG):
        print(f"{a:4d}  " + "  ".join(f"{table[k]['t'][j]:+.3f} (+-{table[k]['iqr'][j]:.3f})" for k in table))
    D = np.zeros_like(C)
    ai = np.searchsorted(AG, age)
    for sc in ("ssp126", "ssp585"):
        for col, c in ((0, "height"), (1, "agb")):
            t = np.array(table[f"{sc}_{c}"]["t"])
            m = W[site] & (scen == sc)
            D[m, col] = C[m, col] * (np.exp(t[ai[m]]) - 1.0)
    np.save(ROOT / f"work/dir_cx_sys_{args.tag}.npy", D)
    (ROOT / f"work/cx_sys_table_{args.tag}.json").write_text(json.dumps(dict(table=table, n_sites=int(W.sum()), ages=AG.tolist()), indent=1))
    band = shift.band_of(age)
    print(f"direction work/dir_cx_sys_{args.tag}.npy: E_all {np.mean(D ** 2):.5f}; by block (share): " +
          ", ".join(f"{b[0]}{'ha'[c]} {np.sum(D[band == b, c] ** 2) / max(np.sum(D ** 2), 1e-30):.2f}" for b in ("young", "mid", "old") for c in (0, 1)) +
          f"; scenario split 126 {np.sum(D[scen == 'ssp126'] ** 2) / max(np.sum(D ** 2), 1e-30):.2f} / 585 {np.sum(D[scen == 'ssp585'] ** 2) / max(np.sum(D ** 2), 1e-30):.2f}")
    for other in ("work/dir_ornl_qex05_qcontrast_ah_all.npy", "work/dir_rep_twin_avg3_lf.npy"):
        o = np.load(ROOT / other)
        print(f"   cos with {Path(other).name}: {np.sum(D * o) / np.sqrt(np.sum(D * D) * np.sum(o * o)):+.3f}")


if __name__ == "__main__":
    main()
