#!/usr/bin/env python
"""Day-46 readout of the fire-window ED probes (nested on C) and the joint cluster-agb fit with the 11-08 twin probe.

Single readouts: lambda_p = (E_pub,p - (s_p - s_C)) / (2 E_pub,p), i.e. b_p = E_pub[D_p (T - C)] = lambda_p E_pub,p.
Board scores carry 3 decimals (+- 0.0005), s_C = 0.0438 +- 0.0003.
Joint cluster agb: directions D_fire (cap-0.1 fire windows, cl_a of day 46) and D_twin (fire-off twin, F1sg of 11-08,
lambda -0.061, E_pub 0.01176). Gram matrix from all rows times the public ratio r (E_pub / E_all of each probe);
weights w = G^-1 b, predicted public gain b' G^-1 b.
    python scripts/ed/readout_day46.py [--scores name=score,...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ed")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import ed_transfer as et  # noqa: E402
from assemble_final import DIV, EPS, e0_rows  # noqa: E402
from cgb import shift  # noqa: E402

S_C = 0.0438
TWIN = dict(lam=-0.061, e_pub=0.01176)


def gated_d(ed_dir: Path, keys, C, R, col: int, rows: np.ndarray, h_raw: bool = False) -> np.ndarray:
    ED, sim, _ = et.read_ed(ed_dir, keys, (1.0, 1.0), form="hybrid", R=R, smooth="despike")
    if h_raw:
        W, _, _ = et.read_ed(ed_dir, keys, (1.0, 1.0), form="raw", R=R, smooth="despike"); ED[:, 0] = W[:, 0]
    e0 = e0_rows(ed_dir, keys)
    gate = np.abs(np.log((R * DIV + EPS) / (np.maximum(e0, 0.0) + EPS))) <= np.log(2.0)
    D = np.zeros_like(C)
    m = sim & rows & gate[:, col]
    D[m, col] = ED[m, col] - C[m, col]
    return D


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="", help="override: comma list probe=score (probe names without prefix)")
    ap.add_argument("--log", default=str(ROOT / "work" / "logs" / "build_day46_fw.log"))
    args = ap.parse_args()
    e_pub = {}
    for line in open(args.log):
        m = re.search(r"spat_ed46fw_(\w+)\.csv: .*E_all\[D\^2\] ([\d.]+) -> E_pub ([\d.]+)", line)
        if m:
            e_pub[m.group(1)] = (float(m.group(3)), float(m.group(2)))
    sc = json.load(open(ROOT / "results" / "board_scores.json"))
    scores = {k[len("spat_ed46fw_"):-4]: v for k, v in sc.items() if k.startswith("spat_ed46fw_")}
    if args.scores:
        scores.update({k: float(v) for k, v in (x.split("=") for x in args.scores.split(","))})
    lam = {}
    print(f"s_C {S_C}; probes with energy in the build log: {sorted(e_pub)}")
    for p, (ep, ea) in e_pub.items():
        if p not in scores:
            print(f"  {p}: no score yet (E_pub {ep:.5f})"); continue
        s = scores[p]; lam[p] = (ep - (s - S_C)) / (2 * ep); err = np.hypot(0.0005, 0.0003) / (2 * ep)
        print(f"  {p:12s} score {s:.3f}  E_pub {ep:.5f}  lambda {lam[p]:+.3f} +- {err:.3f}  -> applied alone at lambda: gain {lam[p] ** 2 * ep:.5f} (score {S_C - max(lam[p], 0) ** 2 * ep:.4f})")
    if "cl_a" in lam and "cl_old_a" in lam:
        ea, eo = e_pub["cl_a"][0], e_pub["cl_old_a"][0]; eym = ea - eo
        lam_ym = (lam["cl_a"] * ea - lam["cl_old_a"] * eo) / eym
        print(f"  cl_ym_a (by subtraction): E_pub {eym:.5f} lambda {lam_ym:+.3f} +- {np.hypot(0.0005, 0.0005) / (2 * eym):.3f}")
    if not any(k in lam for k in ("cl_a", "cl_a_v4")):
        return
    # joint fits on the cluster (all directions C-centred, cluster rows, each run with its own gate): agb with the
    # fire-off twin (11-08 readout) and the artefact-free rerun (cl_a_pr); heights with cl_h_pr
    keys = shift.template_keys(); site = keys["site"].to_numpy()
    rd = lambda p: shift.read_predictions(ROOT / p).astype(np.float64)
    C, R = rd("results/spatcorr_stack_20261106_s08.csv"), rd("results/latent_ref_y0_m114114.csv")
    cl = (np.load(ROOT / "work" / "collapse_cluster_labels.npy") == 0)[site]
    runs = {"cl_a": ("work/edv3/full_c400a/results", 1, False), "twin": ("work/edv3/cl_twin_nofire/results", 1, False),
            "cl_a_pr": ("work/edv3/cl_pr/results", 1, False), "cl_h": ("work/edv3/full_c400a/results", 0, True),
            "cl_h_pr": ("work/edv3/cl_pr/results", 0, True),
            "cl_a_v4": ("work/edv3/full_v4/results", 1, False), "cl_h_v4": ("work/edv3/full_v4/results", 0, True),
            "cl_a_v4asis": ("work/edv3/cl_v4asis/results", 1, False), "cl_h_v4asis": ("work/edv3/cl_v4asis/results", 0, True)}
    b_of = dict(lam_b={k: lam[k] * e_pub[k][0] for k in lam if k in e_pub})
    b_of["lam_b"]["twin"] = TWIN["lam"] * TWIN["e_pub"]
    epub_of = {k: v[0] for k, v in e_pub.items()}; epub_of["twin"] = TWIN["e_pub"]
    for group in (["cl_a_v4", "cl_a_v4asis"], ["cl_a_v4", "twin"], ["cl_a_v4", "cl_a_v4asis", "twin"], ["cl_a_v4", "cl_a_v4asis", "cl_a"], ["cl_a_v4", "cl_a"], ["cl_h_v4", "cl_h_v4asis"],
                  ["cl_a", "cl_a_pr", "twin"], ["cl_a", "cl_a_pr"], ["cl_a", "twin"], ["cl_h", "cl_h_pr"]):
        g = [k for k in group if k in b_of["lam_b"] and (ROOT / runs[k][0]).exists()]
        if len(g) < 2:
            continue
        Ds = [gated_d(ROOT / runs[k][0], keys, C, R, runs[k][1], cl, h_raw=runs[k][2]) for k in g]
        n = Ds[0].size
        G = np.array([[np.sum(a * b) for b in Ds] for a in Ds]) / n
        r = float(np.mean([epub_of[k] / G[i, i] for i, k in enumerate(g)]))
        b = np.array([b_of["lam_b"][k] for k in g])
        w = np.linalg.solve(r * G, b)
        corr = G[0, 1] / np.sqrt(G[0, 0] * G[1, 1])
        best_single = max(b_of["lam_b"][k] ** 2 / epub_of[k] for k in g)
        print(f"  joint {'+'.join(g)}: r {r:.3f}, corr({g[0]},{g[1]}) {corr:+.3f}; weights " + " ".join(f"{k} {wi:+.3f}" for k, wi in zip(g, w))
              + f"; public gain {float(b @ w):.5f} (best single {best_single:.5f}) -> score {S_C - float(b @ w):.4f}")


if __name__ == "__main__":
    main()
