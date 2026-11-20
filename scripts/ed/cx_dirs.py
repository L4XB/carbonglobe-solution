#!/usr/bin/env python
"""C-centred probe directions from an exact-input ED root, with site splits the generic builder does not have.

ED prediction on the solution scale (truth = physical / 1.14114 in both columns): height raw x40, agb hybrid x40 * y0 / e0
(clip 0.5..2), age 1 and "cold" sites additive y0 + (x40 - e0); rows whose seed misses y0 by more than --gate (|log|, eps
0.1 m / 0.01 kgC) in that column are left at C. Site classes from ED's own old-stand agb (ages >= 150): burn = ratio x40 /
e0 below 0.9 in either scenario, else noburn; zones trop (|lat| < 30) / extra.
    --parts name=sel[,sel...]  with sel from: burn | noburn | trop | extra | cl | out | young | mid | old | ym | h | a | s126 | s585
                               e.g. fire_a=burn,trop,a   wet_ym_a=noburn,trop,ym,a   all_h=h
Writes work/dir_cx_<tag>_<name>.npy and prints E_all, sites, rows, mean D per block.
    python scripts/ed/cx_dirs.py --root work/edv3/cl_cx3/results --tag cx3 --parts fire_a=burn,trop,a wet_a=noburn,trop,a
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402
import ed_transfer as et  # noqa: E402

DIV = np.array([10.90645, 3.39567]); A = 1.0 / 1.14114; EPS = np.array([0.1, 0.01])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="work/edv3/cl_cx3/results")
    ap.add_argument("--tag", default="cx3")
    ap.add_argument("--gate", type=float, default=float(np.log(2.0)))
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--base", default="results/spatcorr_stack_20261106_s08.csv")
    ap.add_argument("--level-free", action="store_true")
    args = ap.parse_args()
    d = ROOT / args.root
    keys = shift.template_keys(); site = keys["site"].to_numpy(); age = keys["age"].to_numpy(); scen = keys["scenario"].to_numpy(); band = shift.band_of(age)
    C = shift.read_predictions(ROOT / args.base).astype(float)
    R = shift.read_predictions(ROOT / "results/latent_ref_y0_m114114.csv").astype(float)
    reads = {f: et.read_ed(d, keys, (A, A), form=f, R=R, smooth="despike") for f in ("raw", "hybrid", "add")}
    sim = reads["hybrid"][1]
    ex = np.load(ROOT / "work/test_cell_exact.npz"); lat = ex["lat"][site]
    cl = (np.load(ROOT / "work/collapse_cluster_labels.npy") == 0)[site]
    # ED's own year-0 and year-40 states (physical) per row
    import glob, re
    e0 = np.zeros_like(C); e40 = np.zeros_like(C)
    index = {(int(s), str(sc), int(a)): i for i, (s, sc, a) in enumerate(zip(site, scen, age))}
    for f in glob.glob(str(d / "*_ssp*.csv")):
        mm = re.match(r"(\d+)_(ssp\d+)(_y0)?\.csv$", Path(f).name)
        if not mm:
            continue
        tgt = e0 if mm.group(3) else e40
        for row in np.atleast_1d(np.genfromtxt(f, delimiter=",", names=True, dtype=float)):
            k = (int(mm.group(1)), mm.group(2), int(round(row["age"])))
            if k in index:
                tgt[index[k]] = [row["height_m"], row["agb_kgC"]]
    cold = (np.abs(lat) >= 45)
    ED = np.zeros_like(C)
    ED[:, 0] = np.where(cold, reads["add"][0][:, 0], reads["raw"][0][:, 0])
    ED[:, 1] = np.where(cold | (age == 1), reads["add"][0][:, 1], reads["hybrid"][0][:, 1])
    gate = np.abs(np.log((R * DIV + EPS) / (np.maximum(e0, 0.0) + EPS))) <= args.gate
    ns = site.max() + 1
    oldm = sim & (age >= 150)
    s0 = np.zeros((ns, 2)); s40 = np.zeros((ns, 2))
    for j, sc in enumerate(("ssp126", "ssp585")):
        m = oldm & (scen == sc)
        np.add.at(s0[:, j], site[m], e0[m, 1]); np.add.at(s40[:, j], site[m], e40[m, 1])
    ratio = np.where(s0 > 0, s40 / np.maximum(s0, 1e-9), 1.0)
    burn_site = (ratio < 0.9).any(axis=1)
    sel = {"burn": burn_site[site], "noburn": ~burn_site[site], "trop": np.abs(lat) < 30, "extra": np.abs(lat) >= 30, "cl": cl, "out": ~cl,
           "young": band == "young", "mid": band == "mid", "old": band == "old", "ym": band != "old", "s126": scen == "ssp126", "s585": scen == "ssp585"}
    print(f"{d.parent.name}: simulated sites {len(np.unique(site[sim]))} (burning by ED: {int(burn_site[np.unique(site[sim])].sum())}); gate pass on simulated rows h {gate[sim, 0].mean():.3f} a {gate[sim, 1].mean():.3f}")
    for spec in args.parts:
        name, toks = spec.split("=", 1)
        rows = sim.copy(); cols = [0, 1]
        for t in toks.split(","):
            if t == "h":
                cols = [0]
            elif t == "a":
                cols = [1]
            elif t == "ha":
                cols = [0, 1]
            else:
                rows &= sel[t]
        D = np.zeros_like(C)
        for c in cols:
            m = rows & gate[:, c]
            D[m, c] = ED[m, c] - C[m, c]
        if args.level_free:
            for a in np.unique(age):
                for sc in ("ssp126", "ssp585"):
                    for c in cols:
                        m = rows & gate[:, c] & (age == a) & (scen == sc)
                        if m.any():
                            D[m, c] -= D[m, c].mean()
        np.save(ROOT / f"work/dir_cx_{args.tag}_{name}.npy", D)
        nz = np.any(D != 0, axis=1)
        blocks = ", ".join(f"{b[0]}{'ha'[c]} {np.mean(D[(band == b) & nz, c]) if ((band == b) & nz).any() else 0.0:+.3f}" for b in ("young", "mid", "old") for c in cols)
        print(f"  {name:14s} [{toks}] sites {len(np.unique(site[nz])):4d} rows {int(nz.sum()):6d}  E_all {np.mean(D ** 2):.5f}  mean D per block: {blocks}  | rms D on its rows {np.sqrt(np.mean(D[nz] ** 2)) if nz.any() else 0:.3f}")


if __name__ == "__main__":
    main()
