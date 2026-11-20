#!/usr/bin/env python
"""Site-level pattern of an exact-input ED root against the stack C, its board-driven part C - B and the twins.

Per site and scenario: old-stand (ages >= 150) mean agb in real units: R (y0), ED hybrid, C x 1.14114, B x 1.14114,
the twin's training y0 -> y40. Reports, for all done sites and for groups (cluster, other tropics by ED burn status,
extratropics): mean log ratios to R, correlations of log(ED / R) with log(C / R), of log(ED / B) with log(C / B)
(does ED point where the board moved the stack?), and the same for the twin's training ratio. Also the gate statistics
(seed mismatch). Usage:
    python scripts/ed/cx_pattern_check.py work/edv3/cl_cx3/results [--band old|mid|young]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

DIV = np.array([10.90645, 3.39567]); M = 1.14114
d = Path(sys.argv[1])
bname = sys.argv[sys.argv.index("--band") + 1] if "--band" in sys.argv else "old"
keys = shift.template_keys(); site = keys["site"].to_numpy(); age = keys["age"].to_numpy(); scen = keys["scenario"].to_numpy()
C = shift.read_predictions(ROOT / "results/spatcorr_stack_20261106_s08.csv").astype(float) * DIV * M
B = shift.read_predictions(ROOT / "results/stack_20261010.csv").astype(float) * DIV * M
R = shift.read_predictions(ROOT / "results/latent_ref_y0_m114114.csv").astype(float) * DIV
tw = np.load(ROOT / "work/train_twin_states.npz"); tr0 = tw["tr0"][:, :, 0, :].astype(float); tr40 = tw["tr40"].mean(2).astype(float)
idx = {int(s): i for i, s in enumerate(tw["sites"])}
ex = np.load(ROOT / "work/test_cell_exact.npz"); cl = np.load(ROOT / "work/collapse_cluster_labels.npy") == 0
AG = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500])
bm = {"old": AG >= 150, "mid": np.isin(AG, [50, 70, 100]), "young": np.isin(AG, [10, 20, 30])}[bname]
done = sorted(set(int(f.name.split("_")[0]) for f in d.glob("*_ssp585.csv")) & set(int(f.name.split("_")[0]) for f in d.glob("*_ssp126.csv")))
rows = []
for s in done:
    if s not in idx:
        continue
    for sc in ("ssp126", "ssp585"):
        try:
            e40 = pd.read_csv(d / f"{s}_{sc}.csv").set_index("age").loc[AG]; e0 = pd.read_csv(d / f"{s}_{sc}_y0.csv").set_index("age").loc[AG]
        except (FileNotFoundError, KeyError):
            continue
        m = (site == s) & (scen == sc)
        a0, a40 = e0.agb_kgC.to_numpy(), e40.agb_kgC.to_numpy()
        hyb = a40 * np.clip(R[m, 1] / np.maximum(a0, 1e-6), 0.5, 2.0)
        own = a40[bm].mean() / max(a0[bm].mean(), 1e-6)
        seed = np.abs(np.log((a0[bm] + 0.01) / (R[m, 1][bm] + 0.01))).mean()
        i = idx[s]
        rows.append(dict(site=s, scen=sc, lat=float(ex["lat"][s]), cl=bool(cl[s]), R=R[m, 1][bm].mean(), ED=hyb[bm].mean(), C=C[m, 1][bm].mean(), B=B[m, 1][bm].mean(),
                         t0=tr0[i, bm, 1].mean(), t40=tr40[i, bm, 1].mean(), own=own, seed=seed,
                         EDh=e40.height_m.to_numpy()[bm].mean(), Ch=C[m, 0][bm].mean(), Rh=R[m, 0][bm].mean()))
df = pd.DataFrame(rows)
df = df[(df.R > 1.0)]
df["burn"] = df.own < 0.9
L = lambda a, b: np.log(np.maximum(a, 0.02) / np.maximum(b, 0.02))
df["lED"], df["lC"], df["lB"], df["lT"] = L(df.ED, df.R), L(df.C, df.R), L(df.B, df.R), L(df.t40, df.t0)
print(f"{d.parent.name}: band {bname}, {df.site.nunique()} forest sites (R > 1 kgC), {len(df)} site-scenarios; seed mismatch mean |log| {df.seed.mean():.3f}, share < 0.1: {(df.seed < 0.1).mean():.2f}")
groups = {"all": np.ones(len(df), bool), "cluster": df.cl.to_numpy(), "tropics non-cluster, ED burns": (~df.cl & (df.lat.abs() < 30) & df.burn).to_numpy(),
          "tropics non-cluster, ED no burn": (~df.cl & (df.lat.abs() < 30) & ~df.burn).to_numpy(), "extratropics": (df.lat.abs() >= 30).to_numpy(),
          "cluster, ED burns": (df.cl & df.burn).to_numpy(), "cluster, ED no burn": (df.cl & ~df.burn).to_numpy()}
cc = lambda a, b: np.corrcoef(a, b)[0, 1] if len(a) > 3 and np.std(a) > 0 and np.std(b) > 0 else np.nan
for nm, g in groups.items():
    x = df[g]
    if len(x) < 4:
        continue
    print(f"  {nm:32s} n {len(x):4d} ({x.site.nunique()} sites) | mean log ratio to R: ED {x.lED.mean():+.3f}  C {x.lC.mean():+.3f}  B {x.lB.mean():+.3f}  twin {x.lT.mean():+.3f} | corr with C: ED {cc(x.lED, x.lC):+.2f}  twin {cc(x.lT, x.lC):+.2f} | "
          f"corr with C - B: (ED - B) {cc(x.lED - x.lB, x.lC - x.lB):+.2f}  (twin - B) {cc(x.lT - x.lB, x.lC - x.lB):+.2f} | rms log(ED / C) {np.sqrt(np.mean((x.lED - x.lC) ** 2)):.3f}  twin {np.sqrt(np.mean((x.lT - x.lC) ** 2)):.3f}  B {np.sqrt(np.mean((x.lB - x.lC) ** 2)):.3f} | height: median log(ED / C) {np.median(np.log(x.EDh / x.Ch)):+.3f}")
