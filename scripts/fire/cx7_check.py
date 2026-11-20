#!/usr/bin/env python
"""Old-stand agb outcome of an exact-input ED root with the truth-calibrated fire against C, the emulator and the twin.
    python scripts/fire/cx7_check.py work/edv3/cl_cx7/results
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "fire"))
from cgb import shift  # noqa: E402
import build_dir as bd  # noqa: E402

DIV = bd.DIV; M = bd.M
AG = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500])


def main() -> None:
    d = Path(sys.argv[1])
    k = shift.template_keys(); site = k.site.to_numpy(); age = k.age.to_numpy(); scen = k.scenario.to_numpy()
    C = shift.read_predictions(bd.C_PATH).astype(float) * DIV * M
    R = shift.read_predictions("results/latent_ref_y0_m114114.csv").astype(float) * DIV
    g = bd.groups(); ns = len(g["a0"])
    done = sorted(set(int(f.name.split("_")[0]) for f in d.glob("*_ssp585.csv")) & set(int(f.name.split("_")[0]) for f in d.glob("*_ssp126.csv")))
    tw = np.load(ROOT / "work/train_twin_states.npz"); tr = np.full(ns, np.nan)
    tr[tw["sites"]] = np.log(np.maximum(tw["tr40"].mean(2)[:, 7:, 1].mean(1), 0.02) / np.maximum(tw["tr0"][:, 7:, 0, 1].mean(1), 0.02))
    em_gs = {}
    Dg = np.load(ROOT / "work/fire/dir_mer_gs.npy")
    rows = []
    for s in done:
        rec = {"site": s}
        for sc in ("ssp126", "ssp585"):
            e40 = pd.read_csv(d / f"{s}_{sc}.csv").set_index("age").loc[AG]; e0 = pd.read_csv(d / f"{s}_{sc}_y0.csv").set_index("age").loc[AG]
            m = (site == s) & (scen == sc)
            o = AG >= 150; mid = (AG >= 50) & (AG < 150); yg = (AG >= 10) & (AG < 50)
            for bn, bm in (("old", o), ("mid", mid), ("young", yg)):
                rec[f"ed_{bn}_{sc}"] = np.log(max(e40.agb_kgC.to_numpy()[bm].mean(), 0.02) / max(e0.agb_kgC.to_numpy()[bm].mean(), 0.02))
                rec[f"c_{bn}_{sc}"] = np.log(max(C[m, 1][bm].mean(), 0.02) / max(R[m, 1][bm].mean(), 0.02))
                rec[f"edh_{bn}_{sc}"] = e40.height_m.to_numpy()[bm].mean() - e0.height_m.to_numpy()[bm].mean()
                rec[f"ch_{bn}_{sc}"] = C[m, 0][bm].mean() - R[m, 0][bm].mean()
            rec[f"seed_{sc}"] = np.log(max(e0.agb_kgC.to_numpy()[o].mean(), 0.02) / max(R[m, 1][o].mean(), 0.02))
            rec[f"em_{sc}"] = np.log(max(((C[m, 1] / (DIV[1] * M) + Dg[m, 1]) * DIV[1] * M)[o].mean(), 0.02) / max(R[m, 1][o].mean(), 0.02)) if np.any(Dg[m, 1] != 0) else np.nan
        rec["cl"] = bool(g["cl"][s]); rec["forest"] = bool(g["forest"][s]); rec["twin"] = tr[s]
        rows.append(rec)
    T = pd.DataFrame(rows)
    print(f"{d.parent.name}: {len(T)} sites done ({int(T.cl.sum())} cluster forest sites)")
    for nm, sel in (("cluster forest", T.cl), ("other forest", T.forest & ~T.cl)):
        t = T[sel]
        if len(t) < 5:
            continue
        print(f"== {nm}: n {len(t)}; seed log(e0 / y0) old agb median {t.seed_ssp126.median():+.3f} (IQR/2 {np.subtract(*np.percentile(t.seed_ssp126, [75, 25])) / 2:.3f})")
        for bn in ("old", "mid", "young"):
            for sc in ("ssp126", "ssp585"):
                e, c = t[f"ed_{bn}_{sc}"], t[f"c_{bn}_{sc}"]
                extra = f"  emulator mer_gs {t[f'em_{sc}'].mean():+.3f} (corr with ED {np.corrcoef(e[t[f'em_{sc}'].notna()], t[f'em_{sc}'].dropna())[0, 1]:+.2f})  training twin {t.twin.mean():+.3f} (corr with ED {np.corrcoef(e[t.twin.notna()], t.twin.dropna())[0, 1]:+.2f})" if bn == "old" else ""
                print(f"   {bn:5s} {sc}: agb log ratio ED {e.mean():+.3f}  C {c.mean():+.3f}  corr {np.corrcoef(e, c)[0, 1]:+.2f}  rms diff {np.sqrt(np.mean((e - c) ** 2)):.3f} | height change ED {t[f'edh_{bn}_{sc}'].mean():+.2f} m  C {t[f'ch_{bn}_{sc}'].mean():+.2f} m  corr {np.corrcoef(t[f'edh_{bn}_{sc}'], t[f'ch_{bn}_{sc}'])[0, 1]:+.2f}{extra}")
        print(f"   scenario contrast old agb (585 - 126): ED {np.mean(t.ed_old_ssp585 - t.ed_old_ssp126):+.3f}  C {np.mean(t.c_old_ssp585 - t.c_old_ssp126):+.3f}")


if __name__ == "__main__":
    main()
