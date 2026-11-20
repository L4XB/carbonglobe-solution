#!/usr/bin/env python
"""Per-site fire selection by the y0 fit, scored on training truth.

For every training site finished in both roots (fire off and fire on, same spin-up climate) pick the configuration
whose spin-up state matches the organisers' y0 better (mean over the 15 seed ages of |log((e+0.5)/(y0+0.5))| for
height plus |log((e+0.05)/(y0+0.05))| for agb, e = the run's year-1 state), then compare the window MSE of the
selected runs with always-off and always-on, by band and column, on the solution scale.
    python scripts/ed/train_select.py --off work/edv3/train_c400_s30 --on work/edv3/train_c400f_s30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ed")); sys.path.insert(0, str(ROOT / "scripts"))
from train_eval import BANDS, despike, hybrid  # noqa: E402
from ed_transfer import DIV  # noqa: E402


def load_root(root: Path, truth_root: Path) -> pd.DataFrame:
    rows = []
    for tf in sorted(truth_root.glob("*.csv")):
        s = int(tf.stem)
        f40, f0 = root / "results" / f"{s}_ssp126.csv", root / "results" / f"{s}_ssp126_y0.csv"
        if not (f40.exists() and f0.exists()):
            continue
        t, r, r0 = pd.read_csv(tf), pd.read_csv(f40), pd.read_csv(f0)
        rows.append(pd.DataFrame({"site": s, "age": t.age, "y0_h": t.y0_height, "y0_a": t.y0_agb, "y40_h": t.y40_height,
                                  "y40_a": t.y40_agb, "e0_h": r0.height_m, "e0_a": r0.agb_kgC, "x40_h": r.height_m, "x40_a": r.agb_kgC}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--off", default=str(ROOT / "work/edv3/train_c400_s30"))
    ap.add_argument("--on", default=str(ROOT / "work/edv3/train_c400f_s30"))
    ap.add_argument("--truth", default="", help="truth dir (default: <on>/truth)")
    args = ap.parse_args()
    truth = Path(args.truth) if args.truth else Path(args.on) / "truth"
    off, on = load_root(Path(args.off), truth), load_root(Path(args.on), truth)
    common = sorted(set(off.site) & set(on.site))
    off, on = off[off.site.isin(common)].reset_index(drop=True), on[on.site.isin(common)].reset_index(drop=True)
    fit = {}
    for nm, d in (("off", off), ("on", on)):
        e = np.abs(np.log((d.e0_h + 0.5) / (d.y0_h + 0.5))) + np.abs(np.log((d.e0_a + 0.05) / (d.y0_a + 0.05)))
        fit[nm] = e.groupby(d.site).mean()
    pick_on = (fit["on"] < fit["off"])
    print(f"{len(common)} common sites; fire ON chosen at {int(pick_on.sum())}: {list(pick_on[pick_on].index)}")
    sel = pd.concat([on[on.site.isin(pick_on[pick_on].index)], off[off.site.isin(pick_on[~pick_on].index)]]).sort_values(["site", "age"]).reset_index(drop=True)
    out = []
    for col, name in (("h", "height"), ("a", "agb")):
        div = DIV[name]
        for b, (lo, hi) in BANDS.items():
            res = {"col": name, "band": b}
            for nm, d in (("persist", off), ("off", off), ("on", on), ("select", sel)):
                m = ((d.age >= lo) & (d.age <= hi)).to_numpy()
                y0, y40 = d[f"y0_{col}"].to_numpy() / div, d[f"y40_{col}"].to_numpy() / div
                if nm == "persist":
                    res[nm] = float(np.mean((y40 - y0)[m] ** 2)); continue
                x40 = despike(d[f"x40_{col}"].to_numpy(), d[f"e0_{col}"].to_numpy(), d.site.to_numpy()) / div
                h = hybrid(x40, d[f"e0_{col}"].to_numpy() / div, y0, 2.0)
                res[nm] = float(np.mean((h - y40)[m] ** 2))
            out.append(res)
    print("hybrid (alpha 1, despiked) MSE on the solution scale; persist = y40 vs y0")
    print(pd.DataFrame(out).round(5).to_string(index=False))
    print("per-site y0 misfit (off / on):")
    print(pd.DataFrame({"off": fit["off"], "on": fit["on"]}).round(3).T.to_string())


if __name__ == "__main__":
    main()
