#!/usr/bin/env python
"""Compare ED fire variants on the same training sites, per site group: gated MSE (normalized units) of the readout
forms used on the test (height raw x40 at fire-window sites, agb hybrid x40 * clip(y0/e0, 0.5, 2)), persistence, and
the median old-band agb ratio (truth vs ED).

    python scripts/ed/fire_variant_eval.py \
        --groups random=35845,...  --roots fp1_cap10=work/edv3/train_fwo_fmax10_rand,work/edv3/train_fwo_fmax10_trop ...
Each --roots entry is name=comma list of roots (the union of their sites is used).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_eval import load  # noqa: E402

DIV = {"h": 10.90645, "a": 3.39567}
EPS = {"h": 0.1, "a": 0.01}


def preds(d: pd.DataFrame, c: str, form: str) -> tuple[np.ndarray, np.ndarray]:
    """Both scenarios' gated predictions against the same truth row (n, 2)."""
    y0 = d[f"y0_{c}"].to_numpy(); out = []
    for sx in ("", "b"):
        e0, x40 = np.maximum(d[f"e0{sx}_{c}"].to_numpy(), 0.0), d[f"x40{sx}_{c}"].to_numpy()
        g = np.abs(np.log((y0 + EPS[c]) / (e0 + EPS[c]))) <= np.log(2.0)
        p = x40 if form == "raw" else np.where(y0 > 0, x40 * np.where(e0 > 0, np.clip(y0 / np.maximum(e0, 1e-12), 0.5, 2.0), 1.0), 0.0)
        out.append(np.where(g, p, y0))
    return np.stack(out, axis=1), y0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", required=True, help="name=site,site,...")
    ap.add_argument("--roots", nargs="+", required=True, help="name=root,root,...")
    args = ap.parse_args()
    groups = {g.split("=")[0]: [int(s) for s in g.split("=")[1].split(",")] for g in args.groups}
    frames = {}
    for r in args.roots:
        name, paths = r.split("=")
        frames[name] = pd.concat([load(Path(p)) for p in paths.split(",") if Path(p).exists()], ignore_index=True).drop_duplicates(["site", "age"])
    pd.set_option("display.width", 220)
    for gname, sites in groups.items():
        common = set(sites)
        for d in frames.values():
            common &= set(d.site)
        if not common:
            print(f"== {gname}: no common sites"); continue
        rows = []
        for name, d in frames.items():
            d = d[d.site.isin(common)].sort_values(["site", "age"]).reset_index(drop=True)
            band = np.where(d.age <= 30, "young", np.where(d.age <= 150, "mid", "old"))
            row = {"variant": name}
            for c, form in (("h", "raw"), ("a", "hyb")):
                P, y0 = preds(d, c, form); y40 = d[f"y40_{c}"].to_numpy()
                for b in ("young", "mid", "old"):
                    m = band == b
                    row[f"{c}_{b}"] = float(np.mean(((P[m] - y40[m, None]) / DIV[c]) ** 2))
                row[f"{c}_all"] = float(np.mean(((P - y40[:, None]) / DIV[c]) ** 2))
            o = d[band == "old"]
            row["oa_ratio_ED"] = float(np.median(o.x40_a / o.e0_a.clip(lower=1e-6)))
            rows.append(row)
        d0 = frames[next(iter(frames))]; d0 = d0[d0.site.isin(common)].sort_values(["site", "age"]).reset_index(drop=True)
        band = np.where(d0.age <= 30, "young", np.where(d0.age <= 150, "mid", "old"))
        pers = {"variant": "persistence"}
        for c in ("h", "a"):
            for b in ("young", "mid", "old"):
                m = band == b
                pers[f"{c}_{b}"] = float(np.mean(((d0[f"y0_{c}"] - d0[f"y40_{c}"])[m] / DIV[c]) ** 2))
            pers[f"{c}_all"] = float(np.mean(((d0[f"y0_{c}"] - d0[f"y40_{c}"]) / DIV[c]) ** 2))
        o = d0[band == "old"]; pers["oa_ratio_ED"] = float(np.median(o.y40_a / o.y0_a.clip(lower=1e-6)))   # truth ratio
        rows.append(pers)
        print(f"== {gname}: {len(common)} common sites (persistence row: oa_ratio = TRUTH median old agb ratio)")
        print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
