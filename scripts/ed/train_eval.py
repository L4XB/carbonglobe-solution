#!/usr/bin/env python
"""Score our ED on training sites against the organisers' trajectories (output of ``train_site.py``).

Per site and seed age: truth y0 / y40 (raw), our e0 (state one year into the window) and x40 (two runs of
identical forcing: the ``ssp126`` and ``ssp585`` files). On the solution scale (raw / DIV) it reports, by age
band and column:
  * persistence MSE (y40 vs y0) and the hybrid ED's MSE, hybrid = alpha * x40 * clip(y0 / e0, 1/cap, cap),
    with the least-squares alpha per column and alpha = 0.876;
  * the run-to-run noise (half the mean squared difference of the two runs) and the MSE of the two-run mean;
  * the share of the true change (y40 - y0) that the hybrid change explains (R^2 and slope).

    python scripts/ed/train_eval.py [--root work/edv3/train_c400f]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from ed_transfer import DIV  # noqa: E402

BANDS = {"young": (1, 30), "mid": (50, 150), "old": (200, 500)}


def load(root: Path) -> pd.DataFrame:
    rows = []
    for tf in sorted((root / "truth").glob("*.csv")):
        s = int(tf.stem)
        need = [root / "results" / f"{s}_{sc}{suf}.csv" for sc in ("ssp126", "ssp585") for suf in ("", "_y0")]
        if not all(p.exists() for p in need):
            continue
        t = pd.read_csv(tf)
        r1, r1y0, r2, r2y0 = (pd.read_csv(p) for p in need)
        for i in range(len(t)):
            rows.append({"site": s, "age": int(t.age[i]),
                         "y0_h": t.y0_height[i], "y0_a": t.y0_agb[i], "y40_h": t.y40_height[i], "y40_a": t.y40_agb[i],
                         "e0_h": r1y0.height_m[i], "e0_a": r1y0.agb_kgC[i], "x40_h": r1.height_m[i], "x40_a": r1.agb_kgC[i],
                         "e0b_h": r2y0.height_m[i], "e0b_a": r2y0.agb_kgC[i], "x40b_h": r2.height_m[i], "x40b_a": r2.agb_kgC[i]})
    return pd.DataFrame(rows)


def hybrid(x40: np.ndarray, e0: np.ndarray, y0: np.ndarray, cap: float) -> np.ndarray:
    fac = np.where(e0 > 0, np.clip(y0 / np.maximum(e0, 1e-12), 1.0 / cap, cap), 1.0)
    out = x40 * fac
    return np.where(y0 > 0, out, 0.0)                    # bare at the start stays bare (the test pipeline's rule)


def despike(x40: np.ndarray, e0: np.ndarray, site: np.ndarray, thr: float = np.log(2.0)) -> np.ndarray:
    """Replace log(x40 / e0) by the running median of 3 only where it deviates from that median by more than thr
    (isolated spikes from ED's run-to-run noise); every other row is left as it is."""
    out = x40.copy()
    for s in np.unique(site):
        idx = np.flatnonzero(site == s)
        lr = np.log((x40[idx] + 1e-6) / (e0[idx] + 1e-6))
        med = lr.copy()
        med[1:-1] = np.median(np.stack([lr[:-2], lr[1:-1], lr[2:]]), axis=0)
        bad = np.abs(lr - med) > thr
        lr[bad] = med[bad]
        out[idx] = (e0[idx] + 1e-6) * np.exp(lr) - 1e-6
    return np.maximum(out, 0.0)


def smooth_ratio(x40: np.ndarray, e0: np.ndarray, site: np.ndarray) -> np.ndarray:
    """x40 with log(x40 / e0) replaced by its running median of 3 over the 15 seed ages of each site (ends kept)."""
    out = x40.copy()
    for s in np.unique(site):
        idx = np.flatnonzero(site == s)                  # rows are in seed-age order per site
        lr = np.log((x40[idx] + 1e-6) / (e0[idx] + 1e-6))
        med = lr.copy()
        med[1:-1] = np.median(np.stack([lr[:-2], lr[1:-1], lr[2:]]), axis=0)
        out[idx] = (e0[idx] + 1e-6) * np.exp(med) - 1e-6
    return np.maximum(out, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "work" / "edv3" / "train_c400f"))
    ap.add_argument("--cap", type=float, default=2.0)
    ap.add_argument("--common", default="", help="comma list of other roots: keep only sites finished in all of them")
    ap.add_argument("--cols", default="", help="comma list of output columns to print (default: all)")
    args = ap.parse_args()
    d = load(Path(args.root))
    if args.common:
        keep = set(d.site)
        for r in args.common.split(","):
            keep &= {int(f.stem) for f in (Path(r) / "truth").glob("*.csv")
                     if all((Path(r) / "results" / f"{f.stem}_{sc}{suf}.csv").exists() for sc in ("ssp126", "ssp585") for suf in ("", "_y0"))}
        d = d[d.site.isin(keep)].reset_index(drop=True)
    if d.empty:
        raise SystemExit("no finished training sites yet")
    print(f"{d.site.nunique()} training sites, {len(d)} site-age rows")
    d["band"] = "?"
    for b, (lo, hi) in BANDS.items():
        d.loc[(d.age >= lo) & (d.age <= hi), "band"] = b
    lines = []
    for col, name in (("h", "height"), ("a", "agb")):
        div = DIV[name]
        y0, y40 = d[f"y0_{col}"].to_numpy() / div, d[f"y40_{col}"].to_numpy() / div
        e0, x40, x40b = d[f"e0_{col}"].to_numpy() / div, d[f"x40_{col}"].to_numpy() / div, d[f"x40b_{col}"].to_numpy() / div
        h1 = hybrid(x40, e0, y0, args.cap); h2 = hybrid(x40b, e0, y0, args.cap); hm = 0.5 * (h1 + h2)
        site = d.site.to_numpy()
        hs = hybrid(smooth_ratio(x40, e0, site), e0, y0, args.cap); hs2 = hybrid(smooth_ratio(x40b, e0, site), e0, y0, args.cap)
        hd = hybrid(despike(x40, e0, site), e0, y0, args.cap); hd2 = hybrid(despike(x40b, e0, site), e0, y0, args.cap)
        a_ls = float(h1 @ y40 / max(h1 @ h1, 1e-30))
        for b in ("all",) + tuple(BANDS):
            m = np.ones(len(d), bool) if b == "all" else (d.band == b).to_numpy()
            dt, dh = (y40 - y0)[m], (a_ls * h1 - y0)[m]
            r2 = 1 - np.mean((dt - dh) ** 2) / max(np.mean((dt - dt.mean()) ** 2), 1e-30)
            slope = float(dh @ dt / max(dh @ dh, 1e-30))
            a_b = float(h1[m] @ y40[m] / max(h1[m] @ h1[m], 1e-30))
            lines.append({"col": name, "band": b, "n": int(m.sum()), "alpha_band": a_b,
                          "mse_hyb_aband": np.mean((a_b * h1 - y40)[m] ** 2),
                          "mse_persist": np.mean((y40 - y0)[m] ** 2),
                          "mse_hyb_a1": np.mean((h1 - y40)[m] ** 2),
                          "mse_hyb_0876": np.mean((0.876 * h1 - y40)[m] ** 2),
                          "mse_hyb_als": np.mean((a_ls * h1 - y40)[m] ** 2),
                          "mse_hyb_2run_als": np.mean((a_ls * hm - y40)[m] ** 2),
                          "mse_hyb_smooth_als": np.mean((a_ls * hs - y40)[m] ** 2),
                          "noise_var": 0.5 * np.mean((h1 - h2)[m] ** 2),
                          "noise_var_smooth": 0.5 * np.mean((hs - hs2)[m] ** 2),
                          "mse_hyb_despike_a1": np.mean((hd - y40)[m] ** 2),
                          "noise_var_despike": 0.5 * np.mean((hd - hd2)[m] ** 2),
                          "mse_raw_x40": np.mean((x40 - y40)[m] ** 2),
                          "chg_R2": r2, "chg_slope": slope, "alpha_ls": a_ls,
                          "e0_logratio_rms": float(np.sqrt(np.mean(np.log((e0[m] + 0.01) / (y0[m] + 0.01)) ** 2)))})
    t = pd.DataFrame(lines)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    if args.cols:
        t = t[["col", "band", "n"] + [c for c in args.cols.split(",") if c in t.columns]]
    print(t.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
