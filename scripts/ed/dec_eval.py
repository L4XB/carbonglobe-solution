#!/usr/bin/env python
"""Exact-input ED against the board-calibrated stack C, read from the DECEMBER rows (the organisers' yearly record).

Reads <root>/csv/site*.csv (run_site.py with the pre_* / end_* columns), builds the ED prediction on the solution scale
(truth = physical / 1.14114, divided by the metric's 10.90645 m / 3.39567 kgC):
    height  raw December height of year 40, or the hybrid end * y0 / pre (--h hybrid, clip 0.67..1.5)
    agb     hybrid end * (y0 + eps) / (pre + eps) (clip 0.5..2); additive y0 + (end - pre) for seed age 1
    gate    rows whose first record misses test_y0 by more than ln 2 in that column stay at C
and prints, per climate stratum x age band x column, the median log ratio ED / C, its robust scatter and the energy of
ED - C (mean squared difference over ALL entries of the submission). `--rows jan` reads the old January-row columns.

    python scripts/ed/dec_eval.py name=work/edv3/sz_cx12s [name2=work/edv3/sz_cx0 ...] [--write] [--sites file]
--write saves work/dir_dec_<name>_<part>.npy (parts: all, plus one per stratum) for the board screen / probes.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

DIV = np.array([10.90645, 3.39567]); A = 1.0 / 1.14114; EPS = np.array([0.1, 0.01])
STRATA = ("boreal", "cool", "warm", "moist", "dry", "cluster")


def strata_of_sites() -> np.ndarray:
    """Climate stratum per test site (real mean T / annual P of the first window decade), 'cluster' for the collapse cluster."""
    z = np.load(ROOT / "work/edv3/site_zone_info.npz"); T, P = z["T"], z["P"]
    cl = np.load(ROOT / "work/collapse_cluster_labels.npy") == 0
    s = np.where(T < 3, "boreal", np.where(T < 10, "cool", np.where(T < 18, "warm", np.where(P >= 1000, "moist", "dry")))).astype(object)
    s[cl] = "cluster"
    return s


def load_root(root: Path, rows: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(root / "csv" / "site*.csv")))
    if not fs:
        return pd.DataFrame(columns=["site", "scenario", "age", "y0_height", "y0_agb", "e40_h", "e40_a", "e0_h", "e0_a"])
    t = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    if rows == "dec":
        need = ("end_h_m12", "end_agb_m12", "pre_h_m12", "pre_agb_m12")
        t = t[t[list(need)].notna().all(axis=1)] if all(c in t.columns for c in need) else t.iloc[0:0]
        t = t.assign(e40_h=t.end_h_m12, e40_a=t.end_agb_m12, e0_h=t.pre_h_m12, e0_a=t.pre_agb_m12)
    else:
        t = t.assign(e40_h=t.ed40_height_m, e40_a=t.ed40_agb_kgC, e0_h=t.y0est_height_m, e0_a=t.y0est_agb_kgC)
    return t


def predict(t: pd.DataFrame, keys: pd.DataFrame, C: np.ndarray, h_mode: str, gate: float, gate_lo: float = 0.0, agb_factor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """ED prediction (normalised solution scale) on the template rows; `sim` marks rows changed against C per column."""
    index = {(int(s), str(sc), int(a)): i for i, (s, sc, a) in enumerate(zip(keys.site, keys.scenario, keys.age))}
    ii = np.array([index[(int(s), str(sc), int(a))] for s, sc, a in zip(t.site, t.scenario, t.age)])
    f = lambda c: t[c].to_numpy(dtype=float)
    y0 = np.column_stack([f("y0_height"), f("y0_agb") * agb_factor])   # physical (decoded x 1.14114; agb_factor = extra agb decode factor,
    # 11-10: with the cycled lookup at 415 ppm the first record matches y0 heights to 1-2 % while the agb decoded with 1.14114 is a
    # uniform 5.4 % low in the moist and warm strata -> agb is coded with ~1 / 1.204; the hybrid prediction does not depend on it)
    e0 = np.column_stack([f("e0_h"), f("e0_a")]); e40 = np.column_stack([f("e40_h"), f("e40_a")])
    ratio = np.clip((y0 + EPS) / (np.maximum(e0, 0.0) + EPS), [0.67, 0.5], [1.5, 2.0])
    hyb = e40 * ratio
    add = np.maximum(y0 + (e40 - e0), 0.0)
    pred = hyb.copy()
    if h_mode == "raw":
        pred[:, 0] = e40[:, 0]
    age1 = t.age.to_numpy() == 1
    pred[age1, 1] = add[age1, 1]
    if h_mode != "raw":
        pred[age1, 0] = add[age1, 0]
    mis = np.abs(np.log((y0 + EPS) / (np.maximum(e0, 0.0) + EPS)))
    ok = (mis <= gate) & (mis >= gate_lo)
    # age-1 rows (11-10): the organisers' age-1 start state is one growth year younger than ours (y0 agb ~1/3 of our first
    # record at every forest site), a structural offset that the log gate rejects although year 40 hardly depends on it
    # (bare ground + 41 vs 42 years): they pass whenever both start states are small against the simulated year-40 value
    small = age1[:, None] & (np.maximum(e0, y0) <= 0.35 * np.maximum(e40, 1e-9))
    ok = np.where(small, gate_lo == 0.0, ok)
    ok &= np.isfinite(pred)
    P = C.copy(); sim = np.zeros(C.shape, dtype=bool)
    val = pred * np.array([A, A / agb_factor]) / DIV
    P[ii] = np.where(ok, val, C[ii]); sim[ii] = ok
    return P, sim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="name=root dir")
    ap.add_argument("--rows", default="dec", choices=("dec", "jan"))
    ap.add_argument("--h", default="raw", choices=("raw", "hybrid"))
    ap.add_argument("--gate", type=float, default=float(np.log(2.0)), help="rows whose first record misses test_y0 by more than this |log| (per column) stay at C; 0.07 = well-seeded rows only")
    ap.add_argument("--gate-lo", type=float, default=0.0, help="lower bound of the seed mismatch (to isolate the badly seeded rows)")
    ap.add_argument("--suffix", default="", help="appended to the name in the written file names")
    ap.add_argument("--agb-factor", type=float, default=1.055, help="extra decode factor of test_y0 agb beyond 1.14114 (seed gate; cancels in the hybrid)")
    ap.add_argument("--base", default="results/spatcorr_stack_20261106_s08.csv")
    ap.add_argument("--sites", default="", help="optional file with site ids: restrict to them")
    ap.add_argument("--common", action="store_true", help="restrict every root to the sites present in all of them")
    ap.add_argument("--exclude", default="", help="file with site ids to leave at C (e.g. work/fire/fireprone_sites.txt for a no-fire run)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--min-block", type=int, default=6, help="rows needed to define the level of a stratum x age x scenario x column block")
    ap.add_argument("--scale", type=float, default=1.0, help="with --write: also save <part>_x<scale> copies (step toward ED)")
    ap.add_argument("--agb-off", type=float, default=0.0, help="subtract this from the simulated agb in SOLUTION units (11-10: the solution keeps the y0 floor, 0.4014 / 3.39567 = 0.1182)")
    ap.add_argument("--h-off", type=float, default=0.0, help="the same for height (floor 0.9009 / 10.90645 = 0.0826)")
    args = ap.parse_args()
    keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy(); band = shift.band_of(age); scen = keys.scenario.to_numpy()
    C = shift.read_predictions(ROOT / args.base).astype(float)
    strat = strata_of_sites()[site]
    tabs = {}
    for item in args.roots:
        nm, r = item.split("=", 1)       # several roots joined by "+": the first root that has a site wins
        parts = [load_root(ROOT / q, args.rows) for q in r.split("+")]
        seen: set[int] = set(); keep = []
        for q in parts:
            if len(q) == 0:
                continue
            q = q[~q.site.isin(seen)]; seen |= set(q.site.unique()); keep.append(q)
        tabs[nm] = pd.concat(keep, ignore_index=True)
    if args.sites:
        keep = {int(x) for x in open(ROOT / args.sites).read().split()}
        tabs = {k: v[v.site.isin(keep)] for k, v in tabs.items()}
    if args.exclude:
        drop = {int(x) for x in open(ROOT / args.exclude).read().split()}
        tabs = {k: v[~v.site.isin(drop)] for k, v in tabs.items()}
    if args.common:
        common = set.intersection(*[set(v.site.unique()) for v in tabs.values()])
        tabs = {k: v[v.site.isin(common)] for k, v in tabs.items()}
    n_all = C.size
    for nm, t in tabs.items():
        P, sim = predict(t, keys, C, args.h, args.gate, args.gate_lo, args.agb_factor)
        if args.agb_off or args.h_off:
            P = np.where(sim, np.maximum(P - np.array([args.h_off, args.agb_off]), 0.0), P)
        D = P - C
        ns = t.site.nunique()
        print(f"== {nm}: {ns} sites, rows {len(t)}; gate pass h {sim[:, 0].sum() / max(len(t), 1):.3f} a {sim[:, 1].sum() / max(len(t), 1):.3f}; "
              f"E_all[(ED - C)^2] = {np.sum(D ** 2) / n_all:.5f} ({args.rows} rows, height {args.h})")
        print("   stratum   band  | sites |  height: median log(ED / C)  scatter  energy  |  agb: median log(ED / C)  scatter  energy  | scenario contrast agb ED / C")
        for st in STRATA:
            for b in ("young", "mid", "old"):
                m = (strat == st) & (band == b)
                out = []
                for c in (0, 1):
                    q = m & sim[:, c] & (C[:, c] * DIV[c] > (2.0 if c == 0 else 0.3)) & (P[:, c] > 0)
                    if q.sum() < 20:
                        out.append("          -          "); continue
                    lr = np.log(P[q, c] / C[q, c])
                    sc_ = 0.5 * (np.percentile(lr, 84) - np.percentile(lr, 16))
                    out.append(f"{np.median(lr):+.3f}  {sc_:.3f}  {np.sum(D[m, c] ** 2) / n_all:.5f}")
                q = m & sim[:, 1] & (C[:, 1] * DIV[1] > 0.3)
                if q.sum() < 20:
                    continue
                s5 = q & (scen == "ssp585"); s1 = q & (scen == "ssp126")
                con = f"{np.median(np.log(np.maximum(P[s5, 1], 1e-4))) - np.median(np.log(np.maximum(P[s1, 1], 1e-4))):+.3f} / {np.median(np.log(np.maximum(C[s5, 1], 1e-4))) - np.median(np.log(np.maximum(C[s1, 1], 1e-4))):+.3f}"
                print(f"   {st:8s} {b:5s} | {len(np.unique(site[q])):4d} | {out[0]} | {out[1]} | {con}")
        # level-free part: the median log offset of every stratum x age x scenario x column block is removed, so that
        # only the within-block SITE PATTERN of ED remains (C's block levels are board-calibrated); `lev` = D - `lf`
        P_lf = P.copy()
        P_lm = P.copy()   # pattern at the SCENARIO-MEAN block level (the level part then splits into mean + scenario contrast)
        for st in STRATA:
            for a in np.unique(age):
                offs = {}
                for sc in ("ssp126", "ssp585"):
                    blk = (strat == st) & (age == a) & (scen == sc)
                    for c in (0, 1):
                        q = blk & sim[:, c] & (C[:, c] * DIV[c] > (2.0 if c == 0 else 0.3)) & (P[:, c] > 0)
                        if q.sum() >= args.min_block:
                            o = float(np.median(np.log(P[q, c] / C[q, c])))
                            P_lf[q, c] = P[q, c] * np.exp(-o); offs[(sc, c)] = (o, q)
                        else:
                            P_lf[blk, c] = C[blk, c]          # too few rows to define a level: the pattern leaves the block at C (its D counts as mean level)
                for c in (0, 1):
                    have = [offs[(sc, c)] for sc in ("ssp126", "ssp585") if (sc, c) in offs]
                    if have:
                        mo = float(np.mean([o for o, _ in have]))
                        for o, q in have:
                            P_lm[q, c] = P[q, c] * np.exp(mo - o)
        D_lf = P_lf - C
        D_levm = P_lm - P_lf        # block levels common to both scenarios
        D_levc = P - P_lm           # scenario contrast of the block levels
        print(f"   level-free pattern: E_all {np.sum(D_lf ** 2) / n_all:.5f}; level part: E_all {np.sum((D - D_lf) ** 2) / n_all:.5f}; "
              f"cos(pattern, level) {np.sum(D_lf * (D - D_lf)) / max(np.sqrt(np.sum(D_lf ** 2) * np.sum((D - D_lf) ** 2)), 1e-30):+.3f}")
        print(f"   level part = scenario-mean block level: E_all {np.sum(D_levm ** 2) / n_all:.5f} + scenario contrast: E_all {np.sum(D_levc ** 2) / n_all:.5f}; "
              f"pattern by column: height {np.sum(D_lf[:, 0] ** 2) / n_all:.5f}, agb {np.sum(D_lf[:, 1] ** 2) / n_all:.5f}")
        if args.write:
            for kind, Dk in (("", D), ("lf_", D_lf), ("lev_", D - D_lf), ("levm_", D_levm), ("levc_", D_levc)):
                for part in ("all",) + STRATA:
                    m = np.ones(len(site), dtype=bool) if part == "all" else (strat == part)
                    Dp = np.where(m[:, None], Dk, 0.0)
                    if np.abs(Dp).sum() > 0:
                        np.save(ROOT / f"work/dir_dec_{nm}{args.suffix}_{kind}{part}.npy", Dp.astype(np.float32))
                        print(f"   wrote work/dir_dec_{nm}{args.suffix}_{kind}{part}.npy  E_all {np.mean(Dp ** 2):.5f}  sites {len(np.unique(site[np.abs(Dp).sum(1) > 0]))}")
                        if args.scale != 1.0:
                            np.save(ROOT / f"work/dir_dec_{nm}{args.suffix}_{kind}{part}_x{int(round(args.scale * 100)):03d}.npy", (args.scale * Dp).astype(np.float32))


if __name__ == "__main__":
    main()
