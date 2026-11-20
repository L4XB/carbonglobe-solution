#!/usr/bin/env python
"""Final assembly of a submission from the board-read ED parts.

    P = base + sum_p lambda_p,b * D_p,   D_p = (ED - C) on the rows of part p (gated),
    lambda_p,b = lambda_p,C - E[(base - C) . D_p] / E[D_p^2]   (exact transfer of the C-centred readout to the base)

ED is the hybrid year-40 state of every finished site in --ed-dir (alpha 1, despiked x40, offset-log hard gate
|log((y0 + eps) / (e0 + eps))| <= ln 2, eps 0.1 m / 0.01 kgC), exactly as in the readout probes
(work/strategist/round7/build_day46_ed.py). Part names are <group>_<band>_<cols>:
    group  cl (collapse cluster) | warm (non-cluster, first-decade T >= 8 C) | cold (non-cluster, T < 8 C)
    band   young | mid | old | all
    cols   h | a | ha
The day-46 probe names map as cl_a = cl_all_a, cl_h = cl_all_h, warm_old = warm_old_ha, warm_young = warm_young_ha.
A part may name an ED variant, <part>@<variant> (variants: c400a = work/edv3/full_c400a, v4 = work/edv3/full_v4,
v4asis = work/edv3/cl_v4asis; default --ed-dir). Parts of different variants may share rows: their weights are then a
joint fit (w = G^-1 b, C-centred), and the transfer to the base is solved jointly over all parts.

    python scripts/ed/assemble_final.py --base results/spatcorr_stack_20261109_s08.csv \
        --lambdas cl_all_a=0.6,cl_all_h=0.2,warm_old_ha=0.5,warm_young_ha=0.7 --out results/final_ed_test.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402
import ed_transfer as et  # noqa: E402

DIV = np.array([10.90645, 3.39567])
EPS = np.array([0.1, 0.01])
BANDS = ("young", "mid", "old")


def e0_rows(ed_dir: Path, keys) -> np.ndarray:
    """ED's state after the year-0 step (results/<site>_<scen>_y0.csv) on the template rows, raw units."""
    site, age, scen = keys["site"].to_numpy(), keys["age"].to_numpy(), keys["scenario"].to_numpy()
    index = {(int(s), str(sc), int(a)): i for i, (s, sc, a) in enumerate(zip(site, scen, age))}
    e0 = np.zeros((len(keys), 2))
    for f in glob.glob(str(ed_dir / "*_y0.csv")):
        m = re.match(r"(\d+)_(ssp\d+)_y0\.csv$", Path(f).name)
        arr = np.atleast_1d(np.genfromtxt(f, delimiter=",", names=True, dtype=float))
        for row in arr:
            k = (int(m.group(1)), m.group(2), int(round(row["age"])))
            if k in index:
                e0[index[k]] = [row["height_m"], row["agb_kgC"]]
    return e0


def part_mask(name: str, group: dict[str, np.ndarray], band: np.ndarray, n: int) -> np.ndarray:
    """(rows, 2) boolean mask of a part name <group>_<band>_<cols>."""
    g, b, c = name.split("_")
    rows = group[g] & (np.ones(n, bool) if b == "all" else band == b)
    cols = {"h": [True, False], "a": [False, True], "ha": [True, True]}[c]
    return rows[:, None] & np.array(cols)[None, :]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--centre", default="results/spatcorr_stack_20261106_s08.csv")
    ap.add_argument("--ed-dir", default=str(ROOT / "work" / "edv3" / "full_c400a" / "results"))
    ap.add_argument("--lambdas", required=True, help="comma list part=lambda (C-centred readout values)")
    ap.add_argument("--no-transfer", action="store_true", help="apply the C-centred lambdas unchanged on the base")
    ap.add_argument("--h-raw", default="cl,warm", help="groups whose height column is the raw despiked x40 (as in the day-46 probes); empty = hybrid everywhere")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    keys = shift.template_keys(); site = keys["site"].to_numpy(); band = shift.band_of(keys["age"].to_numpy()); n = len(keys)
    rd = lambda p: shift.read_predictions(ROOT / p).astype(np.float64)
    C, base, R = rd(args.centre), rd(args.base), rd("results/latent_ref_y0_m114114.csv")
    cl = (np.load(ROOT / "work" / "collapse_cluster_labels.npy") == 0)[site]
    cold = (np.load(ROOT / "work" / "edv3" / "test_first_decade_T.npy") < 8.0)[site]
    h_raw_groups = args.h_raw.split(",") if args.h_raw else []

    def directions(ed_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
        """Gated D = ED - C (rows, 2) of one ED root, its groups, and its simulated-row mask."""
        ED, sim, _ = et.read_ed(ed_dir, keys, (1.0, 1.0), form="hybrid", R=R, smooth="despike")
        W = et.read_ed(ed_dir, keys, (1.0, 1.0), form="raw", R=R, smooth="despike")[0] if h_raw_groups else None
        e0 = e0_rows(ed_dir, keys)
        gate = np.abs(np.log((R * DIV + EPS) / (np.maximum(e0, 0.0) + EPS))) <= np.log(2.0)
        fwx = np.zeros(6171, bool)                     # sites run with ED potential fire in the windows (fire_on and an explicit cap)
        for jf in glob.glob(str(ed_dir.parent / "csv" / "site*.json")):
            try:
                cfg = json.load(open(jf)).get("config") or {}
                if cfg.get("fire_on") and (cfg.get("params_over") or {}).get("fire_max_disturbance_rate") is not None:
                    fwx[int(re.search(r"site(\d+)\.json$", jf).group(1))] = True
            except Exception:  # noqa: BLE001
                pass
        group = {"cl": cl, "warm": ~cl & ~cold & fwx[site], "cold": ~cl & cold}
        for g in h_raw_groups:
            ED[group[g], 0] = W[group[g], 0]
        print(f"ED {ed_dir}: {len(np.unique(site[sim]))} finished sites, {int(fwx.sum())} fire-window sites")
        return np.where(sim[:, None] & gate, ED - C, 0.0), group, sim

    variants = {"default": Path(args.ed_dir), "c400a": ROOT / "work/edv3/full_c400a/results", "v4": ROOT / "work/edv3/full_v4/results",
                "v4asis": ROOT / "work/edv3/cl_v4asis/results"}
    cache: dict[str, tuple] = {}
    print(f"base {args.base}; centre {args.centre}")
    names, lam_c, Ds = [], [], []
    used: dict[str, np.ndarray] = {}
    for item in args.lambdas.split(","):
        name, lam = item.split("=")
        part, var = (name.split("@") + ["default"])[:2]
        if var not in cache:
            cache[var] = directions(variants[var])
        D, group, _ = cache[var]
        m = part_mask(part, group, band, n) & (D != 0.0)
        u = used.setdefault(var, np.zeros_like(D, bool))
        if (m & u).any():
            raise SystemExit(f"part {name} overlaps an earlier part of the same variant")
        u |= m
        Dp = np.where(m, D, 0.0)
        names.append(name); lam_c.append(float(lam)); Ds.append(Dp)
        print(f"  {name}: {int(m.any(axis=1).sum())} rows, {len(np.unique(site[m.any(axis=1)]))} sites, E[D^2] {float(np.sum(Dp * Dp)) / D.size:.5f}")
    # lam_c are C-centred weights (single readouts, or a joint fit's w); transfer to the base jointly:
    # w_b = w_C - G^-1 <D, base - C>, G_pq = <D_p, D_q>
    G = np.array([[float(np.sum(a * b)) for b in Ds] for a in Ds])
    shiftv = np.array([float(np.sum((base - C) * Dp)) for Dp in Ds])
    w_b = np.array(lam_c) if args.no_transfer else np.array(lam_c) - np.linalg.lstsq(G, shiftv, rcond=None)[0]
    P = base.copy()
    for name, lc, wb, Dp in zip(names, lam_c, w_b, Ds):
        P += wb * Dp
        print(f"  {name}: lambda_C {lc:+.3f} -> lambda_base {wb:+.3f}")
    P = np.clip(P, 0.0, None)
    shift.write_predictions(P, ROOT / args.out)
    print(f"wrote {args.out}: E[(P - base)^2] {float(np.mean((P - base) ** 2)):.5f}, E[(P - C)^2] {float(np.mean((P - C) ** 2)):.5f}")


if __name__ == "__main__":
    main()
