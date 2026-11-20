#!/usr/bin/env python
"""Board test between two ED configurations on the same rows.

F1 = C + D1 and F2 = C + D2 with Dk = EDk - C (hybrid, despiked, alpha per column), restricted to rows that both
configurations simulated, outside the collapse cluster unless --cluster, and (by default) to rows where both runs
pass the hard fit-quality gate |log(y0 / e0)| <= ln 2. The lower board score is the configuration closer to the truth
on those rows; with s_C known, each file also reads its own lambda: lambda_k = (E_pub[Dk^2] - (s_k - s_C)) / (2 E_pub[Dk^2]).

    python scripts/ed/readout_config.py --ed1 work/edv3/full_c400a/results --ed2 work/edv3/results --tag cfg0950
writes work/strategist/round6/<tag>_F1.csv, _F2.csv and work/strategist/round6/readout_<tag>.txt
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402
import ed_transfer as et  # noqa: E402

DIVS = np.array([10.90645, 3.39567])


def e0_rows(ed_dir: Path, keys) -> np.ndarray:
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ed1", required=True); ap.add_argument("--ed2", required=True)
    ap.add_argument("--alpha", type=float, nargs=2, default=[1.0, 1.0])
    ap.add_argument("--cluster", choices=["out", "in", "all"], default="out")
    ap.add_argument("--bands", default="young,mid,old")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--tag", default="cfgtest"); ap.add_argument("--out-dir", default="work/strategist/round6")
    ap.add_argument("--s-c", type=float, default=0.0438)
    args = ap.parse_args()
    out = ROOT / args.out_dir; out.mkdir(parents=True, exist_ok=True)
    keys = shift.template_keys(); site = keys["site"].to_numpy(); band = shift.band_of(keys["age"].to_numpy())
    rd = lambda p: shift.read_predictions(ROOT / p).astype(np.float64)
    C = rd("results/spatcorr_stack_20261106_s08.csv"); R = rd("results/latent_ref_y0_m114114.csv")
    EDs, sims, gates = [], [], []
    for d in (args.ed1, args.ed2):
        ed, sim, rep = et.read_ed(ROOT / d, keys, tuple(args.alpha), form="hybrid", R=R, smooth="despike")
        e0 = e0_rows(ROOT / d, keys); y0p = R * DIVS
        # fit gate on offset logs, so near-bare y0 rows that ED's spin-up overgrows (e0 >> y0 ~ 0) are gated out too
        eps = np.array([0.1, 0.01])
        r0 = np.log((y0p + eps) / (np.maximum(e0, 0.0) + eps))
        EDs.append(ed); sims.append(sim); gates.append(np.abs(r0) <= np.log(2.0))
    cl = (np.load(ROOT / "work/collapse_cluster_labels.npy") == 0)[site]
    rows = sims[0] & sims[1] & np.isin(band, args.bands.split(","))
    rows &= {"out": ~cl, "in": cl, "all": np.ones_like(cl)}[args.cluster]
    mask = np.repeat(rows[:, None], 2, axis=1)
    if not args.no_gate:
        mask &= gates[0] & gates[1]
    Ds = [np.where(mask, ed - C, 0.0) for ed in EDs]
    rho = 1.0182
    lines = [f"ed1 {args.ed1} vs ed2 {args.ed2}: {len(np.unique(site[rows]))} common sites ({args.cluster}, bands {args.bands}), "
             f"{int(mask.sum())} cells after {'no gate' if args.no_gate else 'the hard gate'}; alpha {args.alpha}"]
    for k, D in enumerate(Ds, 1):
        e_all = float(np.mean(D * D)); e_pub = rho * e_all
        F = np.clip(C + D, 0, None)
        shift.write_predictions(F, out / f"{args.tag}_F{k}.csv")
        blk = " ".join(f"{b[0]}{'ha'[c]}:{np.sum(D[band == b, c] ** 2) / D.size:.5f}" for b in ("young", "mid", "old") for c in (0, 1))
        lines.append(f"F{k} = C + D{k}: E_all {e_all:.5f}, E_pub ~{e_pub:.5f} ({blk}); expected score if lambda=0: {args.s_c + e_pub:.4f}, "
                     f"lambda=0.5: {args.s_c:.4f}, lambda=1: {args.s_c - e_pub:.4f}; lambda_k = (E_pub - (s_k - s_C)) / (2 E_pub), sd ~{np.hypot(0.00029, 0.0003) / (2 * e_pub):.3f}")
    d12 = Ds[0] - Ds[1]
    lines.append(f"E[(D1-D2)^2] {float(np.mean(d12 * d12)):.5f}; s1 - s2 = E_pub[D1^2] - E_pub[D2^2] - 2 E_pub[(T-C)(D1-D2)]: the lower score is the configuration closer to the truth on these cells")
    (out / f"readout_{args.tag}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
