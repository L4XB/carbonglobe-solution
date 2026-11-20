#!/usr/bin/env python
"""Board readout of C-centred ED probes (any day) and joint fits over chosen probe files.

Single readouts: lambda_p = (E_pub,p - (s_p - s_C)) / (2 E_pub,p), b_p = lambda_p E_pub,p (board scores have 3 decimals,
s_C = 0.0438 +- 0.0003). E_pub comes from the builder logs (lines '<file>.csv: ... E_all[D^2] x -> E_pub y').
Joint fit over a group of probe files (all centred on C): D_p = file - C, public Gram G_pub = r G_all with
r = mean(E_pub,p / E_all,p), weights w = G_pub^-1 b, predicted public gain b'w.
    python scripts/ed/readout_ed.py --logs work/logs/build_day46_fw.log,work/logs/build_day47.log \\
        --joint spat_ed46fw_cl_a_v4+spat_ed46fw_cl_a_v4asis
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

S_C = 0.0438
C_FILE = "results/spatcorr_stack_20261106_s08.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", required=True, help="comma list of builder logs")
    ap.add_argument("--scores", default="", help="override: comma list file_stem=score")
    ap.add_argument("--joint", nargs="*", default=[], help="groups of probe stems joined by +")
    args = ap.parse_args()
    e_pub: dict[str, tuple[float, float]] = {}
    for lg in args.logs.split(","):
        if not Path(ROOT / lg).exists():
            continue
        for line in open(ROOT / lg):
            m = re.search(r"(\S+?)\.csv: .*E_all\[D\^2\] ([\d.]+) -> E_pub ([\d.]+)", line)
            if m:
                e_pub[Path(m.group(1)).name] = (float(m.group(3)), float(m.group(2)))
    sc = {Path(k).stem: v for k, v in json.load(open(ROOT / "results/board_scores.json")).items()}
    if args.scores:
        sc.update({k: float(v) for k, v in (x.split("=") for x in args.scores.split(","))})
    lam: dict[str, float] = {}
    print(f"s_C {S_C}")
    for p, (ep, ea) in sorted(e_pub.items()):
        if p not in sc:
            print(f"  {p}: no score yet (E_pub {ep:.5f})"); continue
        s = sc[p]; lam[p] = (ep - (s - S_C)) / (2 * ep); err = np.hypot(0.0005, 0.0003) / (2 * ep)
        print(f"  {p:34s} score {s:.3f}  E_pub {ep:.5f}  lambda {lam[p]:+.3f} +- {err:.3f}  gain at lambda {max(lam[p], 0) ** 2 * ep:.5f}")
    if not args.joint:
        return
    rd = lambda f: shift.read_predictions(ROOT / f).astype(np.float64)
    C = rd(C_FILE)
    for grp in args.joint:
        g = [p for p in grp.split("+") if p in lam]
        if len(g) < 2:
            print(f"  joint {grp}: fewer than 2 scored probes"); continue
        Ds = [rd(f"results/{p}.csv") - C for p in g]
        n = Ds[0].size
        G = np.array([[float(np.sum(a * b)) / n for b in Ds] for a in Ds])
        r = float(np.mean([e_pub[p][0] / G[i, i] for i, p in enumerate(g)]))
        b = np.array([lam[p] * e_pub[p][0] for p in g])
        w = np.linalg.solve(r * G, b)
        cor = G / np.sqrt(np.outer(np.diag(G), np.diag(G)))
        print(f"  joint {'+'.join(g)}: r {r:.3f}; corr " + " ".join(f"{cor[i, j]:+.3f}" for i in range(len(g)) for j in range(i + 1, len(g)))
              + "; weights " + " ".join(f"{p} {wi:+.3f}" for p, wi in zip(g, w)) + f"; public gain {float(b @ w):.5f} -> score {S_C - float(b @ w):.4f}")


if __name__ == "__main__":
    main()
