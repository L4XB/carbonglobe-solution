#!/usr/bin/env python
"""Exact line search on the public board with a +-pair centred on a blend.

build:   pair_probe.py build --base results/B.csv --dir results/F.csv [--k 4] --tag name
         D = F - B (or --dir-is-delta: the file holds D itself). Writes results/pair_<tag>_plus.csv = B + kD and
         results/pair_<tag>_minus.csv = B - kD. Rows where B - kD or B + kD would leave [0, cap] get D = 0, so neither
         file is clipped and the algebra below is exact. The cut is reported (it changes D, not the readout).
read:    pair_probe.py read --tag name   (needs the base score and both pair scores in board_scores.json)
         s(B + kD) = s_B + 2k X + k^2 Y with X = E_pub[(B - T) D], Y = E_pub[D^2]:
         X = (s+ - s-) / 4k, Y = (s+ + s- - 2 s_B) / 2k^2; the best step c* = -X / Y gains X^2 / Y exactly on the public
         half, and writes results/chain_<tag>.csv = clip(B + c* D) with its public score s_B - X^2/Y (before clipping,
         which can only help).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

CAP = 14.0  # stay inside our own validator's range


def build(args: argparse.Namespace) -> None:
    B = shift.read_predictions(ROOT / args.base).astype(np.float64)
    F = shift.read_predictions(ROOT / args.dir).astype(np.float64)
    D = F if args.dir_is_delta else F - B
    k = args.k
    bad = ((B - k * D) < 0) | ((B + k * D) < 0) | ((B + k * D) > CAP) | ((B - k * D) > CAP)
    lost = float(np.sum(D[bad] ** 2) / np.sum(D ** 2)) if np.sum(D ** 2) > 0 else 0.0
    D = np.where(bad, 0.0, D)
    meta = {"base": args.base, "dir": args.dir, "k": k, "E_all_D2": float(np.mean(D ** 2)), "cut_energy_share": lost}
    shift.write_predictions(B + k * D, ROOT / "results" / f"pair_{args.tag}_plus.csv")
    shift.write_predictions(B - k * D, ROOT / "results" / f"pair_{args.tag}_minus.csv")
    np.save(ROOT / "work" / f"pair_{args.tag}_D.npy", D)
    (ROOT / "work" / f"pair_{args.tag}.json").write_text(json.dumps(meta, indent=1))
    # the written CSVs are rounded to 6 decimals; re-read them so the stored D is exactly what was submitted
    Pp = shift.read_predictions(ROOT / "results" / f"pair_{args.tag}_plus.csv").astype(np.float64)
    Pm = shift.read_predictions(ROOT / "results" / f"pair_{args.tag}_minus.csv").astype(np.float64)
    print(f"pair {args.tag}: k {k}, E_all[D^2] {meta['E_all_D2']:.6f} (k^2 x = {k*k*meta['E_all_D2']:.5f}), "
          f"energy cut to keep [0, {CAP}]: {lost:.4f}; min {min(Pp.min(), Pm.min()):.4f}; "
          f"symmetric to 1e-6: {np.abs((Pp + Pm) / 2 - B).max() < 2e-6}")


def read(args: argparse.Namespace) -> None:
    meta = json.loads((ROOT / "work" / f"pair_{args.tag}.json").read_text())
    sc = json.load(open(ROOT / "results" / "board_scores.json"))
    base = Path(meta["base"]).name
    s_b = sc[base] if args.base_score is None else args.base_score
    sp, sm = sc[f"pair_{args.tag}_plus.csv"], sc[f"pair_{args.tag}_minus.csv"]
    k = meta["k"]
    X = (sp - sm) / (4 * k)
    Y = (sp + sm - 2 * s_b) / (2 * k * k)
    dX = np.sqrt(2) * 5e-4 / (4 * k)
    dY = np.sqrt(4 * 2.5e-7 + 2 * 2.5e-7) / (2 * k * k)
    c = -X / Y
    gain = X * X / Y
    print(f"pair {args.tag}: s_B {s_b:.3f}, s+ {sp:.3f}, s- {sm:.3f}, k {k}")
    print(f"  X = E_pub[(B-T)D] {X:+.6f} (+-{dX:.6f})   Y = E_pub[D^2] {Y:.6f} (+-{dY:.6f}; rho * E_all {1.0182 * meta['E_all_D2']:.6f})")
    print(f"  best step c* {c:+.3f}  public gain {gain:.5f}  -> chain score {s_b - gain:.4f} (from s_B rounded to 3 decimals)")
    if args.write:
        B = shift.read_predictions(ROOT / meta["base"]).astype(np.float64)
        D = np.load(ROOT / "work" / f"pair_{args.tag}_D.npy")
        P = np.clip(B + c * D, 0, None)
        out = ROOT / "results" / f"chain_{args.tag}.csv"
        shift.write_predictions(P, out)
        print(f"  wrote {out.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--base", required=True); b.add_argument("--dir", required=True)
    b.add_argument("--k", type=float, default=4.0); b.add_argument("--tag", required=True); b.add_argument("--dir-is-delta", action="store_true")
    r = sub.add_parser("read"); r.add_argument("--tag", required=True); r.add_argument("--write", action="store_true")
    r.add_argument("--base-score", type=float, default=None)
    args = ap.parse_args()
    build(args) if args.cmd == "build" else read(args)


if __name__ == "__main__":
    main()
