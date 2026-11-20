#!/usr/bin/env python
"""Apply the board-tested modifications to a blend file.

--sym       replace each (site, age) pair of scenario rows by its mean (scenario contrast removed)
--wshrink c subtract c x the projection of (P - R) on the four latent groups' realization-specific weather effects
--young     re-solve seed ages 1-30 from the young-only probes (scripts/block_solve.py logic, start H46)

    ./.venv/bin/python scripts/apply_mods.py --in results/x.csv --out results/y.csv [--sym] [--wshrink 0.5] [--young]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

GROUPS = {"latelat2": "results/latent_latelat2_actual.csv", "ss20L3": "results/latent_ss20L3_actual.csv",
          "noco2": "work/shuffle/noco2_actual.csv", "v1x2": "work/shuffle/v1x2_actual.csv"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sym", action="store_true")
    ap.add_argument("--wshrink", type=float, default=0.0)
    ap.add_argument("--young", action="store_true")
    args = ap.parse_args()
    P = shift.read_predictions(ROOT / args.inp).astype(np.float64)
    P0 = P.copy()
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    if args.wshrink:
        W = [shift.read_predictions(ROOT / a).astype(np.float64) - np.load(ROOT / "work" / "shuffle" / f"{g}_boot_mean.npy") for g, a in GROUPS.items()]
        X = np.stack([w.ravel() for w in W], 1)
        beta, *_ = np.linalg.lstsq(X, (P - R).ravel(), rcond=None)
        P = P - args.wshrink * (X @ beta).reshape(P.shape)
        print(f"wshrink {args.wshrink}: loadings {np.round(beta, 3)}")
    if args.sym:
        o126, o585 = np.load(ROOT / "work" / "contrast_index.npy")
        m = (P[o126] + P[o585]) / 2
        P[o126] = m; P[o585] = m
    P = np.clip(P, 0, None)
    tmp = ROOT / "work" / "apply_mods_tmp.csv"
    shift.write_predictions(P, tmp if args.young else ROOT / args.out)
    if args.young:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "block_solve.py"), "--blocks", "yh,ya", "--tag", "applymods",
                        "--blend", str(tmp.relative_to(ROOT))], check=True, capture_output=True)
        Y = shift.read_predictions(ROOT / "results" / "blocksolve_applymods_apply_mods_tmp.csv").astype(np.float64)
        shift.write_predictions(Y, ROOT / args.out)
        (ROOT / "results" / "blocksolve_applymods_apply_mods_tmp.csv").unlink()
        P = Y
    print(f"wrote {args.out}: E[(out - in)^2] {np.mean((P - P0) ** 2):.5f}")


if __name__ == "__main__":
    main()
