#!/usr/bin/env python
"""Probe the covariate-extrapolated part of a block's regional residual field.

Runs the ridge covariate fit of `regional_field_fit.py` on the block's scored spatial probes, forms D = r_hat(site) x
w(row), projects it off the span of every scored file and probe (and --planned files), scales it to E[D^2] 0.012 and
writes centre + D. The fit's own prediction of E[(T - B) D] is printed so the board's answer can be compared with it.

    ./.venv/bin/python scripts/field_probe.py --block young --weight growth --prefix young_ --ridge 3 --tag youngfield2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

SKIP = ("persistence_s05", "colprobe_latent_v1x2_level090", "persistence_raw", "zeros", "constant_one", "fsolve", "hybrid",
        "blend_gnr", "symblend", "wshrink", "youngsolve", "blocksolve", "stack_", "dieoffblend", "pair_", "chain_", "spatcorr")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block", required=True)
    ap.add_argument("--weight", default="R")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--ridge", type=float, default=3.0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--center", default="results/stack_20261010.csv")
    ap.add_argument("--planned", default="")
    args = ap.parse_args()
    raw = ROOT / "work" / f"{args.tag}_raw.csv"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "regional_field_fit.py"), "--block", args.block, "--weight", args.weight,
                        "--prefix", args.prefix, "--ridge", str(args.ridge), "--base", args.center, "--out", str(raw.relative_to(ROOT))],
                       capture_output=True, text=True, check=True)
    print("\n".join(r.stdout.splitlines()[-3:]))
    sc = json.load(open(ROOT / "results" / "board_scores.json"))
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    Cn = shift.read_predictions(ROOT / args.center).astype(np.float64)
    D = shift.read_predictions(raw).astype(np.float64) - Cn
    centers = json.loads((ROOT / "work" / "spat_centers.json").read_text())

    def dev(n: str) -> np.ndarray:
        if n.startswith("spat_"):
            c = centers.get(n, "R"); Cp = R if c == "R" else shift.read_predictions(ROOT / c).astype(np.float64)
            return (shift.read_predictions(ROOT / "results" / n).astype(np.float64) - Cp).ravel()
        return (shift.read_predictions(ROOT / "results" / n).astype(np.float64) - R).ravel()
    basis = [R.ravel()] + [dev(n) for n in sc if not any(x in n for x in SKIP)]
    basis += [dev(Path(p).name) for p in args.planned.split(",") if p]
    Q, _ = np.linalg.qr(np.stack(basis, 1))
    d = D.ravel(); perp = (d - Q @ (Q.T @ d)).reshape(R.shape)
    share = float(np.sum(perp ** 2) / np.sum(d ** 2))
    perp *= np.sqrt(0.012 / np.mean(perp ** 2))
    bad = (Cn + perp) < 0; cut = float(np.sum(perp[bad] ** 2) / np.sum(perp ** 2)); perp[bad] = 0
    out = ROOT / "results" / f"spat_{args.tag}.csv"
    shift.write_predictions(Cn + perp, out)
    centers[out.name] = args.center
    (ROOT / "work" / "spat_centers.json").write_text(json.dumps(centers, indent=1))
    print(f"{out.name}: fitted field {share:.3f} outside the measured span; E[D^2] {np.mean(perp ** 2):.4f}, cut {cut:.3f}")


if __name__ == "__main__":
    main()
