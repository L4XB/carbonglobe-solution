#!/usr/bin/env python
"""Probe only the novel part of a candidate: D_perp = (P - R) minus its least-squares projection on the span of the
scored files (R and every scored file's deviation from R; earlier blends excluded), added to a base blend.

A full candidate file mixes its novel part with directions the board already measured (with their noise); probing
D_perp alone gives the solver the new direction with the precision of its own (small) energy.

    ./.venv/bin/python scripts/perp_probe.py --cand results/x.csv --tag name [--base results/B.csv] [--k 2] [+planned.csv ...]
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

SKIP = ("persistence_s05", "colprobe_latent_v1x2_level090", "persistence_raw", "zeros", "constant_one", "fsolve", "hybrid",
        "blend_gnr", "symblend", "wshrink", "pair_", "chain_", "perp_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", default="results/fsolve_rho_r2e4_20261005.csv")
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("planned", nargs="*", help="files treated as already measured")
    args = ap.parse_args()
    sc = json.load(open(ROOT / "results" / "board_scores.json"))
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    basis = [R.ravel()]
    for n in sc:
        if any(x in n for x in SKIP):
            continue
        basis.append((shift.read_predictions(ROOT / "results" / n).astype(np.float64) - R).ravel())
    for p in args.planned:
        basis.append((shift.read_predictions(ROOT / p.lstrip("+")).astype(np.float64) - R).ravel())
    Q, _ = np.linalg.qr(np.stack(basis, 1))
    d = (shift.read_predictions(ROOT / args.cand).astype(np.float64) - R).ravel()
    perp = (d - Q @ (Q.T @ d)).reshape(R.shape)
    B = shift.read_predictions(ROOT / args.base).astype(np.float64)
    k = args.k
    bad = (B + k * perp) < 0
    cut = float(np.sum(perp[bad] ** 2) / np.sum(perp ** 2))
    perp[bad] = 0.0
    out = ROOT / "results" / f"perp_{args.tag}.csv"
    shift.write_predictions(B + k * perp, out)
    masks = shift.band_masks(shift.template_keys()["age"].to_numpy())
    parts = " ".join(f"{b[0]}{c} {np.sum(perp[masks[b], i] ** 2) / perp.size:.4f}" for b in ("young", "mid", "old") for i, c in ((0, "h"), (1, "a")))
    print(f"{out.name}: E[D_perp^2] {np.mean(perp ** 2):.5f} of E[D^2] {np.mean(d ** 2):.5f} (cut to stay >= 0: {cut:.3f}); k {k}; blocks {parts}")
    np.save(ROOT / "work" / f"perp_{args.tag}_D.npy", perp)


if __name__ == "__main__":
    main()
