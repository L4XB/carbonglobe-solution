#!/usr/bin/env python
"""Probe an arbitrary direction on a block: its part outside the span of every scored file and spatial probe.

The direction is either a candidate prediction file (direction = candidate - base on the block) or an .npy array of
shape (N, 2) taken as given on the block. Projected off the span (R, every scored file's deviation, every spatial
probe's deviation from its centre, and --planned files), scaled to E[D^2] = --energy and added to the base, which is
recorded as the probe's centre so `spatial_correction.py` reads it like any other spatial probe.

    ./.venv/bin/python scripts/dir_probe.py --dir results/chronodyn1_g46.csv --block ya --tag dir_chrono_ya1
    ./.venv/bin/python scripts/dir_probe.py --dir work/twostep_D.npy --block oh --tag dir_twostep_oh1
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

BLOCKS = {"oa": [("old", 1)], "oh": [("old", 0)], "ma": [("mid", 1)], "mh": [("mid", 0)], "ya": [("young", 1)],
          "yh": [("young", 0)], "young": [("young", 0), ("young", 1)], "mid": [("mid", 0), ("mid", 1)],
          "old": [("old", 0), ("old", 1)], "all": [(b, c) for b in ("young", "mid", "old") for c in (0, 1)]}
SKIP = ("persistence_s05", "colprobe_latent_v1x2_level090", "persistence_raw", "zeros", "constant_one", "fsolve", "hybrid",
        "blend_gnr", "symblend", "wshrink", "youngsolve", "blocksolve", "stack_", "dieoffblend", "pair_", "chain_", "spatcorr")
R_PATH = "results/latent_ref_y0_m114114.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--block", required=True, choices=sorted(BLOCKS))
    ap.add_argument("--mask", default="", help="optional .npy boolean row mask (N,) further restricting the block")
    ap.add_argument("--base", default="results/stack_20261010.csv")
    ap.add_argument("--energy", type=float, default=0.012)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--planned", default="")
    ap.add_argument("--max-cut", type=float, default=1.0,
                    help="lower the energy until at most this share of the probe's energy is clipped at zero")
    ap.add_argument("--rel-cap", type=float, default=0.0,
                    help="clip the probe to +-rel_cap x base per cell (e.g. 0.6) instead of cutting B + D at zero")
    ap.add_argument("--off-rcentred", action="store_true",
                    help="with --no-span: also project off the cleanly measured R-centred probes (work/rcentred_files.json)")
    ap.add_argument("--min-base", type=float, default=0.0,
                    help="leave cells whose base prediction is below this untouched (bare or dying rows)")
    ap.add_argument("--no-span", action="store_true",
                    help="project off the spatial probes and --planned only: the scored files' span is numerically large but "
                         "informationally thin (directions in it need |coef|_1 in the hundreds), so removing it discards signal")
    args = ap.parse_args()
    rd = lambda p: shift.read_predictions(p).astype(np.float64)
    sc = json.load(open(ROOT / "results" / "board_scores.json"))
    R = rd(ROOT / R_PATH)
    B = rd(ROOT / args.base)
    keys = shift.template_keys(); masks = shift.band_masks(keys["age"].to_numpy())
    src = ROOT / args.dir
    direction = np.load(src) if src.suffix == ".npy" else rd(src) - B
    rowmask = np.load(ROOT / args.mask) if args.mask else np.ones(len(R), dtype=bool)
    D = np.zeros_like(R)
    for band, col in BLOCKS[args.block]:
        m = masks[band] & rowmask
        D[m, col] = direction[m, col]
    centers = json.loads((ROOT / "work" / "spat_centers.json").read_text())

    def dev(n: str) -> np.ndarray:
        if n.startswith(("spat_", "vc_")):
            c = centers.get(n, "R")
            Cp = R if (c == "R" or c.endswith(Path(R_PATH).name)) else rd(ROOT / c)
            return (rd(ROOT / "results" / n) - Cp).ravel()
        return (rd(ROOT / "results" / n) - R).ravel()
    rc = set(json.loads((ROOT / "work" / "rcentred_files.json").read_text())) if args.off_rcentred else set()
    keep = (lambda n: n.startswith(("spat_", "perp_spat_")) or n in rc) if args.no_span else (lambda n: True)
    basis = [R.ravel()] + [dev(n) for n in sc if not any(x in n for x in SKIP) and keep(n)]
    basis += [dev(Path(p).name) for p in args.planned.split(",") if p]
    Q, _ = np.linalg.qr(np.stack(basis, 1))
    live = B >= args.min_base
    D[~live] = 0.0
    d = D.ravel(); perp = (d - Q @ (Q.T @ d)).reshape(R.shape)
    perp[~live] = 0.0
    share = float(np.sum(perp ** 2) / np.sum(d ** 2))
    novel = float(np.mean(perp ** 2))
    unit = perp / np.sqrt(np.mean(perp ** 2))
    energy = args.energy
    while True:
        perp = unit * np.sqrt(energy)
        if args.rel_cap > 0:
            capped = np.clip(perp, -args.rel_cap * B, args.rel_cap * B)
            cut = float(np.sum((perp - capped) ** 2) / np.sum(perp ** 2))
        else:
            bad = (B + perp) < 0; cut = float(np.sum(perp[bad] ** 2) / np.sum(perp ** 2))
        if cut <= args.max_cut or energy < 1e-4:
            break
        energy *= 0.8
    if args.rel_cap > 0:
        perp = np.clip(perp, -args.rel_cap * B, args.rel_cap * B)
    else:
        perp[bad] = 0
    out = ROOT / "results" / f"spat_{args.tag}.csv"
    shift.write_predictions(B + perp, out)
    centers[out.name] = args.base
    (ROOT / "work" / "spat_centers.json").write_text(json.dumps(centers, indent=1))
    print(f"{out.name}: novel energy {novel:.5f} ({share:.3f} of the direction outside the span); E[D^2] "
          f"{np.mean(perp ** 2):.4f}, cut {cut:.3f}")


if __name__ == "__main__":
    main()
