#!/usr/bin/env python
"""Regionally varying model weights: a random smooth field over sites x (model family - base blend) on a block.

If the truth follows different models in different regions, T - B = sum_i a_i(site) (M_i - B) with smooth a_i, and a
probe f(site) x (M_i - B) measures a_i's projection on f. Projected off the span and all spatial probes, scaled to
E[D^2] = --energy, added to the base (recorded as the centre for spatial_correction.py).

    ./.venv/bin/python scripts/varcoef_probe.py --family lat --block oa --seed 91 --tag vc_lat_oa1
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

FAMILIES = {"lat": ["latent_noco2_m114", "latent_v1x2_m114", "latent_fullL4_m114", "latent_monthL2_m114"],
            "gb": ["direct_gbdt2_m114", "direct_gbdt6_m114", "direct_gbdt8_m114", "direct_state3_m114"],
            "acc": ["direct_statechain4_m114", "direct_accrate5_m114", "latent_latevl3_m114"]}
BLOCKS = {"oa": [("old", 1)], "young": [("young", 0), ("young", 1)], "mid": [("mid", 0), ("mid", 1)], "oh": [("old", 0)]}
SKIP = ("persistence_s05", "colprobe_latent_v1x2_level090", "persistence_raw", "zeros", "constant_one", "fsolve", "hybrid",
        "blend_gnr", "symblend", "wshrink", "youngsolve", "blocksolve", "stack_", "dieoffblend", "pair_", "chain_", "spatcorr")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
    ap.add_argument("--block", required=True, choices=sorted(BLOCKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", default="1-40")
    ap.add_argument("--base", default="results/stack_20261010.csv")
    ap.add_argument("--energy", type=float, default=0.012)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--planned", default="")
    args = ap.parse_args()
    sc = json.load(open(ROOT / "results" / "board_scores.json"))
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    B = shift.read_predictions(ROOT / args.base).astype(np.float64)
    keys = shift.template_keys(); masks = shift.band_masks(keys["age"].to_numpy()); st = keys["site"].to_numpy()
    Mfam = np.mean([shift.read_predictions(ROOT / "results" / f"{f}.csv").astype(np.float64) for f in FAMILIES[args.family]], 0)
    scale = 0.876 if args.block in ("oa", "oh", "mid") else 1.0     # raw-scale truth for old/mid, the board's young lambda ~1
    modes = np.load(ROOT / "work" / "site_graph_modes240.npy")
    lo, hi = (int(v) for v in args.modes.split("-"))
    f = modes[:, lo:hi + 1] @ np.random.default_rng(args.seed).normal(size=hi - lo + 1); f /= f.std()
    D = np.zeros_like(R)
    for band, col in BLOCKS[args.block]:
        m = masks[band]
        D[m, col] = f[st[m]] * (scale * Mfam[m, col] - B[m, col])
    centers = json.loads((ROOT / "work" / "spat_centers.json").read_text())

    def dev(n: str) -> np.ndarray:
        if n.startswith(("spat_", "vc_")):
            c = centers.get(n, "R"); Cp = R if c == "R" else shift.read_predictions(ROOT / c).astype(np.float64)
            return (shift.read_predictions(ROOT / "results" / n).astype(np.float64) - Cp).ravel()
        return (shift.read_predictions(ROOT / "results" / n).astype(np.float64) - R).ravel()
    basis = [R.ravel()] + [dev(n) for n in sc if not any(x in n for x in SKIP)]
    basis += [dev(Path(p).name) for p in args.planned.split(",") if p]
    Q, _ = np.linalg.qr(np.stack(basis, 1))
    d = D.ravel(); perp = (d - Q @ (Q.T @ d)).reshape(R.shape)
    share = float(np.sum(perp ** 2) / np.sum(d ** 2))
    perp *= np.sqrt(args.energy / np.mean(perp ** 2))
    bad = (B + perp) < 0; cut = float(np.sum(perp[bad] ** 2) / np.sum(perp ** 2)); perp[bad] = 0
    out = ROOT / "results" / f"spat_{args.tag}.csv"
    shift.write_predictions(B + perp, out)
    centers[out.name] = args.base
    (ROOT / "work" / "spat_centers.json").write_text(json.dumps(centers, indent=1))
    print(f"{out.name}: {share:.3f} of the pattern outside the span; E[D^2] {np.mean(perp ** 2):.4f}, cut {cut:.3f}")


if __name__ == "__main__":
    main()
