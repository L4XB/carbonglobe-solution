#!/usr/bin/env python
"""Smooth spatial probes: graph-Laplacian modes of the weather-neighbour site graph on a block, projected off the span.

Directions are (field over sites) x (row weight) on the chosen block, the field being either a random combination of
modes lo..hi (--seed) or a single mode (--mode). The pattern is projected off the span of all scored files and of any
earlier spatial probes, scaled to --energy, and written as decoded persistence + pattern (R-centred, so the readout
E[(T - R) D] = -(s - s_R - E[D^2]) / 2 needs no blend score).

    ./.venv/bin/python scripts/spatial_probe.py --block oa --modes 1-40 --seed 3 --tag oa_r3
    ./.venv/bin/python scripts/spatial_probe.py --block oa --mode 1 --tag oa_m1
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
        "blend_gnr", "symblend", "wshrink", "youngsolve", "blocksolve", "stack_", "dieoffblend", "pair_", "chain_")
BLOCKS = {"oa": [("old", 1)], "oh": [("old", 0)], "ma": [("mid", 1)], "mh": [("mid", 0)], "ya": [("young", 1)], "yh": [("young", 0)],
          "young": [("young", 0), ("young", 1)], "mid": [("mid", 0), ("mid", 1)], "old": [("old", 0), ("old", 1)],
          "yoa": [("young", 0), ("young", 1), ("old", 1)]}   # joint: growth weight on young rows, stand size on old agb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block", required=True, choices=sorted(BLOCKS))
    ap.add_argument("--modes", default="1-40")
    ap.add_argument("--mode", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--energy", type=float, default=0.0065)
    ap.add_argument("--weight", default="R", choices=["R", "one", "growth", "dis"],
                    help="row weight: decoded persistence, 1, or |0.876 A(a+46) - R| (the expected 40-year change)")
    ap.add_argument("--scenario", default="both", choices=["both", "ssp126", "ssp585", "contrast"],
                    help="contrast: +field on ssp585 rows, -field on ssp126 rows (the scenario-specific regional part)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--planned", default="", help="comma list of unscored probe files (results/...) to treat as measured")
    ap.add_argument("--modefile", default="work/site_graph_modes240.npy", help="site modes (columns): geographic or climate graph")
    ap.add_argument("--center", default="", help="scored blend to add the pattern to (default: decoded persistence R); "
                    "recorded in work/spat_centers.json for the readout")
    args = ap.parse_args()
    sc = json.load(open(ROOT / "results" / "board_scores.json"))
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    keys = shift.template_keys(); masks = shift.band_masks(keys["age"].to_numpy())
    st = keys["site"].to_numpy(); scn = keys["scenario"].to_numpy()
    modes = np.load(ROOT / args.modefile)
    if args.mode >= 0:
        f = modes[:, args.mode]
    else:
        lo, hi = (int(v) for v in args.modes.split("-"))
        g = np.random.default_rng(args.seed).normal(size=hi - lo + 1)
        f = modes[:, lo:hi + 1] @ g
    f = f / f.std()
    D = np.zeros_like(R)
    for band, col in BLOCKS[args.block]:
        m = masks[band] & ((scn == args.scenario) if args.scenario in ("ssp126", "ssp585") else True)
        if args.block == "yoa":
            w = np.abs(0.876 * np.load(ROOT / "work" / "ageshift46_dec.npy")[m, col] - R[m, col]) if band == "young" else R[m, col]
        elif args.weight == "R":
            w = R[m, col]
        elif args.weight == "dis":
            w = np.load(ROOT / "work" / "oa_family_disagreement.npy")[m]   # sd of three model families' agb predictions
        elif args.weight == "growth":
            w = np.abs(0.876 * np.load(ROOT / "work" / "ageshift46_dec.npy")[m, col] - R[m, col])
        else:
            w = np.ones(m.sum())
        sgn = np.where(scn[m] == "ssp585", 1.0, -1.0) if args.scenario == "contrast" else 1.0
        part = sgn * f[st[m]] * w
        if args.block == "yoa":   # equal energy for the young and the old-agb part
            part = part / np.sqrt(np.sum(part ** 2) * (2.0 if band == "young" else 1.0))
        D[m, col] = part
    basis = [R.ravel()]
    for n in sc:
        if any(x in n for x in SKIP):
            continue
        basis.append((shift.read_predictions(ROOT / "results" / n).astype(np.float64) - R).ravel())
    for pf in [x for x in args.planned.split(",") if x]:
        basis.append((shift.read_predictions(ROOT / pf).astype(np.float64) - R).ravel())
    Q, _ = np.linalg.qr(np.stack(basis, 1))
    d = D.ravel()
    perp = (d - Q @ (Q.T @ d)).reshape(R.shape)
    perp *= np.sqrt(args.energy / np.mean(perp ** 2))
    base = R if not args.center else shift.read_predictions(ROOT / args.center).astype(np.float64)
    bad = (base + perp) < 0
    cut = float(np.sum(perp[bad] ** 2) / np.sum(perp ** 2)); perp[bad] = 0.0
    out = ROOT / "results" / f"spat_{args.tag}.csv"
    shift.write_predictions(base + perp, out)
    np.save(ROOT / "work" / f"spat_{args.tag}_D.npy", perp)
    cfile = ROOT / "work" / "spat_centers.json"
    centers = json.loads(cfile.read_text()) if cfile.exists() else {}
    centers[out.name] = args.center or "R"
    cfile.write_text(json.dumps(centers, indent=1))
    print(f"{out.name}: E[D^2] {np.mean(perp ** 2):.5f} (cut {cut:.3f}); share of the raw pattern outside the span "
          f"{np.mean((d.reshape(R.shape) - 0) ** 0) if False else np.sum((d - Q @ (Q.T @ d)) ** 2) / np.sum(d ** 2):.3f}")


if __name__ == "__main__":
    main()
