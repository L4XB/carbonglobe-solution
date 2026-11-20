#!/usr/bin/env python
"""Screen candidate directions for novelty against the measured + planned probe span in one pass.

Builds the span once (every scored spatial probe's deviation from its centre, plus the planned probe files) and reports,
for each candidate .npy direction of shape (N, 2): its raw energy E[D^2] over all cells, the share outside the span and
the novel energy, per block. Nothing is written; build the probe with `scripts/dir_probe.py` afterwards.

    python scripts/novelty_screen.py --planned-list planned.txt cand1.npy cand2.npy
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

R_PATH = "results/latent_ref_y0_m114114.csv"


def span_basis(planned: list[str]) -> np.ndarray:
    """Orthonormal basis (2N, K) of the scored spatial probes' and the planned probes' deviations from their centres."""
    rd = lambda p: shift.read_predictions(p).astype(np.float64)
    sc = json.load(open(ROOT / "results" / "board_scores.json"))
    centers = json.loads((ROOT / "work" / "spat_centers.json").read_text())
    R = rd(ROOT / R_PATH)
    cache: dict[str, np.ndarray] = {}

    def centre(c: str) -> np.ndarray:
        if c == "R" or c.endswith(Path(R_PATH).name):
            return R
        if c not in cache:
            cache[c] = rd(ROOT / c)
        return cache[c]
    names = sorted(n for n in sc if n.startswith(("spat_", "perp_spat_")))
    names += [Path(p).name for p in planned if Path(p).name not in sc]
    cols = [(rd(ROOT / "results" / n) - centre(centers.get(n, "R"))).ravel() for n in names]
    Q, _ = np.linalg.qr(np.stack(cols, 1))
    print(f"span: {len(names)} probes ({len(planned)} planned)")
    return Q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cands", nargs="+")
    ap.add_argument("--planned-list", default="")
    args = ap.parse_args()
    planned = [l.strip() for l in open(args.planned_list)] if args.planned_list else []
    Q = span_basis([p for p in planned if p])
    keys = shift.template_keys(); masks = shift.band_masks(keys["age"].to_numpy())
    blocks = {f"{b[0]}{'ha'[c]}": (masks[b], c) for b in ("young", "mid", "old") for c in (0, 1)}
    for path in args.cands:
        D = np.load(path).astype(np.float64)
        d = D.ravel(); perp = (d - Q @ (Q.T @ d)).reshape(D.shape)
        raw, nov = float(np.mean(D ** 2)), float(np.mean(perp ** 2))
        per = " ".join(f"{k}:{np.sum(perp[m, c] ** 2) / D.size:.5f}" for k, (m, c) in blocks.items()
                       if np.sum(D[m, c] ** 2) > 0)
        print(f"{Path(path).stem:34s} raw {raw:.5f} novel {nov:.5f} ({nov / max(raw, 1e-30):.2f})  {per}")


if __name__ == "__main__":
    main()
