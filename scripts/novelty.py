#!/usr/bin/env python
"""Novel energy of unscored candidate files against the span of the scored files.

For a candidate file P the blend solver gains only through the part of P that no combination of the
scored files reproduces: P_perp = P - proj(P) (least squares over all cells). Its energy E[P_perp^2],
split by band and column, bounds what one board slot spent on P can return: the gain is
lambda_perp^2 E[P_perp^2] where lambda_perp is the truth's coefficient on P_perp.

    ./.venv/bin/python scripts/novelty.py results/latent_latelat_m114.csv results/direct_state3_m114.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

SKIP = {"persistence_s05.csv", "colprobe_latent_v1x2_level090.csv", "persistence_raw.csv", "zeros.csv", "constant_one.csv"}


def main() -> None:
    sc = json.load(open(ROOT / "results" / "board_scores.json"))
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    keys = shift.template_keys(); masks = shift.band_masks(keys["age"].to_numpy())
    basis = [R.ravel()]
    for name in sc:
        if name in SKIP:
            continue
        F = shift.read_predictions(ROOT / "results" / name).astype(np.float64)
        basis.append((F - R).ravel())
    B = np.stack(basis, axis=1)  # (2N, K)
    extra = [a for a in sys.argv[1:] if a.startswith("+")]
    for a in extra:  # files treated as already measured (a planned batch)
        F = shift.read_predictions(a[1:]).astype(np.float64)
        B = np.concatenate([B, (F - R).reshape(-1, 1)], axis=1)
    Q, _ = np.linalg.qr(B)
    print(f"{B.shape[1]} basis vectors (R, the deviations of the scored files, {len(extra)} planned)")
    for path in [a for a in sys.argv[1:] if not a.startswith("+")]:
        P = shift.read_predictions(path).astype(np.float64)
        d = (P - R).ravel()
        perp = d - Q @ (Q.T @ d)
        perp = perp.reshape(R.shape)
        tot = np.mean((P - R) ** 2)
        parts = []
        for b in ("young", "mid", "old"):
            m = masks[b]
            share = m.mean()
            for c, cn in ((0, "h"), (1, "a")):
                parts.append(f"{b[:1]}{cn} {share * np.mean(perp[m, c] ** 2) / 2:.4f}")
        print(f"{Path(path).name:40s} E[(P-R)^2] {tot:.4f} novel {np.mean(perp ** 2):.4f} | pooled novel by block: " + " ".join(parts))


if __name__ == "__main__":
    main()
