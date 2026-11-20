#!/usr/bin/env python
"""Everything the board has said, in one place: read ``results/board_scores.json`` (file -> public
score), turn every file into its truth functional E[F.T], and report (a) per-band, per-column
functionals wherever a file pair isolates them, (b) the greedy optimal blend over a chosen set
of base files, written as a candidate.

    ./.venv/bin/python scripts/board_registry.py [--blend a.csv,b.csv,...] [--out results/blend_x.csv]
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

E_T2 = 2.588558


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blend", default="", help="comma-separated result files to blend (default: all scored, greedy)")
    ap.add_argument("--out", default="")
    ap.add_argument("--intercept", action="store_true")
    args = ap.parse_args()
    scores = json.load(open(ROOT / "results" / "board_scores.json"))
    files = {}
    for name, sc in scores.items():
        F = shift.read_predictions(ROOT / "results" / name).astype(np.float64)
        files[name] = (F, sc, (float(np.mean(F**2)) + E_T2 - sc) / 2.0)
    keys = shift.template_keys()
    masks = shift.band_masks(keys["age"].to_numpy())
    R = files["latent_ref_y0_m114114.csv"][0] if "latent_ref_y0_m114114.csv" in files else shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    print(f"{'file':44s} {'score':>6s} {'E[F^2]':>7s} {'E[F.T]':>7s}  corr(F,T)")
    for n, (F, sc, b) in files.items():
        print(f"{n:44s} {sc:6.3f} {np.mean(F**2):7.4f} {b:7.4f}  {b / np.sqrt(np.mean(F**2) * E_T2):.4f}")
    # band probes against R: any file equal to R outside one band (and possibly one column)
    print("\nband probes (files equal to the decoded persistence outside one band):")
    for n, (F, sc, b) in files.items():
        d = np.abs(F - R) > 1e-6
        rows = d.any(1)
        if not rows.any() or rows.all():
            continue
        bands = np.unique(shift.band_of(keys["age"].to_numpy()[rows]))
        if len(bands) != 1:
            continue
        m = masks[bands[0]]
        if (rows & ~m).any():
            continue
        cols = [k for k in range(2) if d[:, k].any()]
        D = F - R
        e_t_d = b - files["latent_ref_y0_m114114.csv"][2] if "latent_ref_y0_m114114.csv" in files else None
        # E[T.R] for the reference is known from the persistence anchors: 0.8825 + 1.6083
        e_t_r = 0.8825 + 1.6083
        e_t_d = b - e_t_r
        e_r_d = float(np.mean(R * D)); e_d2 = float(np.mean(D * D))
        lam = (e_t_d - e_r_d) / e_d2
        ratio = np.median(F[m][:, cols][np.abs(R[m][:, cols]) > 0.05] / R[m][:, cols][np.abs(R[m][:, cols]) > 0.05])
        scaled = np.ptp(F[m][:, cols][np.abs(R[m][:, cols]) > 0.05] / R[m][:, cols][np.abs(R[m][:, cols]) > 0.05]) < 5e-3
        if scaled:
            # F = s R on the block: E[T.D] = (s-1) E[T.R_block] -> slope of T on R inside the block
            e_r2_block = float(np.mean((R * (np.abs(D) > 0))**2))
            e_t_rb = e_t_d / (ratio - 1.0)
            print(f"  {n:40s} band {bands[0]:5s} cols {cols} scale {ratio:.3f}: slope of truth on persistence in the block {e_t_rb / e_r2_block:.4f}")
        else:
            print(f"  {n:40s} band {bands[0]:5s} cols {cols}: E[T.(M-R)_b] {e_t_d:+.4f}, lambda*_band {lam:.3f}, gain at lambda* {-(e_t_d - e_r_d)**2 / e_d2:+.4f} (at 1: {-2 * (e_t_d - e_r_d) + e_d2:+.4f})")
    names = [n for n in (args.blend.split(",") if args.blend else files) if n in files]
    B = np.stack([files[n][0].reshape(-1) for n in names], 1)
    bv = np.array([files[n][2] for n in names])
    if args.intercept:
        B = np.concatenate([B, np.ones((B.shape[0], 1))], 1); bv = np.append(bv, 0.9933); names = names + ["const_1"]
    G = B.T @ B / B.shape[0]
    chosen: list[int] = []
    print("\ngreedy blend:")
    while len(chosen) < len(names):
        best = None
        for j in range(len(names)):
            if j in chosen:
                continue
            idx = chosen + [j]
            w = np.linalg.pinv(G[np.ix_(idx, idx)], rcond=1e-9) @ bv[idx]
            s = E_T2 - bv[idx] @ w
            if best is None or s < best[0]:
                best = (s, j, w)
        prev = E_T2 - bv[chosen] @ (np.linalg.pinv(G[np.ix_(chosen, chosen)], rcond=1e-9) @ bv[chosen]) if chosen else E_T2
        if chosen and prev - best[0] < 1e-3:
            break
        chosen.append(best[1])
        print(f"  + {names[best[1]]:44s} -> {best[0]:.4f}")
    w = np.linalg.pinv(G[np.ix_(chosen, chosen)], rcond=1e-9) @ bv[chosen]
    print("  weights:", {names[i]: round(float(x), 3) for i, x in zip(chosen, w)})
    if args.out:
        pred = np.clip((B[:, chosen] @ w).reshape(-1, 2), 0, None)
        shift.write_predictions(pred, ROOT / args.out)
        print(f"wrote {args.out}; predicted {E_T2 - bv[chosen] @ w:.4f}")


if __name__ == "__main__":
    main()
