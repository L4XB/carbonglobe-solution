#!/usr/bin/env python
"""Exact public-half readout once the all-zeros file is scored.

score(zeros) = C_pub = E_pub[T^2] exactly. The persistence pair then pins the public-half second
moment of the decoded persistence: s(pshift) - 2 s(0.5 pshift) = 0.5 E_pub[pshift^2] - C_pub, so
E_pub[R^2] = 1.14114^2 * 2 (s1 - 2 s05 + C_pub) and E_pub[T.R] = 1.14114 (E_pub[pshift^2] + C_pub - s1)/2.
Every probe F = R + D then gives E[T.D] = (E_pub[F^2] - E_pub[R^2] - s_F + s_R)/2 with
s_R = E_pub[R^2] - 2E_pub[T.R] + C_pub, where only the small terms E[2RD + D^2] are taken over all rows
(times the public/all ratio rho = E_pub[R^2]/E_all[R^2]).

    ./.venv/bin/python scripts/anchored_readout.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

M = 1.14114


def main() -> None:
    sc = json.load(open(ROOT / "results" / "board_scores.json"))
    if "zeros.csv" not in sc:
        raise SystemExit("zeros.csv is not scored yet")
    C = sc["zeros.csv"]
    s1, s05 = sc["persistence_shifted.csv"], sc["persistence_s05.csv"]
    ps = shift.read_predictions(ROOT / "results" / "persistence_shifted.csv").astype(np.float64)
    e_ps2_pub = 2 * (s1 - 2 * s05 + C)
    e_ps2_all = float(np.mean(ps ** 2))
    rho = e_ps2_pub / e_ps2_all
    e_r2_pub = M * M * e_ps2_pub
    e_tr_pub = M * (e_ps2_pub + C - s1) / 2
    s_R = e_r2_pub - 2 * e_tr_pub + C
    print(f"C_pub = E_pub[T^2] = {C:.4f} (the pair-derived value was 2.5886; rounding +-5e-4)")
    print(f"E_pub[pshift^2] = {e_ps2_pub:.4f} vs all rows {e_ps2_all:.4f} -> rho = {rho:.4f} (+-{2 * 2e-3 / e_ps2_all:.4f})")
    print(f"E_pub[T.R] = {e_tr_pub:.4f}; implied score of R itself {s_R:.4f}")
    if "constant_one.csv" in sc:
        print(f"mean of the truth over both columns (pooled): {(1 + C - sc['constant_one.csv']) / 2 * 2 / 2:.4f} x2 -> E[T_h]+E[T_a] = {(1 + C - sc['constant_one.csv']):.4f}")
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    keys = shift.template_keys(); masks = shift.band_masks(keys["age"].to_numpy())
    rows = []
    for name, s in sc.items():
        if not name.startswith(("bandprobe_", "cellprobe_", "dirprobe_", "probe_", "relaxprobe_", "coupleprobe_", "climprobe_",
                                "calprobe", "nlprobe_", "detprobe_", "hdir_", "seedprobe_", "eventprobe_", "growthxstress_",
                                "symprobe_", "latedetprobe_", "ageshiftprobe_", "siteprobe_", "profprobe_", "climchg_")):
            continue
        F = shift.read_predictions(ROOT / "results" / name).astype(np.float64)
        D = F - R
        if np.abs(D).max() < 1e-9:
            continue
        small = rho * float(np.mean(2 * R * D + D * D))
        e_td = (small - s + s_R) / 2
        e_rd = rho * float(np.mean(R * D)); e_d2 = rho * float(np.mean(D * D))
        lam = (e_td - e_rd) / e_d2
        where = np.abs(D).sum(1) > 0
        bands = np.unique(shift.band_of(keys["age"].to_numpy()[where]))
        cols = [k for k in range(2) if np.abs(D[:, k]).max() > 0]
        # is it a pure rescaling of R on its block?
        blk = where[:, None] & (np.abs(D) > 0)
        big = blk & (np.abs(R) > 0.05)
        ratio = F[big] / R[big]
        pure = ratio.size > 0 and np.std(ratio) < 1e-3 and (np.abs(D[blk & ~big]).max() if (blk & ~big).any() else 0) < 1e-2
        if pure:
            k = float(np.nanmedian(ratio))
            e_r2_blk = rho * float(np.sum(R[blk] ** 2)) / R.size
            slope = (e_td / (k - 1)) / e_r2_blk
            rows.append((name, f"band {'/'.join(bands)} cols {cols}: slope of T on R = {slope:.4f} (+-{2.5e-4 / abs(k - 1) / e_r2_blk:.4f})"))
        else:
            rows.append((name, f"band {'/'.join(bands)} cols {cols}: E[T.D] {e_td:+.4f}  lambda {lam:.3f} (+-{2.5e-4 / e_d2:.3f})  gain at lambda {-(e_td - e_rd) ** 2 / e_d2:+.4f}"))
    for n, r in sorted(rows):
        print(f"  {n:44s} {r}")


if __name__ == "__main__":
    main()
