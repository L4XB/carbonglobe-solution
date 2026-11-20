#!/usr/bin/env python
"""Read a scored batch of files as truth functionals and derive the corrections they imply.

Inputs are ``path=score`` pairs. Every file gives ``E[F.T] = (E[F^2] + E[T^2] - score) / 2``.
Two files that differ only by a factor on one column separate that column's functional; a
band probe (model on one band, reference elsewhere) gives ``E[T.(M - R)_band]`` and hence the
band's optimal damping ``lambda_b = (E[T.(M-R)_b] - E[R.(M-R)_b]) / E[(M-R)_b^2]``. With the
reference's own functionals (from the persistence anchors) the per-column optimum of
``a * M_col + b * R_col + d`` follows, and the best file of that family is written if asked.

    ./.venv/bin/python scripts/batch_interpret.py --ref results/latent_ref_y0_m114114.csv \
        --file results/latent_noco2_m114.csv=0.07 --file results/latent_noco2_m114_h0876.csv=0.06 ... \
        --band results/bandprobe_noco2_old.csv=0.17
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgb import shift  # noqa: E402

E_T2 = 2.588558
# reference functionals from the 2026-09-25 joint solve (pooled over 2N cells), for R = latent_ref_y0_m114114
E_T_RH, E_T_RA, M_H, M_A = 0.8825, 1.6083, 1.0091, 0.9775


def pooled(F: np.ndarray) -> float:
    return float(F.reshape(-1).mean())


def col(F: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(F)
    out[:, k] = F[:, k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--file", action="append", default=[], help="path=score, full files (a pair with a column-scaled twin separates columns)")
    ap.add_argument("--band", action="append", default=[], help="path=score, band-probe files built by band_probes.py from a --file model")
    args = ap.parse_args()
    R = shift.read_predictions(args.ref).astype(np.float64)
    keys = shift.template_keys()
    ages = keys["age"].to_numpy()
    masks = shift.band_masks(ages)
    files = {}
    for spec in args.file:
        path, _, score = spec.rpartition("=")
        F = shift.read_predictions(path).astype(np.float64)
        files[Path(path).stem] = (F, float(score), (pooled(F**2) + E_T2 - float(score)) / 2.0)
    print(f"{'file':40s} {'score':>7s} {'E[F^2]':>8s} {'E[F.T]':>8s}")
    for n, (F, s, b) in files.items():
        print(f"{n:40s} {s:7.4f} {pooled(F**2):8.4f} {b:8.4f}")
    # group column-scaled twins: same agb column, height column proportional
    stems = list(files)
    done = set()
    for i in stems:
        for j in stems:
            if i >= j or i in done or j in done:
                continue
            Fi, si, bi = files[i]
            Fj, sj, bj = files[j]
            if not np.allclose(Fi[:, 1], Fj[:, 1], atol=1e-6):
                continue
            nz = np.abs(Fi[:, 0]) > 0.05
            ratio = Fj[nz, 0] / Fi[nz, 0]
            if np.ptp(ratio) > 5e-3:
                continue
            k = float(np.median(ratio))
            # bi = E[T.Fh] + E[T.Fa]; bj = k E[T.Fh] + E[T.Fa]
            e_th = (bi - bj) / (1.0 - k)
            e_ta = bi - e_th
            e_h2 = pooled(col(Fi, 0) ** 2)
            e_a2 = pooled(col(Fi, 1) ** 2)
            e_rh_h = pooled(col(Fi, 0) * col(R, 0))
            e_ra_a = pooled(col(Fi, 1) * col(R, 1))
            e_rh2 = pooled(col(R, 0) ** 2)
            e_ra2 = pooled(col(R, 1) ** 2)
            print(f"\n{i} & {j} (height x{k:.4f}): E[T.F_h] {e_th:.4f} (+-{5e-4 / (1 - k):.4f})  E[T.F_a] {e_ta:.4f}")
            print(f"  optimal height factor alone: {e_th / e_h2:.4f}; optimal agb factor alone: {e_ta / e_a2:.4f}")
            best_total = E_T2
            for colk, (e_tf, e_f2, e_fr, e_r2, e_tr, m_t, name) in enumerate(
                [(e_th, e_h2, e_rh_h, e_rh2, E_T_RH, M_H, "height"), (e_ta, e_a2, e_ra_a, e_ra2, E_T_RA, M_A, "agb")]):
                mf, mr = Fi[:, colk].mean() / 2, R[:, colk].mean() / 2
                G = np.array([[e_f2, e_fr, mf], [e_fr, e_r2, mr], [mf, mr, 0.5]])
                bb = np.array([e_tf, e_tr, m_t / 2])
                w = np.linalg.solve(G, bb)
                w2 = np.linalg.solve(G[:2, :2], bb[:2])
                lam = (e_tf - e_fr - (e_tr - e_r2)) / (e_f2 - 2 * e_fr + e_r2)  # damping of the change toward R
                print(f"  {name}: a*F + b*R + d = {np.round(w, 4)} (contrib {-bb @ w:+.4f}) | no intercept {np.round(w2, 4)} | change damping lambda* {lam:.3f}")
                best_total -= bb @ w
            print(f"  -> predicted score of the per-column affine optimum: {best_total:.4f} (E[T_h^2]/E[T_a^2] split does not matter)")
            done.update({i, j})
    for spec in args.band:
        path, _, score = spec.rpartition("=")
        B = shift.read_predictions(path).astype(np.float64)
        s = float(score)
        b_band = (pooled(B**2) + E_T2 - s) / 2.0
        diff = np.abs(B - R) > 1e-6
        rows = diff.any(1)
        band = shift.band_of(ages[rows])
        bname = np.unique(band)
        m = masks[bname[0]] if len(bname) == 1 else rows
        # which model?
        src = None
        for n, (F, _, _) in files.items():
            if np.allclose(F[m], B[m], atol=1e-6):
                src = n
                break
        d = B - R  # nonzero only in the band
        e_t_d = b_band - (E_T_RH + E_T_RA)  # E[T.(M-R)_band]
        e_r_d = pooled(R * d)
        e_d2 = pooled(d * d)
        lam = (e_t_d - e_r_d) / e_d2
        # per column inside the band
        print(f"\nband probe {Path(path).stem} (band {bname}, model {src}): score {s:.4f}, E[T.(M-R)_b] {e_t_d:+.4f}, "
              f"E[R.(M-R)_b] {e_r_d:+.4f}, E[(M-R)_b^2] {e_d2:.4f} -> lambda*_band {lam:.3f} (+-{5e-4 / e_d2:.3f}); "
              f"score gain of the band at lambda*: {-(e_t_d - e_r_d) ** 2 / e_d2:+.4f} (at lambda=1: {-2 * (e_t_d - e_r_d) + e_d2:+.4f})")


if __name__ == "__main__":
    main()
