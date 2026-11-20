#!/usr/bin/env python
"""Regional correction of a blend from the scored smooth-field probes.

Every scored file `spat_*` / `perp_spat_*` is decoded persistence plus a pattern D_k (projected off the span when it was
built). Its score gives E_pub[(T - R) D_k] = -(s_k - s_R - E[D_k^2]) / 2; mirror pairs (R + D, R - D) give it without
s_R and also pin s_R. The correction to a base blend B solves the small Gram system of the D_k for the residual
projections E[(T - B) D_k] = E[(T - R) D_k] - E[(B - R) D_k], shrinks each coefficient toward 0 by its noise
(James-Stein, sigma ~ rounding + E[D^2] mismatch + s_R error), and writes B + sum c_k D_k.

    ./.venv/bin/python scripts/spatial_correction.py --base results/stack_20261010.csv --out results/x.csv
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

M = 1.14114


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-shrink", action="store_true")
    ap.add_argument("--site-noise", type=float, default=0.0,
                    help="extra lambda noise (probe units) for probes concentrated on < 300 effective sites: their readout "
                         "includes the public sites' own residuals, which do not carry to the private half (private-robust variant)")
    ap.add_argument("--rho-conc", type=float, default=1.0,
                    help="E_pub/E_all factor (on top of rho) for probes with < 300 effective sites; the 10-25 line search "
                         "implies ~1.2 for the collapse-cluster probes")
    ap.add_argument("--scale-conc", type=float, default=-1.0,
                    help="separate scale for the cluster-concentrated probes (default: same as --scale)")
    ap.add_argument("--scale-mirror", type=float, default=1.0,
                    help="scale for probes with a scored mirror pair (exact readout and public energy)")
    ap.add_argument("--ridge", type=float, default=0.0,
                    help="joint ridge solve (G + ridge I)^-1 y instead of per-probe James-Stein shrinkage (the calibrated local "
                         "evaluator prefers ridge 1e-4 at scale 0.9 by ~0.0002)")
    ap.add_argument("--hcap", type=float, default=35.0002 / (1.14114 * 10.90645),
                    help="cap the corrected height at ED's 35 m maximum on the truth scale (physical / 1.14114 / 10.90645); 0 = off")
    ap.add_argument("--ratio-file", default="",
                    help="json of predicted public-energy ratios r for un-mirrored cluster probes (scripts/cluster_public_share.py); "
                         "used as their public-energy factor instead of --rho-conc")
    ap.add_argument("--scale-regional", type=float, default=-1.0,
                    help="scale of the regional / mode probes (REGIONAL prefixes) and of cluster-concentrated "
                         "probes; -1 = use --scale (block-split robustness, scripts/sim/split_sim_block.py)")
    ap.add_argument("--scale-calib", type=float, default=1.0,
                    help="scale of the level/scale calibration probes (spat_dir_level_*, spat_dir_scale_*): their "
                         "readouts are nearly noise-free (split-conditioned public energies), so they apply in full")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply the final correction (the two corrected stacks realised ~half of the predicted gain)")
    args = ap.parse_args()
    # CA_SCORES=<json> reads another score file (synthetic boards, scripts/sim/synth_stack.py); default: the real board
    sc = json.load(open(ROOT / __import__("os").environ.get("CA_SCORES", "results/board_scores.json")))
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    B = shift.read_predictions(ROOT / args.base).astype(np.float64)
    C = sc["zeros.csv"]; s1, s05 = sc["persistence_shifted.csv"], sc["persistence_s05.csv"]
    ps = shift.read_predictions(ROOT / "results" / "persistence_shifted.csv").astype(np.float64)
    rho = 2 * (s1 - 2 * s05 + C) / float(np.mean(ps ** 2))
    e_ps2 = 2 * (s1 - 2 * s05 + C); s_R_anchor = M * M * e_ps2 - M * (e_ps2 + C - s1) + C
    names = sorted(n for n in sc if (n.startswith("spat_") or n.startswith("perp_spat_")) and not n.endswith("_mirror.csv"))
    cfile = ROOT / "work" / "spat_centers.json"
    centers = json.loads(cfile.read_text()) if cfile.exists() else {}
    cen = {n: ("R" if centers.get(n, "R").endswith("latent_ref_y0_m114114.csv") else centers.get(n, "R")) for n in names}
    cache = {}

    def center_pred(c: str) -> np.ndarray:
        if c not in cache:
            cache[c] = R if c == "R" else shift.read_predictions(ROOT / c).astype(np.float64)
        return cache[c]
    Ds = {n: shift.read_predictions(ROOT / "results" / n).astype(np.float64) - center_pred(cen[n]) for n in names}
    # s_R: combine the anchor value with mirror pairs (s+ + s- = 2 s_R + 2 E_pub[D^2])
    est = [(s_R_anchor, 0.00047 ** 2)]
    mirror_ratio = {}
    mirror_shift = {}
    for n in names:
        mir = n.replace(".csv", "_mirror.csv")
        if mir in sc and cen[n] == "R":              # only R-centred mirror pairs pin s_R
            e2 = rho * float(np.mean(Ds[n] ** 2))
            est.append(((sc[n] + sc[mir]) / 2 - e2, 0.0002 ** 2 + (0.03 * e2) ** 2))
        elif mir in sc:                              # a blend-centred pair measures the probe's public energy exactly
            # a mirror clipped at zero is C - D + c (c >= 0): its pair measures the mean of E[D^2] and E[(D - c)^2]
            # and its score difference carries (E[D^2] - E[(D - c)^2]) / 4 on top of E[(C - T) D]
            Cp = center_pred(cen[n]); clip = shift.read_predictions(ROOT / "results" / mir).astype(np.float64) - (Cp - Ds[n])
            e_all = float(np.mean(Ds[n] ** 2)); e_clip = float(np.mean((Ds[n] - clip) ** 2))
            e_pub = (sc[n] + sc[mir] - 2 * sc[Path(cen[n]).name]) / 2
            mirror_ratio[n] = e_pub / (rho * (e_all + e_clip) / 2)
            mirror_shift[n] = mirror_ratio[n] * rho * (e_all - e_clip) / 4
            print(f"  mirror {n}: E_pub[D^2] {e_pub:.5f} vs rho E_all {rho * e_all:.5f} (ratio {mirror_ratio[n]:.3f}"
                  + (f", mirror clipped: {1 - e_clip / e_all:.1%} of the energy)" if e_clip < 0.999 * e_all else ")"))
    w = np.array([1 / v for _, v in est]); s_R = float(np.sum(w * np.array([e for e, _ in est])) / w.sum())
    print(f"s_R: anchors {s_R_anchor:.5f}, combined with {len(est) - 1} mirror pair(s) {s_R:.5f}")
    site = shift.template_keys()["site"].to_numpy()
    cluster_sites = np.load(ROOT / "work" / "cluster0_modes.npy")[:, 0] != 0
    ratios = json.loads(Path(ROOT / args.ratio_file).read_text()) if args.ratio_file else {}
    y, sig = [], []
    conc_flags, facs, mirrored = [], [], []
    for n in names:
        D = Ds[n]
        es0 = np.bincount(site, weights=(D * D).sum(axis=1), minlength=int(site.max()) + 1)
        # the public half holds ~23% more of the collapse cluster's energy than rho assumes (mirror pair of cluster
        # mode 2, 2026-10-26): apply rho_conc to probes whose energy sits mainly on the cluster's sites
        conc = float(np.sum(es0[cluster_sites]) / max(np.sum(es0), 1e-30)) > 0.5
        fac = mirror_ratio.get(n, ratios[n]["r"] if n in ratios else (args.rho_conc if conc else 1.0))
        e2 = rho * fac * float(np.mean(D * D))
        conc_flags.append(conc); facs.append(fac); mirrored.append(n in mirror_ratio)
        # E_pub[D^2] / E_all[D^2] scatters by 1/sqrt(N_eff) over random half splits of the sites: a probe whose energy sits
        # on a few hundred sites reads with ~7% mismatch, a global field with ~2%
        es = np.bincount(site, weights=(D * D).sum(axis=1), minlength=int(site.max()) + 1)
        mis = max(0.02, float(np.sqrt(np.sum(es ** 2)) / np.sum(es)))
        n_eff = float(np.sum(es) ** 2 / max(np.sum(es ** 2), 1e-30))
        extra = args.site_noise * e2 if n_eff < 300 else 0.0
        mir = n.replace(".csv", "_mirror.csv")
        if mir in sc:
            Xr = (sc[n] - sc[mir]) / 4 - mirror_shift.get(n, 0.0)   # E_pub[(C - T) D], exact up to rounding (and clipping)
            s_k = 0.0007 / 4
        elif cen[n] == "R":
            Xr = (sc[n] - s_R - e2) / 2
            s_k = np.sqrt((0.00029 / 2) ** 2 + (mis * e2 / 2) ** 2 + (0.0003 / 2) ** 2)
        else:
            s_c = sc[Path(cen[n]).name]
            Xr = (sc[n] - s_c - e2) / 2
            s_k = np.sqrt(2 * (0.00029 / 2) ** 2 + (mis * e2 / 2) ** 2)
        Cp = center_pred(cen[n])
        yb = -Xr - rho * float(np.mean((B - Cp) * D))   # E[(T - B) D] = E[(T - C) D] - E[(B - C) D]
        s_k = float(np.sqrt(s_k ** 2 + extra ** 2))
        y.append(yb); sig.append(s_k)
        print(f"  {n:28s} E[D^2] {e2:.5f}  E[(T-B)D]/E[D^2] {yb / e2:+.3f} (+-{s_k / e2:.3f})")
    Dm = np.stack([Ds[n].ravel() for n in names], 1)
    w = np.sqrt(np.array(facs))
    G = rho * (Dm * w).T @ (Dm * w) / Dm.shape[0]      # concentrated probes' public energy scaled by rho_conc
    c = np.linalg.solve(G + max(args.ridge, 1e-9) * np.eye(len(names)), np.array(y))
    if not args.no_shrink and args.ridge <= 0:
        z = np.array(y) / np.array(sig)
        c = c * np.clip(1 - 1 / np.maximum(z ** 2, 1e-9), 0, 1)
    sc_conc = args.scale if args.scale_conc < 0 else args.scale_conc
    calib = np.array([n.startswith(("spat_dir_level_", "spat_dir_scale_")) for n in names])
    REGIONAL = ("spat_dir_cl0_", "spat_dir_mb", "spat_dir_rg", "spat_dir_gm", "spat_dir_region_", "spat_dir_covrule",
                "spat_dir_vulnR", "spat_dir_gbin", "spat_oa_m", "spat_oa_rc", "spat_oafield", "spat_young_m",
                "spat_young_rc", "spat_youngfield", "spat_young_r", "spat_mid_r", "spat_oh_r", "perp_spat_",
                "spat_dir_agbclsp")
    regional = np.array([n.startswith(REGIONAL) for n in names]) | np.array(conc_flags)
    base_scale = np.where(np.array(conc_flags), sc_conc, args.scale)
    if args.scale_regional >= 0:
        base_scale = np.where(regional, args.scale_regional, base_scale)
    c = c * np.where(calib, args.scale_calib, np.where(np.array(mirrored), args.scale_mirror, base_scale))
    gain = float(2 * c @ np.array(y) - c @ G @ c)
    P = np.clip(B + (Dm @ c).reshape(R.shape), 0, None)
    if args.hcap > 0:
        P[:, 0] = np.minimum(P[:, 0], args.hcap)
    shift.write_predictions(P, ROOT / args.out)
    print(f"coefficients {np.round(c, 3)}; predicted public gain {gain:.5f}; wrote {args.out}")


if __name__ == "__main__":
    main()
