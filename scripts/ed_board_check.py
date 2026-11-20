#!/usr/bin/env python
"""Check ED v3.0 site simulations against every scored board probe before spending a slot on them.

For each scored probe k (direction D_k = file - centre, centre B = stack_20261010 or R = decoded persistence) the
board gives lambda_k = E_pub[(T - c_k) D_k] / (rho E[D_k^2]). If ED is the truth, the same regression coefficient
computed on the simulated rows, lambda_ED_k = sum_sim (ED - c_k) D_k / sum_sim D_k^2, estimates it (up to site
sampling of the probe's support). The script prints both per probe, their correlation and the precision-weighted
slope of lambda_board on lambda_ED, and the through-origin block slopes of ED, C and the board on R.

    ./.venv/bin/python scripts/ed_board_check.py --ed-dir work/edv3/results [--alpha-h 0.876 --alpha-a 1.0]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io, shift  # noqa: E402

DIV = np.array([10.90645, 3.39567])
SIGMA_SCORE = 0.00029            # rounding noise of one 3-decimal board score
BOARD_SLOPES = {"oh": 0.891, "oa": 0.888, "mh": 0.941, "ma": 1.05}   # exact through-origin slopes T on R (public)


def read_ed(ed_dir: Path, keys, alpha: np.ndarray, form: str = "abs", R: np.ndarray | None = None
            ) -> tuple[np.ndarray, np.ndarray, int]:
    """(ED on the solution scale (N, 2), simulated-row mask, number of site-scenario files).

    form 'abs': year-40 state x alpha / divisor. form 'ratio': R x (year-40 / ED's own year-0 state from the
    companion <site>_<scen>_y0.csv) x alpha; rows whose ED year-0 state is 0 are left unsimulated.
    """
    site = keys["site"].to_numpy(); age = keys["age"].to_numpy(); scen = keys["scenario"].to_numpy()
    index = {(int(s), str(sc), int(a)): i for i, (s, sc, a) in enumerate(zip(site, scen, age))}
    ED = np.zeros((len(keys), 2)); sim = np.zeros(len(keys), dtype=bool); n = 0
    for f in sorted(glob.glob(str(ed_dir / "*_ssp*.csv"))):
        m = re.match(r"(\d+)_(ssp126|ssp585)\.csv$", Path(f).name)
        if not m:
            continue
        s, sc = int(m.group(1)), m.group(2)
        arr = np.genfromtxt(f, delimiter=",", names=True)
        if arr.size != 15 or not np.all(np.isfinite(arr["height_m"])) or not np.all(np.isfinite(arr["agb_kgC"])):
            print(f"skip {f}: incomplete")
            continue
        if form in ("ratio", "hybrid"):
            f0 = Path(f).with_name(f"{s}_{sc}_y0.csv")
            if not f0.exists():
                continue
            a0 = np.genfromtxt(f0, delimiter=",", names=True)
        for j, (a, h, b) in enumerate(zip(arr["age"], arr["height_m"], arr["agb_kgC"])):
            i = index[(s, sc, int(a))]
            if form == "abs":
                ED[i] = np.array([h, b]) * alpha / DIV
                sim[i] = True
            elif form == "ratio":
                h0, b0 = float(a0["height_m"][j]), float(a0["agb_kgC"][j])
                if h0 > 0 and b0 > 0:
                    ED[i] = R[i] * np.array([h / h0, b / b0]) * alpha
                    sim[i] = True
            else:   # hybrid: absolute year-40 state x the clipped spin-up mismatch y0 / ED year-0 (assemble_ed.py)
                e0 = np.array([float(a0["height_m"][j]), float(a0["agb_kgC"][j])])
                fac = np.where(e0 > 0, np.clip(R[i] * DIV / np.maximum(e0, 1e-12), 0.5, 2.0), 1.0)
                ED[i] = np.array([h, b]) * fac * alpha / DIV
                sim[i] = True
        n += 1
    ED[:, 0] = np.minimum(ED[:, 0], 35.0002 * alpha[0] / DIV[0])
    return np.clip(ED, 0.0, None), sim, n


def read_ed_redteam(csv_dir: Path, keys, alpha: np.ndarray, form: str, R: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Reader for scripts/ed/run_site.py output: <root>/csv/site<id>.csv, one row per (scenario, age).

    form 'abs': physical year-40 state x alpha / divisor; form 'ratio': R x (ED40 / ED0) x alpha, i.e. ED's own
    40-year change applied to the test's decoded y0 (cancels spin-up level bias). Rows with ED0 = 0 are skipped.
    """
    import csv
    site = keys["site"].to_numpy(); age = keys["age"].to_numpy(); scen = keys["scenario"].to_numpy()
    index = {(int(s), str(sc), int(a)): i for i, (s, sc, a) in enumerate(zip(site, scen, age))}
    ED = np.zeros((len(keys), 2)); sim = np.zeros(len(keys), dtype=bool); n = 0
    for f in sorted(glob.glob(str(csv_dir / "site*.csv"))):
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) != 30:
            continue
        n += 1
        for r in rows:
            i = index[(int(r["site"]), r["scenario"], int(float(r["age"])))]
            h40, a40 = float(r["ed40_height"]), float(r["ed40_agb"])
            if form == "abs":
                ED[i] = np.array([h40, a40]) * alpha / DIV; sim[i] = True
            else:
                h0, a0 = float(r["ed0_height"]), float(r["ed0_agb"])
                if h0 > 0 and a0 > 0 and np.isfinite(h40) and np.isfinite(a40):
                    ED[i] = R[i] * np.array([h40 / h0, a40 / a0]) * alpha; sim[i] = True
    ED[:, 0] = np.minimum(ED[:, 0], 35.0002 * alpha[0] / DIV[0])
    return np.clip(ED, 0.0, None), sim, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ed-dir", default="work/edv3/results")
    ap.add_argument("--probes", default="work/sim_probes_1107.npz")
    ap.add_argument("--alpha-h", type=float, default=0.876)
    ap.add_argument("--alpha-a", type=float, default=1.0)
    ap.add_argument("--base", default="results/spatcorr_stack_20261106_s08.csv")
    ap.add_argument("--min-support", type=float, default=0.10, help="minimum share of a probe's energy on simulated rows")
    ap.add_argument("--show", type=int, default=40)
    ap.add_argument("--form", default="abs", choices=["abs", "ratio", "hybrid"], help="absolute, ratio-to-y0, or absolute x clipped y0/ED0 mismatch")
    args = ap.parse_args()
    keys = shift.template_keys()
    alpha = np.array([args.alpha_h, args.alpha_a])
    if glob.glob(str(ROOT / args.ed_dir / "site*.csv")):
        Rref = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
        ED, sim, nfiles = read_ed_redteam(ROOT / args.ed_dir, keys, alpha, args.form, Rref)
    else:
        Rref = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
        ED, sim, nfiles = read_ed(ROOT / args.ed_dir, keys, alpha, args.form, Rref)
    print(f"{nfiles} ED site-scenario files, {sim.sum()} rows ({sim.mean():.1%} of the template)")
    if nfiles == 0:
        return
    z = np.load(ROOT / args.probes, allow_pickle=True)
    names = [str(x) for x in z["names"]]; rho = float(z["rho"])
    B = z["B"].astype(np.float64); R = z["R"].astype(np.float64)
    C = shift.read_predictions(ROOT / args.base).astype(np.float64)
    isR = z["cen_is_R"]; s_k = z["s_k"]; s_c = z["s_c"]
    base = np.where(isR, 0.2515, s_c)          # s_R pinned by the mirror pairs (spatial_correction.py)
    sc = json.load(open(ROOT / "results" / "board_scores.json"))
    ratio_file = ROOT / "work" / "cluster_ratio_pred.json"
    ratios = json.loads(ratio_file.read_text()) if ratio_file.exists() else {}
    rows = []
    for k, nm in enumerate(names):
        D = z["D"][k].astype(np.float64)
        e2 = float(np.mean(D ** 2))
        if e2 <= 0:
            continue
        cen = R if isR[k] else B
        mir = nm.replace(".csv", "_mirror.csv")
        if mir in sc:                       # mirrored pair: exact readout and the measured public-energy ratio
            fac = (sc[nm] + sc[mir] - 2 * base[k]) / 2 / (rho * e2)
            lam_b = -((sc[nm] - sc[mir]) / 4) / (rho * fac * e2)
            sd_b = SIGMA_SCORE / (2 * rho * fac * e2)
        else:
            fac = ratios[nm]["r"] if nm in ratios else 1.0
            y = -(s_k[k] - base[k] - rho * fac * e2) / 2
            lam_b = y / (rho * fac * e2)
            sd_b = SIGMA_SCORE * np.sqrt(2) / (2 * rho * fac * e2)
        Ds = D[sim]; supp = float(np.sum(Ds ** 2) / np.sum(D ** 2))
        if supp < args.min_support:
            continue
        lam_ed = float(np.sum((ED[sim] - cen[sim]) * Ds) / np.sum(Ds ** 2))
        lam_c = float(np.sum((C[sim] - cen[sim]) * Ds) / np.sum(Ds ** 2))
        rows.append((nm, supp, lam_b, sd_b, lam_ed, lam_c))
    if not rows:
        print("no probe has enough support on the simulated rows yet")
    else:
        rows.sort(key=lambda r: -abs(r[2]) / r[3])
        print(f"\n{'probe':40s} {'supp':>5s} {'lam_board':>10s} {'sd':>6s} {'lam_ED':>8s} {'lam_C':>8s}")
        for nm, supp, lb, sd, le, lc in rows[: args.show]:
            print(f"{nm[:40]:40s} {supp:5.2f} {lb:+10.3f} {sd:6.3f} {le:+8.3f} {lc:+8.3f}")
        lb = np.array([r[2] for r in rows]); sd = np.array([r[3] for r in rows]); le = np.array([r[4] for r in rows])
        lc = np.array([r[5] for r in rows]); w = 1 / sd ** 2
        for nm, x in (("ED", le), ("C", lc)):
            r = np.corrcoef(lb, x)[0, 1]
            slope = np.sum(w * lb * x) / np.sum(w * x * x)
            chi = np.sum(w * (lb - x) ** 2) / len(lb)
            print(f"{nm}: {len(lb)} probes, corr(lam_board, lam_{nm}) {r:+.3f}, weighted slope {slope:+.3f}, "
                  f"mean chi2 of the difference {chi:.1f}")
    # block slopes on R, outside the cluster, vegetated, simulated rows
    cl_sites = np.where(np.load(ROOT / "work" / "collapse_cluster_labels.npy") == 0)[0]
    out = ~np.isin(keys["site"].to_numpy(), cl_sites)
    masks = shift.band_masks(keys["age"].to_numpy())
    print("\nthrough-origin slopes on R (simulated rows outside the cluster, R > 0.02):")
    for b, bn in (("young", "y"), ("mid", "m"), ("old", "o")):
        for c, cn in ((0, "h"), (1, "a")):
            m = sim & out & masks[b] & (R[:, c] > 0.02)
            if m.sum() < 30:
                continue
            se = np.sum(ED[m, c] * R[m, c]) / np.sum(R[m, c] ** 2)
            sc_ = np.sum(C[m, c] * R[m, c]) / np.sum(R[m, c] ** 2)
            bs = BOARD_SLOPES.get(bn + cn)
            dE, dC = ED[m, c] - R[m, c], C[m, c] - R[m, c]
            cr = np.corrcoef(dE, dC)[0, 1] if dE.std() > 0 and dC.std() > 0 else np.nan
            print(f"  {bn}{cn}: n {m.sum():6d}  slope ED {se:.3f}  C {sc_:.3f}" + (f"  board {bs:.3f}" if bs else "              ")
                  + f" | sum ED/C {ED[m, c].sum() / C[m, c].sum():.3f}  corr(ED-R, C-R) {cr:+.2f}  rms(ED-C) {np.sqrt(np.mean((ED[m, c] - C[m, c]) ** 2)):.3f}")
    d = ED[sim] - C[sim]
    print(f"\nE[(ED - C)^2] on simulated rows {np.mean(d ** 2):.4f} (h {np.mean(d[:, 0] ** 2):.4f}, a {np.mean(d[:, 1] ** 2):.4f}); "
          f"as a share of all rows {np.sum(d ** 2) / (2 * len(C)):.5f}")


if __name__ == "__main__":
    main()
