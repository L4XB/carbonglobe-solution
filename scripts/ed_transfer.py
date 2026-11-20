#!/usr/bin/env python
"""Transfer ED v3.0 site simulations to the unsimulated sites and assemble a submission candidate.

Inputs: ED outputs as ``work/edv3/results/<site>_<scenario>.csv`` with columns ``age,height_m,agb_kgC,...`` (year-40
state, one row per seed age) and the companion ``<site>_<scenario>_y0.csv`` (ED's own year-0 state), the base file C
(default ``results/spatcorr_stack_20261106_s08.csv``), and the per-column solution scale alpha (default 0.876 on BOTH
columns, the 11-07 board check on 120 random sites). Forms (``--form``):
  hybrid (default): sub_k = alpha_k * ED40_k * clip(y0_decoded_k / ED0_k, 1/cap, cap) / divisor_k  (assemble_ed.py),
                    factor 1 where ED0_k = 0; rows bare in y0 stay 0;
  abs:              sub_k = alpha_k * ED40_k / divisor_k;
  ratio:            sub_k = alpha_k * R_k * ED40_k / ED0_k (rows with ED0 = 0 left unsimulated).
``--fit-scale`` rescales each column by least squares of C on ED over the simulated rows (a proxy until the board pins alpha).

Steps:
1. read every available ED file, scale it, and mark the simulated (site, scenario) rows;
2. features per template row from the test side only: the decoded y0 state at the row's age and its two neighbouring
   seed ages, age, scenario, climate normals (seasonal T/P climatology, 40-year means of the 8 monthly drivers, six
   continuous soil statics), C and R, C - R, the chronodyn-58 and A46 anchors minus C, the latent weather effect W,
   the spread of four member files around C, the collapse-cluster flag;
3. LightGBM on the simulated rows, target e = ED - C per column, 5-fold cross-fitting grouped by site -> out-of-fold
   R2 per column and band, the calibration slope t = cov(e, e_hat) / var(e_hat) (0 if R2 <= 0);
4. candidate = ED on the simulated rows, C + t * e_hat on the others; also a transfer-only file (C on the simulated
   rows) for a separate probe, and the two directions as .npy.

Synthetic stand-in (tests the whole path before ED outputs exist): ``--synthetic results/direct_accrate5_m114.csv
--synthetic-scale 0.876 --n-sites 300`` writes fake ED files for the top-priority sites (``work/ed_site_priority.csv``)
from that member, then runs the normal path on them.

    python scripts/ed_transfer.py --ed-dir work/edv3/results --tag ed1
    python scripts/ed_transfer.py --synthetic results/direct_accrate5_m114.csv --tag synth
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io, shift  # noqa: E402

DIV = {"height": 10.90645, "agb": 3.39567}
DECODE = np.array([1.14114, 1.14114, 1.14114, 1.0, 1.14114, 1.14114, 1.14114])
AGES = np.asarray(io.SEED_AGES, dtype=float)
LOG_AGES = np.log(np.asarray(io.SEED_AGES, dtype=float))


def smooth_ratio(x40: np.ndarray, e0: np.ndarray, how: str) -> np.ndarray:
    """Smooth log(x40 / e0) across the 15 seed ages of one site-scenario (per column) and return the smoothed x40.

    'despike': running-median-of-3 replacement only where |log ratio - median| > ln 2 (fire / noise spikes; lead's
    train_eval.despike); 'median3': running median of 3 in age order; 'loess': local-linear fit in log age with a Gaussian kernel of
    bandwidth 0.45. Rows with e0 <= 0.05 or x40 <= 0 are left untouched. ED's run-to-run noise is sparse (2 of 15
    rows on the determinism tests, height only), so this is off by default; it also flattens real chronosequence
    structure (structure change ~0.002-0.003 per cell on noise-free repeat pairs, round 7).
    """
    out = x40.copy()
    if how == "despike":   # scripts/ed/train_eval.py despike(): replace log((x40+1e-6)/(e0+1e-6)) by its running median of 3
        for c in range(x40.shape[1]):   # only where it deviates from that median by more than ln 2 (isolated fire / noise spikes)
            lr = np.log((x40[:, c] + 1e-6) / (e0[:, c] + 1e-6)); med = lr.copy()
            if len(lr) > 2:
                med[1:-1] = np.median(np.stack([lr[:-2], lr[1:-1], lr[2:]]), axis=0)
            bad = np.abs(lr - med) > np.log(2.0); lr[bad] = med[bad]
            out[:, c] = np.maximum((e0[:, c] + 1e-6) * np.exp(lr) - 1e-6, 0.0)
        return out
    for c in range(x40.shape[1]):
        ok = (e0[:, c] > 0.05) & (x40[:, c] > 0)
        if ok.sum() < 4:
            continue
        r = np.log(x40[ok, c] / e0[ok, c]); la = LOG_AGES[: len(x40)][ok]
        rs = np.empty_like(r)
        for i in range(len(r)):
            if how == "median3":
                rs[i] = np.median(r[max(0, i - 1): i + 2])
            else:
                w = np.exp(-0.5 * ((la - la[i]) / 0.45) ** 2); X = np.c_[np.ones(len(la)), la - la[i]]
                rs[i] = np.linalg.lstsq(X * np.sqrt(w)[:, None], r * np.sqrt(w), rcond=None)[0][0]
        out[ok, c] = e0[ok, c] * np.exp(rs)
    return out


MEMBERS = (("results/direct_gbdt2_m114.csv", (0.876, 1.0)), ("results/direct_gbdt4x2_m114_h092.csv", (1.0, 1.0)),
           ("results/latent_noco2_m114_h0876.csv", (1.0, 1.0)), ("results/latent_latevl3_m114.csv", (0.876, 0.876)))


def rd(p: str | Path) -> np.ndarray:
    return shift.read_predictions(ROOT / p if not str(p).startswith("/") else p).astype(np.float64)


def read_ed(ed_dir: Path, keys, alpha: tuple[float, float], form: str = "hybrid", R: np.ndarray | None = None,
            cap: float = 2.0, smooth: str = "none") -> tuple[np.ndarray, np.ndarray, dict]:
    """(ED (N, 2) on the solution scale, simulated-row mask, report) from <site>_<scenario>.csv (+ _y0.csv) files."""
    site = keys["site"].to_numpy(); age = keys["age"].to_numpy(); scen = keys["scenario"].to_numpy()
    index = {(s, sc, a): i for i, (s, sc, a) in enumerate(zip(site, scen, age))}
    div = np.array([DIV["height"], DIV["agb"]]); al = np.asarray(alpha, dtype=float)
    ED = np.zeros((len(keys), 2)); sim = np.zeros(len(keys), dtype=bool); rep = {"files": 0, "rows": 0, "bad": [], "form": form}
    for f in sorted(glob.glob(str(ed_dir / "*_ssp*.csv"))):
        m = re.match(r"(\d+)_(ssp\d+)\.csv$", Path(f).name)
        if not m:
            continue                                   # companions (_y0) and anything else
        s, sc = int(m.group(1)), m.group(2)
        try:
            arr = np.atleast_1d(np.genfromtxt(f, delimiter=",", names=True, dtype=float))
        except Exception as exc:  # noqa: BLE001
            rep["bad"].append(f"{Path(f).name}: {exc}"); continue
        names = arr.dtype.names or ()
        if not {"age", "height_m", "agb_kgC"} <= set(names):
            rep["bad"].append(f"{Path(f).name}: columns {names}"); continue
        a0 = None
        if form in ("hybrid", "ratio", "raw", "add"):    # raw: despiked x40 without the y0 / e0 factor (bare stays bare); add: y0 + (x40 - e0)
            f0 = Path(f).with_name(f"{s}_{sc}_y0.csv")
            if not f0.exists():
                rep["bad"].append(f"{Path(f).name}: no _y0 companion"); continue
            a0 = np.atleast_1d(np.genfromtxt(f0, delimiter=",", names=True, dtype=float))
            if len(a0) != len(arr):
                rep["bad"].append(f"{Path(f).name}: _y0 has {len(a0)} rows"); continue
        x40_all = np.c_[np.maximum(arr["height_m"], 0.0), np.maximum(arr["agb_kgC"], 0.0)].astype(float)
        if smooth != "none" and a0 is not None and len(arr) == 15:
            x40_all = smooth_ratio(x40_all, np.c_[a0["height_m"], a0["agb_kgC"]].astype(float), smooth)
        for j, row in enumerate(arr):
            k = (s, sc, int(round(float(row["age"]))))
            if k not in index or not np.isfinite(row["height_m"]) or not np.isfinite(row["agb_kgC"]):
                rep["bad"].append(f"{Path(f).name}: age {row['age']}"); continue
            i = index[k]
            x40 = x40_all[j]
            if form in ("abs", "raw"):
                ED[i] = al * x40 / div
            elif form == "add":
                e0 = np.array([float(a0["height_m"][j]), float(a0["agb_kgC"][j])])
                ED[i] = al * (R[i] + (x40 - e0) / div)   # al scales the whole physical value (11-09: was R + al * change)
            else:
                e0 = np.array([float(a0["height_m"][j]), float(a0["agb_kgC"][j])])
                if form == "ratio":
                    if not (e0 > 0).all():
                        continue
                    ED[i] = al * R[i] * x40 / e0
                else:
                    fac = np.where(e0 > 0, np.clip(R[i] * div / np.maximum(e0, 1e-12), 1.0 / cap, cap), 1.0)
                    ED[i] = al * x40 * fac / div
            if R is not None and form != "abs":
                ED[i][R[i] == 0] = 0.0                 # bare in y0 stays bare
            sim[i] = True; rep["rows"] += 1
        rep["files"] += 1
    ED[:, 0] = np.minimum(ED[:, 0], 35.0002 * al[0] / div[0])
    return np.clip(ED, 0.0, None), sim, rep


def features(keys, C: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Test-side row features (N, F) and their names."""
    site = keys["site"].to_numpy(); age = keys["age"].to_numpy(); s5 = (keys["scenario"] == "ssp585").to_numpy()
    aidx = np.searchsorted(np.asarray(io.SEED_AGES), age)
    floors = shift.measured_floors()
    cols, names = [], []

    def add(v, nm):
        v = np.asarray(v, dtype=np.float64)
        if v.ndim == 1:
            cols.append(v); names.append(nm)
        else:
            for j in range(v.shape[1]):
                cols.append(v[:, j]); names.append(f"{nm}{j}")
    # decoded y0 state at the row's age and the neighbouring seed ages
    st = np.zeros((len(keys), 7)); nb = np.zeros((len(keys), 4))
    for s in io.SCENARIOS:
        rows = s5 == (s == "ssp585")
        y0 = np.asarray(io.open_test(s)[1], dtype=np.float64); dec = DECODE * np.clip(y0 - floors, 0.0, None)   # (6171, 15, 7)
        si, ai = site[rows], aidx[rows]
        st[rows] = dec[si, ai]
        nb[rows, :2] = dec[si, np.minimum(ai + 1, 14), :2]; nb[rows, 2:] = dec[si, np.maximum(ai - 1, 0), :2]
    add(st, "y0_"); add(nb, "nb_"); add(AGES[aidx], "age"); add(np.log(AGES[aidx]), "logage"); add(np.where(s5, 1.0, -1.0), "scen")
    # climate normals and statics
    clim = {}
    for s in io.SCENARIOS:
        ann = np.load(ROOT / "work/cache" / f"test_{s}_annual.npy", mmap_mode="r"); sea = np.load(ROOT / "work/cache" / f"test_{s}_season.npy", mmap_mode="r")
        mn = np.asarray(ann[:, :, :8], dtype=np.float64).mean(1)
        tsea = np.asarray(sea[:, 0, :], dtype=np.float64); psea = np.asarray(sea[:, 1, :], dtype=np.float64)
        stat = np.asarray(ann[:, 0, [129, 130, 131, 132, 133, 134]], dtype=np.float64)
        clim[s] = np.concatenate([mn, tsea, psea, stat], axis=1)
    cl = np.where(s5[:, None], clim["ssp585"][site], clim["ssp126"][site]); add(cl, "clim")
    # anchors, base, spread, weather effect, cluster
    add(C, "C"); add(R, "R"); add(C - R, "CmR")
    cd = rd("results/chronodyn1g58_g58.csv"); add(cd - C, "CDmC")
    a46 = 0.876 * np.load(ROOT / "work/ageshift46_dec.npy"); add(a46 - C, "A46mC")
    W = 0.876 * 0.5 * ((rd("results/latent_latelat2_actual.csv") - rd("results/latent_latelat2_clim.csv")) + (rd("results/latent_ss20L3_actual.csv") - rd("results/latent_ss20L3_clim.csv")))
    add(W, "W")
    mem = []
    for f, (ah, aa) in MEMBERS:
        X = rd(f); X[:, 0] *= ah; X[:, 1] *= aa; mem.append(X)
    mem = np.stack(mem); add(np.sqrt(((mem - C[None]) ** 2).mean(0)), "spread"); add(mem.mean(0) - C, "memmean")
    cluster = (np.load(ROOT / "work/collapse_cluster_labels.npy") == 0)[site]; add(cluster.astype(float), "cluster")
    dpath = ROOT / "work/strategist/round3/dT_climonly.npy"
    if dpath.exists():
        dT = np.load(dpath); add(np.where(s5, dT[6171:][site], dT[:6171][site]), "climdist")
    return np.stack(cols, axis=1), names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ed-dir", default="work/edv3/results")
    ap.add_argument("--base", default="results/spatcorr_stack_20261106_s08.csv")
    ap.add_argument("--alpha-h", type=float, default=0.876, help="height scale: sub = alpha_h * height_m / 10.90645")
    ap.add_argument("--alpha-a", type=float, default=0.876, help="agb scale (11-07 board check: 0.876 like height; 1.0 would sit 20-28% above the calibrated stack)")
    ap.add_argument("--form", default="hybrid", choices=["hybrid", "abs", "ratio"], help="see the module docstring")
    ap.add_argument("--cap-ratio", type=float, default=2.0, help="hybrid: clip y0/ED0 to [1/cap, cap]")
    ap.add_argument("--smooth-ratio", default="none", choices=["none", "despike", "median3", "loess"], help="smooth log(ED40/ED0) across the 15 ages per site-scenario before the hybrid (default off; despike = running-median replacement only where |dev| > ln 2)")
    ap.add_argument("--fit-scale", action="store_true", help="fit alpha_h, alpha_a by least squares of C on ED over the simulated rows (proxy)")
    ap.add_argument("--tag", default="ed1")
    ap.add_argument("--out-dir", default="work/ed_transfer")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--min-sites", type=int, default=40, help="below this many simulated sites no transfer model is fitted (ED substitution only)")
    ap.add_argument("--synthetic", default="", help="member file used as a stand-in for ED (test of the whole path)")
    ap.add_argument("--synthetic-scale", type=float, default=0.876, help="both-column factor applied to the stand-in member")
    ap.add_argument("--n-sites", type=int, default=300, help="synthetic: number of top-priority sites to fake")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    import lightgbm as lgb

    t0 = time.time()
    out_dir = ROOT / args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    keys = shift.template_keys(); site = keys["site"].to_numpy(); age = keys["age"].to_numpy(); scen = keys["scenario"].to_numpy()
    band = shift.band_of(age)
    C = rd(args.base); R = rd("results/latent_ref_y0_m114114.csv")
    ed_dir = ROOT / args.ed_dir
    if args.synthetic:
        # fake ED files for the top-priority sites from a member file (both scenarios), physical units
        prio = ROOT / "work/ed_site_priority.csv"
        if prio.exists():
            import csv
            rows = sorted(csv.DictReader(open(prio)), key=lambda r: int(r["rank"]))
            sites = [int(r["site"]) for r in rows[: args.n_sites]]
        else:
            sites = sorted(np.random.default_rng(args.seed).choice(6171, args.n_sites, replace=False).tolist())
        X = rd(args.synthetic) * args.synthetic_scale
        ed_dir = out_dir / f"synthetic_ed_{args.tag}"; ed_dir.mkdir(parents=True, exist_ok=True)
        for f in glob.glob(str(ed_dir / "*.csv")):
            Path(f).unlink()
        for s in sites:
            for sc in io.SCENARIOS:
                m = (site == s) & (scen == sc)
                with open(ed_dir / f"{s}_{sc}.csv", "w") as fh, open(ed_dir / f"{s}_{sc}_y0.csv", "w") as f0:
                    fh.write("age,height_m,agb_kgC,soil,lai,gpp,npp,rh\n"); f0.write("age,height_m,agb_kgC,soil,lai,gpp,npp,rh\n")
                    for a, h, b, r0, r1 in zip(age[m], X[m, 0], X[m, 1], R[m, 0], R[m, 1]):
                        fh.write(f"{a},{h * DIV['height'] / args.alpha_h:.6f},{b * DIV['agb'] / args.alpha_a:.6f},0,0,0,0,0\n")
                        f0.write(f"{a},{r0 * DIV['height']:.6f},{r1 * DIV['agb']:.6f},0,0,0,0,0\n")
        print(f"synthetic ED written for {len(sites)} sites from {args.synthetic} (x{args.synthetic_scale}) into {ed_dir}")
    ED, sim, rep = read_ed(ed_dir, keys, (args.alpha_h, args.alpha_a), form=args.form, R=R, cap=args.cap_ratio, smooth=args.smooth_ratio)
    n_sites = len(np.unique(site[sim]))
    print(f"ED files: {rep['files']}, rows {rep['rows']} ({n_sites} sites), unreadable {len(rep['bad'])}" + (f" e.g. {rep['bad'][:3]}" if rep["bad"] else ""), flush=True)
    if rep["rows"] == 0:
        sys.exit("no ED rows; nothing to do")
    if args.fit_scale:
        for c, nm in ((0, "height"), (1, "agb")):
            m = sim & (ED[:, c] > 0.02) & (C[:, c] > 0.02)
            k = float(C[m, c] @ ED[m, c] / (ED[m, c] @ ED[m, c]))
            print(f"  scale fit on {m.sum()} rows: alpha_{nm[0]} x {k:.4f} (C on ED through the origin); applying it")
            ED[sim, c] *= k
    # --- diagnostics of ED vs C on the simulated rows
    e = ED - C
    print("ED - C on the simulated rows: E[e^2] pooled over all cells {:.5f}; by block ".format(np.sum(e[sim] ** 2) / e.size)
          + " ".join(f"{b[0]}{'ha'[c]}:{np.mean(e[sim & (band == b), c] ** 2):.4f}" for b in ("young", "mid", "old") for c in (0, 1)), flush=True)
    cand = C.copy(); cand[sim] = ED[sim]
    transfer = C.copy()
    report = {"files": rep["files"], "rows": int(rep["rows"]), "sites": int(n_sites), "alpha": [args.alpha_h, args.alpha_a], "form": args.form, "smooth": args.smooth_ratio, "fit_scale": args.fit_scale}
    if n_sites >= args.min_sites:
        X, names = features(keys, C, R)
        print(f"features {X.shape} [{time.time() - t0:.0f}s]", flush=True)
        prm = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
                   bagging_freq=1, lambda_l2=5.0, num_threads=args.threads, verbose=-1, seed=args.seed)
        sim_sites = np.unique(site[sim]); fold_of_site = dict(zip(sim_sites, np.random.default_rng(args.seed).integers(0, args.folds, size=len(sim_sites))))
        fold = np.array([fold_of_site.get(s, -1) for s in site])
        for c, nm in ((0, "height"), (1, "agb")):
            y = e[:, c]; oof = np.zeros(len(y))
            for k in range(args.folds):
                tr = sim & (fold != k); te = sim & (fold == k)
                if tr.sum() == 0 or te.sum() == 0:
                    continue
                mdl = lgb.train(prm, lgb.Dataset(X[tr], y[tr]), args.rounds); oof[te] = mdl.predict(X[te], num_threads=args.threads)
            r2 = 1 - np.mean((oof[sim] - y[sim]) ** 2) / np.var(y[sim])
            t = float(oof[sim] @ y[sim] / max(oof[sim] @ oof[sim], 1e-30)) if r2 > 0 else 0.0
            per_band = {b: float(1 - np.mean((oof[sim & (band == b)] - y[sim & (band == b)]) ** 2) / max(np.var(y[sim & (band == b)]), 1e-30)) for b in ("young", "mid", "old")}
            full = lgb.train(prm, lgb.Dataset(X[sim], y[sim]), args.rounds)
            pred = full.predict(X[~sim], num_threads=args.threads) * t
            transfer[~sim, c] = C[~sim, c] + pred; cand[~sim, c] = C[~sim, c] + pred
            imp = full.feature_importance("gain"); top = [names[i] for i in np.argsort(-imp)[:8]]
            report[nm] = {"oof_r2": float(r2), "oof_r2_by_band": per_band, "slope_t": t, "E_e2_sim": float(np.mean(y[sim] ** 2)), "E_pred2_unsim": float(np.mean(pred ** 2)), "top_features": top}
            print(f"  {nm}: OOF R2 {r2:.3f} (young {per_band['young']:.2f} mid {per_band['mid']:.2f} old {per_band['old']:.2f}), slope t {t:.2f}; "
                  f"transfer energy on unsimulated rows {np.mean(pred ** 2):.5f} vs E[e^2] simulated {np.mean(y[sim] ** 2):.5f}; top features {top}  [{time.time() - t0:.0f}s]", flush=True)
        # expected gain of the transfer part if the OOF R2 holds on the unsimulated sites (their e-energy is lower: scale by the
        # priority ratio when the priority file exists)
        g = {nm: report[nm]["oof_r2"] * report[nm]["E_e2_sim"] for nm in ("height", "agb") if nm in report}
        print(f"  transfer gain if the simulated-site R2 holds on the others: ~{0.5 * sum(max(v, 0) for v in g.values()) * (~sim).mean():.5f} (pooled, upper bound: the unsimulated sites have smaller ED - C)")
    else:
        print(f"only {n_sites} simulated sites (< {args.min_sites}): ED substitution only, no transfer model")
    cand = np.clip(cand, 0.0, None); transfer = np.clip(transfer, 0.0, None)
    shift.write_predictions(cand, out_dir / f"cand_{args.tag}_ed_plus_transfer.csv")
    shift.write_predictions(transfer, out_dir / f"cand_{args.tag}_transfer_only.csv")
    sub_only = C.copy(); sub_only[sim] = ED[sim]; shift.write_predictions(np.clip(sub_only, 0.0, None), out_dir / f"cand_{args.tag}_ed_only.csv")
    np.save(out_dir / f"dir_{args.tag}_ed_minus_C.npy", np.where(sim[:, None], ED - C, 0.0)); np.save(out_dir / f"dir_{args.tag}_transfer_minus_C.npy", transfer - C)
    np.save(out_dir / f"sim_mask_{args.tag}.npy", sim)
    json.dump(report, open(out_dir / f"report_{args.tag}.json", "w"), indent=1)
    print(f"wrote cand_{args.tag}_ed_only.csv, cand_{args.tag}_transfer_only.csv, cand_{args.tag}_ed_plus_transfer.csv and the directions in {out_dir}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
