#!/usr/bin/env python
"""ED probe and candidate builder for day 47 on (generalizes work/strategist/round7/build_day46_ed.py's variant parts).

Directions are C-centred: D = (ED - C) on simulated rows that pass the offset-log gate |log((y0 + eps) / (e0 + eps))| <= ln 2
(eps 0.1 m / 0.01 kgC, e0 = that run's own year-0 state), C = results/spatcorr_stack_20261106_s08.csv. Readout forms per site
group (training evidence, notes 11-08): cluster and warm heights raw x40, agb hybrid x40 * clip(y0 / e0, 0.5, 2); cold sites
additive y0 + (x40 - e0) in both columns (boreal training MSE 0.0008 / 0.0011 vs hybrid 0.0013 / 0.0016); near-bare hybrid.
Groups come from work/edv3/order_v4.txt (204 cluster, 2162 warm, 2649 cold, 1156 near-bare).

Part names: <group>_<band>_<cols>_<variant>[_new]
    group    cl | warm | cold | bare | out (warm + cold + bare) | all (every group)
    band     young | mid | old | ym | all
    cols     h | a | ha, optionally + c (scenario contrast: the antisymmetric part (D585 - D126) / 2 with opposite signs on the
             two scenario rows) or + s (the symmetric part (D585 + D126) / 2 on both rows); D = D_s + D_c, the two are orthogonal
    variant  v4 (work/edv3/full_v4) | v4asis (work/edv3/cl_v4asis) | c400a (work/edv3/full_c400a)
    _new     only rows that no file in --exclude-files touches (files that differ from C there)
    _lf      LEVEL-FREE: D minus its mean over the part's rows per (age, scenario, column) block, so the probe carries only
             ED's within-block pattern (day-46 readout: the ED-vs-C level offsets, ~25% in heights, swamp the probes)
    _tNN     only rows whose seed matches y0 within |log| < NN/100 in BOTH columns (e.g. _t05); _uNN the other gated rows
             (suffix order: ..._<variant>[_new][_tNN|_uNN][_lf])
Probes:     --picks p1,p2,... -> <out-dir>/<prefix>_<part>.csv = C + D_part, manifest lines <file>\\t<message>
Candidate:  --candidate p1=w1,p2=w2,... [--base B] -> --candidate-out; weights are C-centred (single readouts or a joint fit);
            the transfer to the base is solved jointly: w_b = w_C - G^-1 <D, base - C>.
    python scripts/ed/build_ed_probes.py --picks warm_old_ha_v4_new,cold_ym_ha_v4 \\
        --prefix spat_ed47 --out-dir results --register --manifest work/strategist/round7/day47_manifest.txt
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402
import ed_transfer as et  # noqa: E402

DIV = np.array([10.90645, 3.39567])
EPS = np.array([0.1, 0.01])
S_C = 0.0438
VARIANTS = {"v4": ROOT / "work/edv3/full_v4/results", "v4asis": ROOT / "work/edv3/cl_v4asis/results",
            "c400a": ROOT / "work/edv3/full_c400a/results", "tn": ROOT / "work/edv3/cl_tn/results",
            "tqn": ROOT / "work/edv3/cl_tqn/results", "nx": ROOT / "work/edv3/cl_nx/results", "cx": ROOT / "work/edv3/cl_cx3/results", "cx1": ROOT / "work/edv3/cl_cx/results"}
VDESC = {"v4": "calibrated selective fire (fp1 1e-4, cap 0.2) in the 40-year windows, every test-input artefact undone (real SSP CO2, precipitation floor removed), fire-free seeds",
         "v4asis": "calibrated selective fire (fp1 1e-4, cap 0.2), inputs AS DELIVERED (stored-channel CO2, precipitation floor kept), fire-free seeds",
         "c400a": "fire fp1 1 cap 0.1 in the windows (cluster/warm), stored-channel CO2, fire-free seeds",
         "tn": "as v4 (selective fire, real SSP CO2, precipitation floor removed) with the test TEMPERATURES shifted to the NEX-GDDP GFDL-ESM4 level",
         "tqn": "as v4 (selective fire, real SSP CO2, precipitation floor removed) with the test TEMPERATURES and SPECIFIC HUMIDITY moved to the NEX-GDDP GFDL-ESM4 level",
         "nx": "as v4 (selective fire, real SSP CO2, precipitation floor removed) with the site positions (latitude, geographic twin for the seeds / cluster spin-up) from NEX-GDDP weather-anomaly geolocation",
         "cx": "ED v3.0 with EXACT inputs: every released driver channel mapped back through the per-channel affine release transform (fitted on the exact training-cell twins of the test sites), exact cell position, real SSP CO2, selective fire in the windows"}
# solution scale per variant: (height, agb) factors on ED's physical values. 11-09: the truth codes BOTH columns as
# physical / 1.14114 (exact-input ED at non-burning sites: heights within 1% of C x 1.141, old agb within 3%); the
# variants before "cx" were built with (1, 1) and keep it for reproducibility of the day-46 files.
ALPHA = {"cx": (1.0 / 1.14114, 1.0 / 1.14114), "cx1": (1.0 / 1.14114, 1.0 / 1.14114)}
FORMS = {"cl": ("raw", "hybrid"), "warm": ("raw", "hybrid"), "cold": ("add", "add"), "bare": ("hybrid", "hybrid")}
GDESC = {"cl": "collapse-cluster sites", "warm": "warm non-cluster sites", "cold": "cold non-cluster sites", "bare": "near-bare sites",
         "out": "all non-cluster sites", "all": "all simulated sites"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--centre", default="results/spatcorr_stack_20261106_s08.csv")
    ap.add_argument("--picks", default="")
    ap.add_argument("--prefix", default="spat_ed47")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--exclude-files", default="", help="comma list of earlier probe files; parts with _new skip the rows they touch")
    ap.add_argument("--npy", action="append", default=[], help="extra probe NAME=work/dir_x.npy (a (rows, 2) C-centred direction), written as <prefix>_NAME.csv")
    ap.add_argument("--npy-msg", action="append", default=[], help="NAME=message for an --npy probe")
    ap.add_argument("--candidate", default="", help="comma list part=w (C-centred weights)")
    ap.add_argument("--base", default="", help="base of the candidate (default: the centre)")
    ap.add_argument("--candidate-out", default="")
    ap.add_argument("--candidate-msg", default="")
    args = ap.parse_args()

    keys = shift.template_keys(); site = keys["site"].to_numpy(); band = shift.band_of(keys["age"].to_numpy())
    rd = lambda p: shift.read_predictions(ROOT / p).astype(np.float64)
    C, R = rd(args.centre), rd("results/latent_ref_y0_m114114.csv")
    order = [int(x) for x in (ROOT / "work/edv3/order_v4.txt").read_text().split()]
    gsite = np.empty(6171, object)
    for i, s in enumerate(order):
        gsite[s] = "cl" if i < 204 else "warm" if i < 2366 else "cold" if i < 5015 else "bare"
    grow = gsite[site]
    scen = keys["scenario"].to_numpy(); is585 = scen == "ssp585"
    pidx = {(int(s_), str(sc_), int(a_)): i for i, (s_, sc_, a_) in enumerate(zip(site, scen, keys["age"]))}
    partner = np.array([pidx[(int(s_), "ssp126" if sc_ == "ssp585" else "ssp585", int(a_))] for s_, sc_, a_ in zip(site, scen, keys["age"])])
    print(f"build_ed_probes at {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC, centre {args.centre}")

    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    mism: dict[str, np.ndarray] = {}

    def direction(var: str) -> tuple[np.ndarray, np.ndarray]:
        """Gated C-centred direction (rows, 2) of one variant with the per-group forms, and its simulated-row mask."""
        if var in cache:
            return cache[var]
        d = VARIANTS[var]
        reads = {f: et.read_ed(d, keys, ALPHA.get(var, (1.0, 1.0)), form=f, R=R, smooth="despike") for f in ("raw", "hybrid", "add")}
        sim = reads["hybrid"][1]
        ED = np.zeros_like(C)
        for g, (fh, fa) in FORMS.items():
            m = grow == g
            ED[m, 0] = reads[fh][0][m, 0]; ED[m, 1] = reads[fa][0][m, 1]
        e0 = np.zeros_like(C)
        index = {(int(s), str(sc), int(a)): i for i, (s, sc, a) in enumerate(zip(site, keys["scenario"], keys["age"]))}
        for f in glob.glob(str(d / "*_y0.csv")):
            mm = re.match(r"(\d+)_(ssp\d+)_y0\.csv$", Path(f).name)
            for row in np.atleast_1d(np.genfromtxt(f, delimiter=",", names=True, dtype=float)):
                k = (int(mm.group(1)), mm.group(2), int(round(row["age"])))
                if k in index:
                    e0[index[k]] = [row["height_m"], row["agb_kgC"]]
        gate = np.abs(np.log((R * DIV + EPS) / (np.maximum(e0, 0.0) + EPS))) <= np.log(2.0)
        D = np.where(sim[:, None] & gate, ED - C, 0.0)
        ns = {g: len(np.unique(site[sim & (grow == g)])) for g in FORMS}
        print(f"  variant {var} ({d.parent.name}): sites {ns}; gate pass on simulated rows h {gate[sim, 0].mean():.3f} a {gate[sim, 1].mean():.3f}")
        mism[var] = np.abs(np.log((np.maximum(e0, 0.0) + EPS) / (R * DIV + EPS))).max(axis=1)   # worse column per row
        cache[var] = (D, sim)
        return cache[var]

    excl = np.zeros(len(keys), bool)
    for f in filter(None, args.exclude_files.split(",")):
        excl |= np.any(np.abs(rd(f) - C) > 1e-9, axis=1)
    if args.exclude_files:
        print(f"  exclusion: {int(excl.sum())} rows ({len(np.unique(site[excl]))} sites) touched by {args.exclude_files}")

    def part(name: str) -> np.ndarray:
        bits = name.split("_")
        lf = bits[-1] == "lf"
        if lf:
            bits = bits[:-1]
        tight = None
        if re.fullmatch(r"[tu]\d\d", bits[-1]):
            tight = (bits[-1][0], int(bits[-1][1:]) / 100.0); bits = bits[:-1]
        new = bits[-1] == "new"
        if new:
            bits = bits[:-1]
        g, b, c, var = bits
        D, _ = direction(var)
        rows = (grow != "cl") if g == "out" else np.ones(len(keys), bool) if g == "all" else (grow == g)
        rows = rows & (np.ones(len(keys), bool) if b == "all" else (band != "old") if b == "ym" else (band == b))
        if new:
            rows = rows & ~excl
        if tight is not None:
            ok = mism[var] < tight[1]
            rows = rows & (ok if tight[0] == "t" else ~ok)
        comp = c[-1] if c[-1] in "cs" and c not in ("h", "a", "ha") else ""
        cols = {"h": [True, False], "a": [False, True], "ha": [True, True]}[c[:-1] if comp else c]
        X = np.where(rows[:, None] & np.array(cols)[None, :], D, 0.0)
        if comp:
            Xp = X[partner]
            X = (X - Xp) / 2 if comp == "c" else (X + Xp) / 2     # contrast: + on the ssp585 row, - on its ssp126 partner
        if lf:
            for col in (0, 1):
                on = X[:, col] != 0
                for a in np.unique(keys["age"]):
                    for scn in ("ssp126", "ssp585"):
                        q = on & (keys["age"].to_numpy() == a) & (scen == scn)
                        if q.any():
                            X[q, col] -= X[q, col].mean()
        return X

    def describe(name: str) -> str:
        bits = name.split("_"); lf = bits[-1] == "lf"
        if lf:
            bits = bits[:-1]
        tdesc = ""
        if re.fullmatch(r"[tu]\d\d", bits[-1]):
            tdesc = (f", only rows whose seed matches y0 within {int(bits[-1][1:])}% in both columns" if bits[-1][0] == "t"
                     else f", only rows whose seed misses y0 by >= {int(bits[-1][1:])}% in a column"); bits = bits[:-1]
        new = bits[-1] == "new"
        g, b, c, var = bits[:4]
        bd = {"young": "young rows", "mid": "mid rows", "old": "old rows", "ym": "young+mid rows", "all": "all bands"}[b]
        comp = c[-1] if c[-1] in "cs" and c not in ("h", "a", "ha") else ""
        cd = {"h": "HEIGHT column only", "a": "AGB column only", "ha": "both columns"}[c[:-1] if comp else c]
        cd += {"": "", "c": ", SCENARIO CONTRAST part only ((D585 - D126) / 2, opposite signs)", "s": ", scenario-symmetric part only ((D585 + D126) / 2)"}[comp]
        fh, fa = FORMS.get(g, ("mixed", "mixed"))
        return (f"PROBE nested on C (spatcorr_stack_20261106_s08): ED v3.0 year-40 state on {GDESC[g]}, {bd}, {cd}"
                f"{', sites not in earlier probes' if new else ''}{tdesc}{', LEVEL-FREE (block means per age x scenario x column removed: within-block pattern only)' if lf else ''}; {VDESC[var]}; readout heights {fh}, agb {fa}; offset-log gate ln 2")

    # public-energy model of the day-46 builder (mirror-probe posterior of the cluster's public share)
    sc_ = json.load(open(ROOT / "results/board_scores.json")); centers_all = json.load(open(ROOT / "work/spat_centers.json"))
    clsite = np.load(ROOT / "work/collapse_cluster_labels.npy") == 0
    ps_ = rd("results/persistence_shifted.csv")
    rho = 2 * (sc_["persistence_shifted.csv"] - 2 * sc_["persistence_s05.csv"] + sc_["zeros.csv"]) / float(np.mean(ps_ ** 2))
    W_, r_ = [], []
    for n_ in sorted(sc_):
        if not n_.startswith("spat_") or n_.endswith("_mirror.csv"):
            continue
        c_ = centers_all.get(n_, "R"); isR = c_ == "R" or c_.endswith("latent_ref_y0_m114114.csv"); mir = n_.replace(".csv", "_mirror.csv")
        if not (mir in sc_ and not isR):
            continue
        Cp = rd(c_); Dk = rd("results/" + n_) - Cp; es_ = np.bincount(site, weights=(Dk * Dk).sum(1), minlength=6171)
        if es_[clsite].sum() / es_.sum() <= 0.5:
            continue
        clip_ = rd("results/" + mir) - (Cp - Dk); ea = float(np.mean(Dk * Dk)); ec = float(np.mean((Dk - clip_) ** 2))
        r_.append(((sc_[n_] + sc_[mir] - 2 * sc_[Path(c_).name]) / 2) / (rho * (ea + ec) / 2)); W_.append(es_[clsite] / es_[clsite].sum())
    W_, r_ = np.stack(W_), np.array(r_); sp_ = 0.5 * rho; S0 = 0.25
    Sig_ = S0 * W_ @ W_.T + np.diag(np.full(len(r_), (0.03 * sp_) ** 2)); K_ = np.linalg.solve(Sig_, W_).T
    post = 0.5 + S0 * K_ @ (r_ * sp_ - W_ @ np.full(W_.shape[1], 0.5))
    Vpost = S0 * np.eye(W_.shape[1]) - S0 ** 2 * W_.T @ np.linalg.solve(Sig_, W_)

    def pub_energy(D: np.ndarray) -> tuple[float, float, float, float]:
        es_ = np.bincount(site, weights=(D * D).sum(1), minlength=6171); e_all = float(np.mean(D * D)); e_cl = es_[clsite].sum() / D.size; e_out = e_all - e_cl
        w = es_[clsite] / max(es_[clsite].sum(), 1e-30)
        r_cl = float(w @ post) / 0.5 if es_[clsite].sum() > 0 else 1.0
        sd_cl = float(np.sqrt(max(w @ Vpost @ w, 0))) / 0.5 if es_[clsite].sum() > 0 else 0.0
        n_out = int((es_[~clsite] > 0).sum()); e_pub = rho * (e_out + e_cl * r_cl)
        sd = rho * np.sqrt((e_out * 0.5 / np.sqrt(max(n_out, 1))) ** 2 + (e_cl * sd_cl) ** 2)
        sd_lam = float(np.sqrt((0.00029 / (2 * e_pub)) ** 2 + (0.0003 / (2 * e_pub)) ** 2 + (0.5 * sd / e_pub) ** 2)) if e_pub > 0 else float("nan")
        return e_pub, sd, sd_lam, r_cl

    out = ROOT / args.out_dir; out.mkdir(parents=True, exist_ok=True)
    centers = json.loads((ROOT / "work/spat_centers.json").read_text()) if args.register else {}
    manifest = []
    for name in filter(None, args.picks.split(",")):
        D = part(name); fn = f"{args.prefix}_{name}.csv"
        if not np.any(D != 0):
            print(f"  {fn}: EMPTY, skipped"); continue
        shift.write_predictions(np.clip(C + D, 0, None), out / fn)
        if args.register:
            centers[fn] = args.centre
        e_pub, sd_e, sd_lam, r_cl = pub_energy(D)
        print(f"  {fn}: {len(np.unique(site[np.any(D != 0, 1)]))} sites; E_all[D^2] {np.mean(D ** 2):.5f} -> E_pub {e_pub:.5f} +- {sd_e:.5f} "
              f"(r_cl {r_cl:.3f}); sd(lambda) {sd_lam:.3f}; score if lambda 0 / 1: {S_C + e_pub:.4f} / {S_C - e_pub:.4f}; blocks "
              + " ".join(f"{b[0]}{'ha'[c]}:{np.sum(D[band == b, c] ** 2) / D.size:.4f}" for b in ("young", "mid", "old") for c in (0, 1) if np.sum(D[band == b, c] ** 2) > 0))
        manifest.append(f"{args.out_dir}/{fn}\t{describe(name)}")
    msgs_npy = dict(x.split("=", 1) for x in args.npy_msg)
    for item in args.npy:
        name, path = item.split("=", 1)
        D = np.load(ROOT / path).astype(np.float64); fn = f"{args.prefix}_{name}.csv"
        assert D.shape == C.shape, D.shape
        shift.write_predictions(np.clip(C + D, 0, None), out / fn)
        if args.register:
            centers[fn] = args.centre
        e_pub, sd_e, sd_lam, r_cl = pub_energy(D)
        print(f"  {fn}: {len(np.unique(site[np.any(D != 0, 1)]))} sites; E_all[D^2] {np.mean(D ** 2):.5f} -> E_pub {e_pub:.5f} +- {sd_e:.5f} "
              f"(r_cl {r_cl:.3f}); sd(lambda) {sd_lam:.3f}; score if lambda 0 / 1: {S_C + e_pub:.4f} / {S_C - e_pub:.4f}")
        manifest.append(f"{args.out_dir}/{fn}\t{msgs_npy.get(name, 'PROBE nested on C (spatcorr_stack_20261106_s08): direction ' + path)}")
    if args.register and manifest:
        (ROOT / "work/spat_centers.json").write_text(json.dumps(centers, indent=1)); print("  centres registered in work/spat_centers.json")
    if args.manifest and manifest:
        Path(ROOT / args.manifest).write_text("\n".join(manifest) + "\n"); print(f"  manifest {args.manifest}: {len(manifest)} probes")

    if args.candidate:
        base = rd(args.base) if args.base else C
        names, w_c, Ds = [], [], []
        for item in args.candidate.split(","):
            nm, w = item.split("="); names.append(nm); w_c.append(float(w)); Ds.append(part(nm))
        G = np.array([[float(np.sum(a * b)) for b in Ds] for a in Ds])
        shiftv = np.array([float(np.sum((base - C) * Dp)) for Dp in Ds])
        w_b = np.array(w_c) - (np.linalg.lstsq(G, shiftv, rcond=None)[0] if args.base else 0.0)
        P = base.copy()
        for nm, wc, wb, Dp in zip(names, w_c, w_b, Ds):
            P += wb * Dp
            print(f"  candidate part {nm}: E_all {np.mean(Dp ** 2):.5f}, w_C {wc:+.3f} -> w_base {wb:+.3f}")
        P = np.clip(P, 0.0, None)
        shift.write_predictions(P, ROOT / args.candidate_out)
        print(f"  candidate {args.candidate_out}: E[(P - C)^2] {np.mean((P - C) ** 2):.5f}, E[(P - base)^2] {np.mean((P - base) ** 2):.5f}")
        if args.register:
            centers = json.loads((ROOT / "work/spat_centers.json").read_text())
            centers[Path(args.candidate_out).name] = args.base or args.centre
            (ROOT / "work/spat_centers.json").write_text(json.dumps(centers, indent=1))
        if args.manifest:
            with open(ROOT / args.manifest, "a") as fh:
                fh.write(f"{args.candidate_out}\t{args.candidate_msg or 'CANDIDATE: C + board-weighted ED parts ' + args.candidate}\n")


if __name__ == "__main__":
    main()
