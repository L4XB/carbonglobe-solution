#!/usr/bin/env python
"""Forced-fire ED: per site the LOOKUP fire strength is tuned on the y0 profile and the WINDOW fire strength on a target for the old rows' agb effect.

Why (notes 11-11 04:45 UTC): given the fire, ED reproduces the organisers' heights and the agb of every seed age (training cells with truth-derived
ignition factors: height response corr 0.90, rms 0.05-0.085 against 0.15-0.32 for the chronosequence). In the test world the fire itself is not
reproducible from the drivers (per-site effective ignitions span three decades), but its OUTCOME is known for the old rows (amplified ORNL effect =
file K) and its imprint on the lookup is the y0 profile. So per site:

  1. lookup: spin-up only, fire strength L = fp1 x fire_spin_scale searched upwards from the main pass's 3e-8 until the old rows of the lookup come down
     to y0 (median log ratio of agb, ages >= 150), then regula falsi in log L. A lookup that already matches keeps 3e-8.
  2. windows: per scenario the factor c on the window's ignition series is searched until ED's old-row agb effect (median over ages >= 200 of
     log(year-40 December agb / own first-record profile at age + 40)) meets the target. Tuning runs hold the old ages only.
  3. final run with all 15 seed ages -> <root>/csv/site*.csv (the usual table) and <root>/tune/site*.json (strengths, trial history, targets).

Test sites: config cx17 (fire series merm:rule8.33 in the lookup, fut:own:rule8.33:monthly in the windows), target from work/ff/targets_K.npz
(make it with --make-targets results/<file>.csv). Training cells (--train; validation against the truth): config trf8 with the test pipeline's lookup
and the same fire structure on the cell's own MERRA-2 years; the target is the truth's own old-row effect, and both "scenarios" (identical inputs)
are used as two probes of the window strength per run.

    ED_EDLU=$PWD/work/edv3/build_fire6/edlu ED_FIRE_X_YEARCAP=1 ED_FIRE_X_SHIFT=1 python scripts/ed/forced_fire.py --root work/edv3/ff1 --sites 4174,5319 --workers 2
    ... --train --root work/edv3/ff_train1 --sites <training cells>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ed")); sys.path.insert(0, str(ROOT / "src"))
import make_site_inputs as msi  # noqa: E402
import run_site as rs  # noqa: E402

AGES = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]); LA = np.log(AGES)
OLD = [200, 250, 300, 350, 400, 450, 500]
B0 = 3e-8                 # ignition strength (fp1) of the main pass
TOL_L, TOL_W = 0.08, 0.08
TRAIN_OVER = dict(co2_spinup_fixed=415.0, spinup_start=0, spinup_years=40, spinup_cycle=True, spinup_both=False, fire_on=True, fire_spin=True, fp1=B0,
                  params_over={"fire_max_disturbance_rate": 0.2}, fire_fuel_fix=True)


def trial(site: int, troot: Path, config: str, over: dict, train: bool) -> pd.DataFrame:
    if train:
        import train_site as ts
        msi.io.open_test = ts._fake_open_test
        over = {**TRAIN_OVER, **over, "co2": "asis", "repair": False}
    shutil.rmtree(troot, ignore_errors=True)
    rs.run_site(site, troot, rs.config_args(config, **over), keep=False)
    df = pd.read_csv(troot / "csv" / f"site{site:05d}.csv")
    shutil.rmtree(troot, ignore_errors=True)
    return df


def lookup_ratio(df: pd.DataFrame, ua: float) -> tuple[float, float, float, float]:
    """Median log(lookup / y0) over the old ages for agb and height, and the rms over ages >= 20."""
    g = df[df.scenario == df.scenario.iloc[0]].drop_duplicates("age").set_index("age").reindex(AGES)
    fa = np.log((np.maximum(g.spin_agb_kgC.to_numpy(float), 0) + 0.05) / (g.y0_agb.to_numpy(float) * ua + 0.05))
    fh = np.log((np.maximum(g.spin_height_m.to_numpy(float), 0) + 0.3) / (g.y0_height.to_numpy(float) + 0.3))
    old = AGES >= 150; mid = AGES >= 20
    return float(np.median(fa[old])), float(np.median(fh[old])), float(np.sqrt(np.mean(fa[mid] ** 2))), float(np.sqrt(np.mean(fh[mid] ** 2)))


def old_effect(df: pd.DataFrame, sc: str) -> tuple[float, float]:
    """ED's old-row effect (agb, height) against its own first-record profile at age + 40."""
    g = df[df.scenario == sc].drop_duplicates("age").sort_values("age")
    ages = g.age.to_numpy(float); x = np.log(np.minimum(ages + 40, 500)); la = np.log(ages)
    ea = np.exp(np.interp(x, la, np.log(np.maximum(g.pre_agb_m12.to_numpy(float), 0) + 0.01))); eh = np.exp(np.interp(x, la, np.log(np.maximum(g.pre_h_m12.to_numpy(float), 0) + 0.1)))
    qa = np.log((np.maximum(g.end_agb_m12.to_numpy(float), 0) + 0.01) / ea); qh = np.log((np.maximum(g.end_h_m12.to_numpy(float), 0) + 0.1) / eh)
    k = ages >= 200
    return float(np.median(qa[k])), float(np.median(qh[k]))


def train_target(cell: int) -> float:
    import train_site as ts
    y0 = ts.train_y0_raw(cell)[:, 1]; y40 = ts.train_y40_raw(cell)[:, 1]
    prof = np.exp(np.interp(np.log(np.minimum(AGES + 40, 500)), LA, np.log(y0 + 0.01)))
    return float(np.median(np.log((y40 + 0.01) / prof)[AGES >= 200]))


def next_c(hist: list[tuple[float, float]], target: float) -> float | None:
    """Next window factor from the (c, effect) pairs so far; None = converged or nothing left to try. The effect falls with c."""
    above = [(c, q) for c, q in hist if q > target]          # too little fire
    below = [(c, q) for c, q in hist if q <= target]         # enough or too much fire
    best = min(hist, key=lambda t: abs(t[1] - target))
    if abs(best[1] - target) < TOL_W:
        return None
    # no bracket yet: jump along a typical dose-response (the old-row effect falls by ~0.4 per decade of ignition between c = 1 and 1000 at the first
    # test sites, less beyond), at least a factor 2 and at most 100 per step
    step = lambda q: float(np.clip(10.0 ** (abs(q - target) / 0.4), 2.0, 100.0))
    if not below:                                            # more fire needed
        c, q = max(above, key=lambda t: t[0])
        return None if c >= 3e5 else (10.0 if c == 0 else c * step(q))
    if not above:                                            # less fire needed
        c, q = min(below, key=lambda t: t[0])
        if c == 0:
            return None                                      # even without window fire ED is below the target
        return 0.0 if c <= 1e-3 else c / step(q)
    (c1, q1), (c2, q2) = max(above, key=lambda t: t[0]), min(below, key=lambda t: t[0])
    if c1 == 0:
        c1 = c2 / 100.0
        if c2 <= 1e-3:
            return None
    if c2 / c1 < 1.3:
        return None
    t = float(np.clip((q1 - target) / max(q1 - q2, 1e-6), 0.2, 0.8))
    return float(np.exp(np.log(c1) + t * (np.log(c2) - np.log(c1))))


def one_site(site: int, root: str, train: bool, targets: dict | None, prior: float | None = None, prior_q: dict | None = None, max_spin: int = 8, max_win: int = 4) -> tuple[int, str, float]:
    t0 = time.time(); rootp = Path(root); scratch = rootp / "trial" / f"s{site}"
    config = "trf8" if train else "cx17"; ua = 1.0 if train else 1.055
    fsx = dict(fire_spin_x="merm:rule8.33", fire_x="trm:rule8.33") if train else {}
    try:
        tgt = {"ssp126": train_target(site), "ssp585": train_target(site)} if train else targets
        tshift = float(os.environ.get("FF_TARGET_SHIFT", "0"))      # training validation of the sensitivity to a wrong old-row target
        if tshift:
            tgt = {k: v + tshift for k, v in tgt.items()}
        log: dict = {"site": site, "train": train, "target": tgt, "lookup": [], "windows": []}
        # ---- 1. lookup strength
        def f(L: float) -> float:
            df = trial(site, scratch, config, {**fsx, "spin_only": True, "fp1": B0, "fire_spin_scale": L / B0}, train)
            fa, fh, ra, rh = lookup_ratio(df, ua); log["lookup"].append(dict(L=L, fa=fa, fh=fh, rms_a=ra, rms_h=rh))
            return fa
        reused = None
        for r in [x for x in os.environ.get("FF_REUSE_TUNE", "").split(",") if x]:      # lookup strength of an earlier forced run of this site
            jf = ROOT / r / "tune" / f"site{site:05d}.json"
            if jf.exists():
                reused = json.load(open(jf)); break
        if reused is not None:
            lo, flo = float(reused["L"]), float(reused["lookup_fa"]); log["lookup"] = list(reused["lookup"]) + [dict(L=lo, fa=flo, source="reused")]
        elif prior is not None and np.isfinite(prior):        # test sites: the main pass ran exactly this lookup (fp1 3e-8, scale 1)
            lo, flo = B0, float(prior); log["lookup"].append(dict(L=B0, fa=float(prior), fh=float("nan"), rms_a=float("nan"), rms_h=float("nan"), source="main pass"))
        else:
            lo, flo = B0, f(B0)
        hi = fhi = None
        if reused is not None:
            flo_search = 0.0                                  # no search: the reused strength stands
        else:
            flo_search = flo
        if flo_search < -0.12:                               # the weak-fire lookup is already below y0: does it help to switch the lookup fire off?
            f0 = f(0.0)
            if abs(f0) < abs(flo):
                lo, flo = 0.0, f0
        if flo_search > 0.12:
            for L in (B0 * 30, B0 * 1e3, B0 * 3e4, B0 * 1e6):
                fv = f(L)
                if fv <= 0:
                    hi, fhi = L, fv; break
                lo, flo = L, fv
            while hi is not None and min(abs(flo), abs(fhi)) > TOL_L and len(log["lookup"]) < max_spin and hi / lo > 1.5:
                t = float(np.clip(flo / max(flo - fhi, 1e-6), 0.2, 0.8)); L = float(np.exp(np.log(lo) + t * (np.log(hi) - np.log(lo))))
                fv = f(L)
                if fv > 0: lo, flo = L, fv
                else: hi, fhi = L, fv
        Lbest = lo if (hi is None or abs(flo) <= abs(fhi)) else hi
        log["L"] = Lbest; log["lookup_fa"] = flo if Lbest == lo else fhi
        # ---- 2. window strength per scenario (training: the two identical scenarios probe two strengths per run)
        hist = {"ssp126": [], "ssp585": []}
        c0 = float(np.clip(Lbest / B0, 1.0, 1e4))             # start from the lookup's own fire regime
        c = {"ssp126": c0 * (10.0 if train else 1.0), "ssp585": c0 * (1000.0 if train else 1.0)}
        if (not train) and prior_q is not None and Lbest == B0 and all(np.isfinite(v) for v in prior_q.values()):
            # the main pass IS the run at c = 1 with this lookup: take its old-row effect as the first point and start the search from there
            for sc in ("ssp126", "ssp585"):
                hist[sc].append((1.0, float(prior_q[sc]))); log["windows"].append(dict(it=-1, sc=sc, c=1.0, qa=float(prior_q[sc]), qh=float("nan"), source="main pass"))
            nxt = {sc: next_c(hist[sc], tgt[sc]) for sc in ("ssp126", "ssp585")}
            c = {sc: (nxt[sc] if nxt[sc] is not None else 1.0) for sc in ("ssp126", "ssp585")}
            if nxt["ssp126"] is None and nxt["ssp585"] is None:
                max_win = 0                                   # both targets are met by the main pass: only the final run
        base_over = {**fsx, "fp1": B0, "fire_spin_scale": Lbest / B0}
        final = None
        regime = [float(x) for x in os.environ.get("FF_REGIME", "").split(",") if x]
        if regime and not train:
            # REGIME run: the windows keep the lookup's own fire strength, times a scenario factor (rho126, rho585) calibrated on the burned-lookup sites
            # with a strict ORNL effect (scripts/diag/ff_regime_rho.py). No search: one full run. The old-row effect is recorded after the run.
            hist = {"ssp126": [], "ssp585": []}; max_win = 0
            creg = float(np.clip(Lbest / B0, 1.0, 1e6))
            log["regime"] = {"rho": regime, "c0": creg}
        for it in range(max_win):
            df = trial(site, scratch, config, {**base_over, "fire_x_scale_w126": c["ssp126"], "fire_x_scale_w585": c["ssp585"], "ages_only": OLD}, train)
            for sc in ("ssp126", "ssp585"):
                qa, qh = old_effect(df, sc); hist[sc].append((c[sc], qa)); log["windows"].append(dict(it=it, sc=sc, c=c[sc], qa=qa, qh=qh))
            if train:      # one pooled history; next two probes around the target
                pooled = hist["ssp126"] + hist["ssp585"]; hist["ssp126"] = hist["ssp585"] = pooled
                n1 = next_c(pooled, tgt["ssp126"])
                if n1 is None:
                    break
                more = all(q > tgt["ssp126"] for _, q in pooled); less = all(q <= tgt["ssp126"] for _, q in pooled)
                c = {"ssp126": n1, "ssp585": n1 * 10.0 if more else (n1 / 10.0 if less else n1 * 1.6)}
                if n1 == 0:
                    c["ssp585"] = 1e-3
            else:
                nxt = {sc: next_c(hist[sc], tgt[sc]) for sc in ("ssp126", "ssp585")}
                if nxt["ssp126"] is None and nxt["ssp585"] is None:
                    break
                c = {sc: (nxt[sc] if nxt[sc] is not None else min(hist[sc], key=lambda t, s=sc: abs(t[1] - tgt[s]))[0]) for sc in ("ssp126", "ssp585")}
        if regime and not train:
            cbest = {"ssp126": creg * regime[0], "ssp585": creg * regime[1]}
            log["c"] = cbest; log["miss"] = {"ssp126": float("nan"), "ssp585": float("nan")}
        else:
            cbest = {sc: min(hist[sc], key=lambda t, s=sc: abs(t[1] - tgt[s]))[0] for sc in ("ssp126", "ssp585")}
            log["c"] = cbest; log["miss"] = {sc: min(abs(q - tgt[sc]) for _, q in hist[sc]) for sc in ("ssp126", "ssp585")}
        # ---- 3. final run, all seed ages
        if train:
            import train_site as ts
            msi.io.open_test = ts._fake_open_test
            over = {**TRAIN_OVER, **base_over, "fire_x_scale_w126": cbest["ssp126"], "fire_x_scale_w585": cbest["ssp585"], "co2": "asis", "repair": False}
        else:
            over = {**base_over, "fire_x_scale_w126": cbest["ssp126"], "fire_x_scale_w585": cbest["ssp585"]}
        rs.run_site(site, rootp, rs.config_args(config, **over), keep=False)
        if train:
            import csv as _csv
            import train_site as ts
            from cgb import io
            y0, y40 = ts.train_y0_raw(site), ts.train_y40_raw(site); (rootp / "truth").mkdir(exist_ok=True)
            with open(rootp / "truth" / f"{site}.csv", "w", newline="") as fh:
                w = _csv.writer(fh); w.writerow(["age"] + [f"y0_{n}" for n in io.STATE_NAMES] + [f"y40_{n}" for n in io.STATE_NAMES])
                for i, a in enumerate(AGES):
                    w.writerow([int(a)] + [f"{v:.6g}" for v in y0[i]] + [f"{v:.6g}" for v in y40[i]])
        if regime and not train:                              # old-row effect of the regime run, from its own table
            dfin = pd.read_csv(rootp / "csv" / f"site{site:05d}.csv")
            for sc in ("ssp126", "ssp585"):
                qa, qh = old_effect(dfin, sc); log["windows"].append(dict(it=0, sc=sc, c=cbest[sc], qa=qa, qh=qh, source="regime run")); log["miss"][sc] = abs(qa - tgt[sc])
        (rootp / "tune").mkdir(exist_ok=True)
        log["seconds"] = time.time() - t0
        json.dump(log, open(rootp / "tune" / f"site{site:05d}.json", "w"), indent=1)
        return site, f"ok L {Lbest:.2e} (lookup ratio {log['lookup_fa']:+.2f}, {len(log['lookup'])} spin-ups) c {cbest['ssp126']:.3g} / {cbest['ssp585']:.3g} miss {log['miss']['ssp126']:.2f} / {log['miss']['ssp585']:.2f}", time.time() - t0
    except Exception as exc:  # noqa: BLE001
        return site, f"FAIL {exc!r}", time.time() - t0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def make_targets(csv_file: str) -> None:
    from cgb import shift
    FL_A = 0.401412 / 3.39567
    keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy(); sc = (keys.scenario.to_numpy() == "ssp585").astype(int)
    P = shift.read_predictions(ROOT / csv_file).astype(float); look = np.load(ROOT / "work/lookup_agb40_kgC.npy").astype(float)
    q = np.log(np.maximum((P[:, 1] + FL_A) * 1.204 * 3.39567, 0.0) + 0.01) - np.log(look + 0.01)
    out = np.full((int(site.max()) + 1, 2), np.nan)
    d = pd.DataFrame(dict(site=site, sc=sc, q=q))[age >= 200].groupby(["site", "sc"]).q.median()
    for (s, j), v in d.items():
        out[int(s), int(j)] = v
    (ROOT / "work/ff").mkdir(exist_ok=True)
    np.savez(ROOT / "work/ff/targets_K.npz", q=out, source=csv_file)
    print(f"targets from {csv_file}: {int(np.isfinite(out).all(axis=1).sum())} sites; median {np.nanmedian(out):+.3f}, p5 {np.nanpercentile(out, 5):+.3f}, p95 {np.nanpercentile(out, 95):+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="work/edv3/ff1")
    ap.add_argument("--sites", default="", help="comma list, or a file with a comma / whitespace list")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--make-targets", default="")
    args = ap.parse_args()
    if args.make_targets:
        make_targets(args.make_targets); return
    txt = Path(args.sites).read_text() if Path(args.sites).exists() else args.sites
    sites = [int(x) for x in txt.replace(",", " ").split()]
    root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    noidx = ROOT / "work" / (root.name + ".noindex")          # ED scratch outside Spotlight's reach
    for d in ("run", "inputs", "trial"):
        (noidx / d).mkdir(parents=True, exist_ok=True)
        if not (root / d).exists():
            (root / d).symlink_to(noidx / d)
    done = {int(p.stem[4:]) for p in (root / "tune").glob("site*.json")} if (root / "tune").exists() else set()
    sites = [s for s in sites if s not in done]
    tq = None if args.train else np.load(ROOT / "work/ff/targets_K.npz")["q"]
    pr = None if args.train or not (ROOT / "work/ff/prior_lookup_ratio.npy").exists() else np.load(ROOT / "work/ff/prior_lookup_ratio.npy")[:, 0]
    pq = None if args.train or not (ROOT / "work/ff/main_pass_old_effect.npy").exists() else np.load(ROOT / "work/ff/main_pass_old_effect.npy")
    print(f"{len(sites)} sites to do ({len(done)} done), workers {args.workers}", flush=True)
    t0 = time.time(); n = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_site, s, str(root), args.train, None if args.train else {"ssp126": float(tq[s, 0]), "ssp585": float(tq[s, 1])},
                          None if pr is None else float(pr[s]), None if pq is None else {"ssp126": float(pq[s, 0]), "ssp585": float(pq[s, 1])}): s for s in sites}
        for fu in as_completed(futs):
            s, msg, dt = fu.result(); n += 1
            print(f"[{n}/{len(sites)} {(time.time() - t0) / 60:.1f} min] site {s}: {msg} ({dt:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
