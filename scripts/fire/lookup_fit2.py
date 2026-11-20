#!/usr/bin/env python
"""Fit the TEST world's fire rule on the test_y0 age profiles of MANY sites (no-fire lookups of the full pass).

State-space emulator of the lookup (500 years on the cycled MERRA-2 years) of one site, vectorised over parameter sets:
    A[t+1] = (A[t] + g(A[t])) * (1 - 0.7 (1 - exp(-b[t]))),
    b[t]   = min(cap_y, mean_m min(cap_m, A[t] * fp1 * X[k, m] * [P[k, m] < thr])),   k = t mod 40,
with g(A) the growth increment of the site's NO-FIRE lookup (ED) at the age where that curve passes A, and
X = (DI / 30000)^power from the rolling 12-slot array. DI variants:
    zero   the January index is always 0 (what the TRAINING truth shows; scripts/fire/emulator.py)
    wrap   the shipped code path: a dry January collects the dry months backwards (Dec, Nov, ...) from the array
Compared with the decoded test_y0 agb (x 1.055) at the seed ages >= 30.

    python scripts/fire/lookup_fit2.py --nofire work/edv3/full_cx16 work/edv3/sp_cxs_tc40b [--tmin 16] [--top 25]
"""
from __future__ import annotations

import argparse
import glob
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "fire")); sys.path.insert(0, str(ROOT / "scripts" / "ed"))
from cgb import io  # noqa: E402
import fire_proxy_calib as fp  # noqa: E402

AGES = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500])
F_AGB = 1.055


def di_avg(T: np.ndarray, P: np.ndarray, b: np.ndarray, wrap: bool, kdry: float = 1.0) -> np.ndarray:
    """T, P (..., 12) deg C and mm/month (the 12-slot array) -> ED's dryness_index_avg.
    kdry: a month counts as dry when precip < kdry x pet (shipped code: 1.0; small values = only (nearly) rainless months)."""
    Tc = np.maximum(T, 0.0001)
    I = np.sum((Tc / 5.0) ** 1.51, axis=-1, keepdims=True)
    a = 67.5e-8 * I ** 3 - 77.1e-6 * I ** 2 + 0.0179 * I + 0.492
    pet = 1.62 * b * (10.0 * Tc / I) ** a * 120.0
    pr = np.maximum(P * 12.0, 1e-9)
    dry = kdry * pet > pr
    inc = 30.0 * (pet - pr)
    di = np.zeros(T.shape)
    if wrap:                                     # January: its own deficit plus the consecutive dry months before it (Dec, Nov, ...)
        acc = np.where(dry[..., 0], inc[..., 0], 0.0); alive = dry[..., 0].copy()
        for m in range(11, 0, -1):
            alive = alive & dry[..., m]
            acc = acc + np.where(alive, inc[..., m], 0.0)
        di[..., 0] = acc
    for m in range(1, 12):
        di[..., m] = np.where(dry[..., m], di[..., m - 1] + inc[..., m], 0.0)
    return di.mean(axis=-1)


def xroll_stale(T: np.ndarray, P: np.ndarray, lat: float, cycles: int = 3) -> np.ndarray:
    """(years, 12) DI of each month, SHIPPED code path: rolling array, a dry January collects the dry months before it,
    a WET January keeps the index of the last dry January (dryness_index[0] is simply not assigned) and a dry February
    builds on it. The years are cycled `cycles` times so that the stale value is in equilibrium with the cycle."""
    b = fp.b_factors(float(lat)); ny = T.shape[0]
    Tp = np.concatenate([T[-1:], T], axis=0); Pp = np.concatenate([P[-1:], P], axis=0)
    out = np.zeros(T.shape); jan = 0.0
    for _ in range(cycles):
        for k in range(ny):
            for m in range(12):
                Ta = T[k].copy(); Pa = P[k].copy()
                Ta[m + 1:] = Tp[k, m + 1:]; Pa[m + 1:] = Pp[k, m + 1:]
                Tc = np.maximum(Ta, 0.0001); I = np.sum((Tc / 5.0) ** 1.51)
                a = 67.5e-8 * I ** 3 - 77.1e-6 * I ** 2 + 0.0179 * I + 0.492
                pet = 1.62 * b * (10.0 * Tc / I) ** a * 120.0; pr = np.maximum(Pa * 12.0, 1e-9)
                dry = pet > pr; inc = 30.0 * (pet - pr)
                di = np.zeros(12)
                if dry[0]:
                    acc = inc[0]
                    for mm in range(11, 0, -1):
                        if not dry[mm]:
                            break
                        acc += inc[mm]
                    jan = acc
                di[0] = jan                                  # wet January: the stale value stays in the array
                for mm in range(1, 12):
                    di[mm] = di[mm - 1] + inc[mm] if dry[mm] else 0.0
                out[k, m] = di.mean()
    return out


def xroll(T: np.ndarray, P: np.ndarray, lat: float, wrap: bool, kdry: float = 1.0) -> np.ndarray:
    """(years, 12) DI of each month with the rolling array (months > m still hold the previous year; cycle wraps)."""
    b = fp.b_factors(float(lat))
    Tp = np.concatenate([T[-1:], T], axis=0); Pp = np.concatenate([P[-1:], P], axis=0)
    out = np.zeros(T.shape)
    for m in range(12):
        Ta = T.copy(); Pa = P.copy()
        Ta[:, m + 1:] = Tp[:-1, m + 1:]; Pa[:, m + 1:] = Pp[:-1, m + 1:]
        out[:, m] = di_avg(Ta, Pa, b, wrap, kdry)
    return out


VARIANTS = {"zero": (False, 1.0), "wrap": (True, 1.0)}
# monthly climate "as delivered" (released = alpha x real + beta; work/cal/cal_maps_cal.npz): P only, or T and P; the rule then acts on the delivered P
AD = {"zero_pad": (False, False, True), "wrap_pad": (True, False, True), "zero_tpad": (False, True, True), "wrap_tpad": (True, True, True)}


def load_sites(roots: list[str], tmin: float) -> dict[int, dict]:
    ex = np.load(ROOT / "work/test_cell_exact.npz"); X = io.open_train_x()
    z = np.load(ROOT / "work/cal/cal_maps_cal.npz"); al, be = z["alpha"][:2].astype(float), z["beta"][:2].astype(float)
    out: dict[int, dict] = {}
    for r in roots:
        for f in sorted(glob.glob(str(ROOT / r / "csv" / "site*.csv"))):
            t = pd.read_csv(f); s = int(t.site.iloc[0])
            if s in out or "spin_agb_kgC" not in t.columns:
                continue
            c = int(ex["cell"][s])
            if c < 0:
                continue
            a = t[t.scenario == "ssp126"].set_index("age").reindex(AGES); b = t[t.scenario == "ssp585"].set_index("age").reindex(AGES)
            nf = a.spin_agb_kgC.to_numpy(dtype=float)
            if not np.isfinite(nf).all() or nf[7:].mean() < 1.0:
                continue
            T = np.asarray(X[c, :, :, 0], dtype=np.float64) - 273.15; P = np.asarray(X[c, :, :, 1], dtype=np.float64)
            if T.mean() < tmin:
                continue
            y0 = 0.5 * (a.y0_agb.to_numpy(dtype=float) + b.y0_agb.to_numpy(dtype=float)) * F_AGB
            lat = float(ex["lat"][s])
            di = {k: xroll(T, P, lat, *v) for k, v in VARIANTS.items()}; Pr = {k: P for k in VARIANTS}
            Tad = al[0] * (T + 273.15) + be[0] - 273.15; Pad = al[1] * P + be[1]
            for k, (wrap, tad, pad) in AD.items():
                di[k] = xroll(Tad if tad else T, Pad if pad else P, lat, wrap); Pr[k] = Pad if pad else P
            out[s] = dict(y0=y0, nf=nf, P=Pr, lat=lat, lon=float(ex["lon"][s]), di=di, root=r.split("/")[-1])
    return out


def simulate(d: dict, par: np.ndarray, variant: str, power: float) -> np.ndarray:
    """par (n, 4) = fp1, thr, cap_m, cap_y -> agb at the seed ages (n, 15)."""
    t = np.concatenate([[0.0], AGES.astype(float)]); a = np.concatenate([[0.0], np.maximum.accumulate(d["nf"])])
    tt = np.arange(0, 501); aa = np.interp(tt, t, a); inc = np.diff(aa, append=aa[-1])
    order = np.argsort(aa); xs, ys = aa[order], inc[order]
    Xf = (np.maximum(d["di"][variant], 1.0) / 30000.0) ** power                  # (40, 12)
    n = len(par); fp1 = par[:, 0][:, None]; cap_m = par[:, 2][:, None]; cap_y = par[:, 3]
    W = Xf[None] * (d["P"][variant][None] < par[:, 1][:, None, None])              # (n, 40, 12)
    A = np.zeros(n); out = np.zeros((n, 501))
    for step in range(500):
        k = step % 40
        A = A + np.interp(A, xs, ys)
        b = np.minimum(cap_y, np.minimum(cap_m, A[:, None] * fp1 * W[:, k]).mean(1))
        A = A * (1.0 - 0.7 * (1.0 - np.exp(-b)))
        out[:, step + 1] = A
    return out[:, AGES]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nofire", nargs="+", default=["work/edv3/full_cx16", "work/edv3/sp_cxs_tc40b"])
    ap.add_argument("--tmin", type=float, default=16.0)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--power", type=float, nargs="+", default=[10.0])
    ap.add_argument("--save", default="work/fire/lookup_fit2.csv")
    args = ap.parse_args()
    data = load_sites(args.nofire, args.tmin)
    names = sorted(data)
    m = AGES >= 30
    dev = {s: float(np.mean(np.log((data[s]["y0"][m] + 0.1) / (data[s]["nf"][m] + 0.1)))) for s in names}
    burned = [s for s in names if dev[s] < -0.25]; mid = [s for s in names if -0.25 <= dev[s] < -0.08]; clean = [s for s in names if abs(dev[s]) < 0.08]
    print(f"{len(names)} sites; burned (y0 / no-fire < 0.78): {len(burned)}; in between: {len(mid)}; unburned: {len(clean)}; above: {len(names) - len(burned) - len(mid) - len(clean)}")
    grid = np.array(list(itertools.product(10.0 ** np.arange(-8.0, -1.9, 0.25), (2.3, 4.0, 8.33, 10.0, 1e9), (1e9, 0.25), (0.2, 1e9))))
    rows = []
    for power in args.power:
        for variant in list(VARIANTS) + list(AD):
            L = np.zeros((len(names), len(grid)))
            for i, s in enumerate(names):
                sim = simulate(data[s], grid, variant, power)
                L[i] = np.sqrt(np.mean(np.log((sim[:, m] + 0.1) / (data[s]["y0"][m] + 0.1)) ** 2, axis=1))
            ib = [names.index(s) for s in burned]; im = [names.index(s) for s in mid]; ic = [names.index(s) for s in clean]
            for j, g in enumerate(grid):
                rows.append((power, variant, *g, float(np.sqrt(np.mean(L[ib, j] ** 2))), float(np.sqrt(np.mean(L[im, j] ** 2))), float(np.sqrt(np.mean(L[ic, j] ** 2))),
                             float(np.mean(L[ic, j] > 0.15)), float(np.sqrt(np.mean(L[:, j] ** 2)))))
    r = pd.DataFrame(rows, columns=["power", "variant", "fp1", "thr", "cap_m", "cap_y", "rms_burned", "rms_mid", "rms_clean", "clean_gt15", "rms_all"])
    r["score"] = np.sqrt((r.rms_burned ** 2 + r.rms_mid ** 2 + r.rms_clean ** 2) / 3.0)        # the three groups weigh the same
    pd.set_option("display.width", 250)
    base = float(np.sqrt(np.mean([np.mean(np.log((data[s]["nf"][m] + 0.1) / (data[s]["y0"][m] + 0.1)) ** 2) for s in burned])))
    print(f"no fire: rms burned {base:.3f}")
    r["lfp1"] = np.log10(r.fp1)
    show = ["power", "variant", "lfp1", "thr", "cap_m", "cap_y", "rms_burned", "rms_mid", "rms_clean", "clean_gt15", "score"]
    print(r.sort_values("score").head(args.top)[show].round(3).to_string(index=False))
    r.to_csv(ROOT / args.save, index=False)
    # per-site table for the best parameter set of each variant
    best = r.sort_values("score").groupby("variant", as_index=False).first().sort_values("score")
    print("best per DI variant:"); print(best.round(5).to_string(index=False))
    for variant in list(best.variant[:3]):
        q = r[r.variant == variant].sort_values("score").iloc[0]
        par = np.array([[q.fp1, q.thr, q.cap_m, q.cap_y]])
        print(f"-- {variant}: fp1 {q.fp1:.1e} thr {q.thr} cap_m {q.cap_m:g} cap_y {q.cap_y:g} power {q.power}: burned and in-between sites (log(y0 / no-fire) vs emulated)")
        for s in burned + mid:
            sim = simulate(data[s], par, variant, float(q.power))[0]
            print(f"   {s} lat {data[s]['lat']:6.2f} lon {data[s]['lon']:7.2f}: y0 dev {dev[s]:+.2f}  emulated {np.mean(np.log((sim[m] + 0.1) / (data[s]['nf'][m] + 0.1))):+.2f}   DI med {np.median(data[s]['di'][variant].mean(1)) / 1e3:5.1f}k")
        bad = [(s, float(np.mean(np.log((simulate(data[s], par, variant, float(q.power))[0][m] + 0.1) / (data[s]["nf"][m] + 0.1))))) for s in clean]
        bad = sorted(bad, key=lambda x: x[1])[:10]
        print("   unburned sites the rule burns most: " + ", ".join(f"{s} ({v:+.2f}; lat {data[s]['lat']:.1f})" for s, v in bad))


if __name__ == "__main__":
    main()
