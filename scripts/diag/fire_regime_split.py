#!/usr/bin/env python
"""Does the TEST world's own fire behaviour (seen in the test lookups y0) tell which sites burn in the test WINDOWS?

The lookup of every test site is a 500-year spin-up of the competition's ED configuration on the twin cell's 40 MERRA-2 years (cycled); whether the
lookup is burned (y0 far below the no-fire lookup) is a direct observation of that configuration's fire regime under a KNOWN climate. Step 1 fits
"burned lookup" on climate features of the MERRA years; step 2 applies the fit to the window climate of every site x scenario (release transform
undone) and compares the predicted regime with what the ORNL product shows at the same node (strict old-growth effect q_ornl).

    python scripts/diag/fire_regime_split.py [--out work/fire/regime_split.csv]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "fire")); sys.path.insert(0, str(ROOT / "scripts" / "ed"))
from cgb import io  # noqa: E402
import emulator as em  # noqa: E402
import fire_x as fx  # noqa: E402

FEATS = ["nr1", "nr2", "nr8", "run2", "Pann", "Tm", "ppet", "ldi", "ldi_max"]


def features(T: np.ndarray, P: np.ndarray, lat: float) -> dict:
    """Climate features of a (years, 12) monthly series: T in deg C, P in mm per month (real units)."""
    xr = em.xroll(T, P, lat, T[-1], P[-1])                       # (years, 12) monthly factors (DI / 30000)^10
    di = 30000.0 * np.maximum(xr, 1e-30) ** 0.1
    r2 = P < 2.3
    run = []
    for y in range(P.shape[0]):                                  # longest run of months below 2.3 mm in the year (wrapping)
        v = np.concatenate([r2[y], r2[y]]); best = cur = 0
        for b in v:
            cur = cur + 1 if b else 0; best = max(best, cur)
        run.append(min(best, 12))
    # Thornthwaite PET (mm per year) from the mean monthly temperatures, as the model's own formula does
    Tm = np.clip(T.mean(0), 0.0001, None); I = float(np.sum((Tm / 5.0) ** 1.51)); a = 67.5e-8 * I ** 3 - 77.1e-6 * I ** 2 + 0.0179 * I + 0.492
    pet = float(np.sum(16.2 * (10.0 * Tm / I) ** a))
    return dict(nr1=float((P < 1.0).sum(1).mean()), nr2=float(r2.sum(1).mean()), nr8=float((P < 8.33).sum(1).mean()), run2=float(np.mean(run)),
                Pann=float(P.sum(1).mean()), Tm=float(T.mean()), ppet=float(P.sum(1).mean() / max(pet, 1.0)), ldi=float(np.log10(np.median(di.mean(1)) + 1.0)),
                ldi_max=float(np.log10(di.mean(1).max() + 1.0)))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="work/fire/regime_split.csv"); args = ap.parse_args()
    d = pd.read_csv(ROOT / "work/ff/landscape_factor.csv")
    ex = np.load(ROOT / "work/test_cell_exact.npz"); z = np.load(ROOT / "work/cal/cal_maps_cal.npz"); al, be = z["alpha"][:2].astype(float), z["beta"][:2].astype(float)
    X = {sc: io.open_test(sc)[0] for sc in ("ssp126", "ssp585")}
    sites = np.sort(d.site.unique()); rows_m = {}; rows_w = []
    for s in sites:
        cell = int(ex["cell"][s]); lat = float(ex["lat"][s]) if np.isfinite(ex["lat"][s]) else 0.0
        if cell >= 0:
            T, P = fx._train_tp(cell); rows_m[s] = features(np.asarray(T, float), np.asarray(P, float), lat)
        for sc in ("ssp126", "ssp585"):
            x = np.asarray(X[sc][int(s), :, :, :2], dtype=np.float64)
            T = (x[:, :, 0] - be[0]) / al[0] - 273.15; P = np.clip((x[:, :, 1] - be[1]) / al[1], 0.0, None)
            f = features(T, P, lat); f2 = features(T[20:], P[20:], lat)
            rows_w.append(dict(site=s, sc=sc, **{k + "_w": v for k, v in f.items()}, **{k + "_w2": v for k, v in f2.items()}))
    M = pd.DataFrame.from_dict(rows_m, orient="index").add_suffix("_m").rename_axis("site").reset_index()
    W = pd.DataFrame(rows_w)
    t = d.merge(M, on="site", how="left").merge(W, on=["site", "sc"], how="left")
    t.to_csv(ROOT / args.out, index=False)
    print(f"wrote {args.out}: {len(t)} site x scenarios, {t.nr2_m.notna().sum()} with a twin cell")
    # 1. the lookup's regime against the MERRA climate (one row per site: ssp126 row; the lookup is common to both scenarios)
    L = t[(t.sc == "ssp126") & t.nr2_m.notna() & (t.Tm_m >= 14)].copy()
    lookup0 = L.y0_old / np.exp(L.seed)                                                 # no-fire lookup level implied by the seed mismatch
    L = L[lookup0 > 0.25]
    burned = (L.seed < -0.25).to_numpy(); clean = (L.seed.abs() < 0.07).to_numpy(); use = burned | clean
    print(f"warm sites with a forest lookup: {len(L)}; burned lookup (seed < -0.25): {int(burned.sum())}; clean (|seed| < 0.07): {int(clean.sum())}")
    from scipy.stats import rankdata
    def auc(score, y):
        r = rankdata(score); n1 = y.sum(); n0 = (~y).sum(); return float((r[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))
    for k in FEATS:
        v = L[k + "_m"].to_numpy()[use]; y = burned[use]; a_ = auc(v, y)
        print(f"   {k:8s} AUC {max(a_, 1 - a_):.3f} ({'higher' if a_ >= 0.5 else 'lower'} = burned)   burned median {np.median(v[y]):8.2f} [{np.percentile(v[y], 10):7.2f} .. {np.percentile(v[y], 90):7.2f}]"
              f"   clean median {np.median(v[~y]):8.2f} [{np.percentile(v[~y], 10):7.2f} .. {np.percentile(v[~y], 90):7.2f}]")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cols = ["nr1", "nr2", "run2", "ppet", "ldi", "Tm", "Pann"]
    Xl = L[[c + "_m" for c in cols]].to_numpy()[use]; yl = burned[use]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    pcv = cross_val_predict(clf, Xl, yl, cv=5, method="predict_proba")[:, 1]
    print(f"logistic fit on {cols}: 5-fold AUC {auc(pcv, yl):.3f}; burned sites with p < 0.5: {int((pcv[yl] < 0.5).sum())} of {int(yl.sum())}; clean sites with p >= 0.5: {int((pcv[~yl] >= 0.5).sum())} of {int((~yl).sum())}")
    clf.fit(Xl, yl)
    # 2. the same fit on the window climates
    for suf in ("_w", "_w2"):
        ok = t[[c + suf for c in cols]].notna().all(1).to_numpy(); p = np.full(len(t), np.nan)
        p[ok] = clf.predict_proba(t.loc[ok, [c + suf for c in cols]].to_numpy())[:, 1]; t["p" + suf] = p
    okm = t[[c + "_m" for c in cols]].notna().all(1).to_numpy(); p = np.full(len(t), np.nan)
    p[okm] = clf.predict_proba(t.loc[okm, [c + "_m" for c in cols]].to_numpy())[:, 1]; t["p_m"] = p
    t.to_csv(ROOT / args.out, index=False)
    print("predicted burned-regime probability under MERRA (p_m), the 40 window years (p_w) and the last 20 window years (p_w2):")
    for cls in ("never", "b", "pos", "cluster"):
        g = t[t.cls == cls]
        print(f"   {cls:8s} n {len(g):5d}  mean p_m {g.p_m.mean():.3f}  p_w {g.p_w.mean():.3f}  p_w2 {g.p_w2.mean():.3f}  share p_w2 > 0.5: {np.mean(g.p_w2 > 0.5):.3f}  share with a burned lookup (seed < -0.25): {np.mean(g.seed < -0.25):.3f}")
    # 3. against ORNL's strict effect at fire-possible sites
    g = t[(t.cls == "pos") & (t.qs == 1) & np.isfinite(t.q_ornl)]
    print(f"fire-possible site x scenarios with a strict ORNL effect: {len(g)}")
    for nm, m in (("ORNL burning (q < -0.3)", g.q_ornl < -0.3), ("ORNL mid (-0.3 .. -0.05)", (g.q_ornl >= -0.3) & (g.q_ornl < -0.05)), ("ORNL quiet (q > -0.05)", g.q_ornl >= -0.05)):
        h = g[m]
        print(f"   {nm:26s} n {len(h):4d}  p_m median {h.p_m.median():.3f}  p_w2 median {h.p_w2.median():.3f}  p_w2 > 0.5: {np.mean(h.p_w2 > 0.5):.2f}  p_w2 < 0.1: {np.mean(h.p_w2 < 0.1):.2f}"
              f"  nr2 MERRA {h.nr2_m.median():.2f} -> window {h.nr2_w2.median():.2f}  burned lookup {np.mean(h.seed < -0.25):.2f}")
    print(f"   AUC of p_w2 for ORNL burning vs quiet: {auc(g.p_w2[(g.q_ornl < -0.3) | (g.q_ornl >= -0.05)].to_numpy(), (g.q_ornl < -0.3)[(g.q_ornl < -0.3) | (g.q_ornl >= -0.05)].to_numpy()):.3f}")


if __name__ == "__main__":
    main()
