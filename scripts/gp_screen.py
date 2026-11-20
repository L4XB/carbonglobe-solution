#!/usr/bin/env python
"""GP screen of probe candidates: per-block x scale-variable x covariate prior fitted by type-II ML on every scored readout
(scripts/gp_correction.py features). Prints LOO / leave-family-out diagnostics, then the predicted lambda +- sd of each
candidate file (default: every unscored probe in the day 40-51 batches).

    python scripts/gp_screen.py scale [spat_dir_x.csv ...]
"""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import gp_correction as gp
import os
gp.EXTRA_COV = bool(os.environ.get('GP_EXTRA_COV'))
from scipy.optimize import minimize
from cgb import shift
from local_eval import readouts

B, rho, Dm, y, fac, sc = readouts()
names = sorted(n for n in sc if (n.startswith("spat_") or n.startswith("perp_spat_")) and not n.endswith("_mirror.csv"))
K = len(y); N = B.size
keys = shift.template_keys(); site = keys["site"].to_numpy(); age = keys["age"].to_numpy()
sig = gp.probe_noise(sc, names, Dm, rho, fac, site); rk = rho * fac; G = Dm.T @ Dm / N
rd = lambda p: shift.read_predictions(p).astype(np.float64)
R = rd(ROOT / "results/latent_ref_y0_m114114.csv"); CD = rd(ROOT / "results/chronodyn1g58_g58.csv")
A46 = 0.876 * np.load(ROOT / "work/ageshift46_dec.npy")
svars = {"R": R, "dev": B - R, "cd": CD - B, "a46": A46 - B}
Zc, fam = gp.covariates(keys, R); fam = np.array(fam)
cl = (np.load(ROOT / "work/collapse_cluster_labels.npy") == 0)[site]; band = shift.band_of(age)
blocks = [(b, c, i) for b in ("young", "mid", "old") for c in (0, 1) for i in (False, True)]
mode = sys.argv[1] if len(sys.argv) > 1 else "scale_family"
A_parts, gid, meta, gnames = [], [], [], []
def grp(s, f):
    g = {"scale_family": f"{s}:{f}", "family": f, "scale": s, "cd_only": f"{s}:{f}" if s == "cd" else "other"}[mode]
    if g not in gnames: gnames.append(g)
    return gnames.index(g)
for b, c, inside in blocks:
    rows = np.where((band == b) & (cl == inside))[0]
    cells = rows * 2 + c; Zb = Zc[rows]
    for s in gp.SCALES:
        v = svars[s][rows, c]; norm = np.sqrt(np.mean((v[:, None] * Zb) ** 2, axis=0)) + 1e-12
        W = Dm[cells] * v[:, None]; A_parts.append(rk[:, None] * (W.T @ Zb) / N / norm[None, :])
        for j in range(Zb.shape[1]):
            gid.append(grp(s, fam[j])); meta.append((b, c, inside, s, fam[j], j, norm[j]))
A = np.concatenate(A_parts, 1); gid = np.array(gid); ng = len(gnames)
RGR = rk[:, None] * G * rk[None, :]; S = np.diag(sig ** 2)
AAg = np.stack([A[:, gid == g] @ A[:, gid == g].T for g in range(ng)])
def cov(th, idx=None):
    C = np.exp(th[0]) * RGR + S + np.tensordot(np.exp(th[1:]), AAg, axes=1)
    return C if idx is None else C[np.ix_(idx, idx)]
def fit(idx):
    def nll(th):
        C = cov(th, idx)
        try: L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError: return 1e10
        a = np.linalg.solve(L, y[idx]); return float(0.5 * a @ a + np.sum(np.log(np.diag(L))))
    th0 = np.r_[np.log(4e-4), np.full(ng, np.log(1e-4))]; best = None
    for st in (0.0, -3.0, 3.0):
        r = minimize(nll, th0 + np.r_[0, np.full(ng, st)], method="L-BFGS-B", bounds=[(-25, 5)] * len(th0))
        if best is None or r.fun < best.fun: best = r
    return best.x
allidx = np.arange(K)
th = fit(allidx); C = cov(th); Ci = np.linalg.inv(C); w = Ci @ y
loo = y - w / np.diag(Ci); e2 = rk * np.diag(G)
print(f"groups={mode}: LOO corr {np.corrcoef(y, loo)[0,1]:+.3f}, explained {1 - np.sum((y-loo)**2)/np.sum(y**2):+.3f}")
beta = np.exp(th[1:])[gid] * (A.T @ w)
print("per-block const coefficients (kappa on the scale variable):")
for f, (b, c, inside, s, fm, j, norm) in enumerate(meta):
    if fm == "const" and abs(beta[f] / norm) > 0.02:
        print(f"  {b:5s} {'ha'[c]} {'in ' if inside else 'out'} {s:4s}: {beta[f] / norm:+.3f}")
order = np.argsort(-np.abs(loo - 0) * 0 - np.abs(y / e2))
print("probe readouts vs LOO (lambda scale), largest |lambda| first:")
for k in order[:25]:
    print(f"  {names[k][:42]:42s} lam {y[k]/e2[k]:+.3f}  loo {loo[k]/e2[k]:+.3f}  (+-{sig[k]/e2[k]:.3f})")
# leave-family-out: hold out probes by name family keyword, refit hyperparameters on the rest
fams = {"chronodyn": ["chronodyn", "cdy", "cdm"], "cl0": ["cl0_"], "mb": ["_mb", "mbx"], "u_": ["dir_u_", "dir_uR"], "cap": ["cap_"],
        "field": ["field"], "pull": ["pull"]}
for fname, keys_ in fams.items():
    hold = np.array([any(kk in n for kk in keys_) for n in names])
    if hold.sum() == 0 or hold.sum() == K: continue
    tr = np.where(~hold)[0]; te = np.where(hold)[0]
    th2 = fit(tr); C2 = cov(th2)
    pred = C2[np.ix_(te, tr)] @ np.linalg.solve(C2[np.ix_(tr, tr)], y[tr])
    print(f"leave-family-out {fname:9s} ({hold.sum():2d} probes): corr {np.corrcoef(y[te], pred)[0,1] if hold.sum() > 2 else float('nan'):+.3f}, "
          f"explained {1 - np.sum((y[te]-pred)**2)/np.sum(y[te]**2):+.3f}; lam meas {np.round(y[te]/e2[te], 2)} pred {np.round(pred/e2[te], 2)}")

# ---- predicted lambda of queued (unscored) probes under the fitted prior ----
import glob, re
centers = json.loads((ROOT / "work/spat_centers.json").read_text())
queued = [("cand", a) for a in sys.argv[2:]]
for f in ([] if queued else sorted(glob.glob(str(ROOT / "scripts/submit_batch_day4[0-9].sh")) + glob.glob(str(ROOT / "scripts/submit_batch_day5[01].sh")))):
    for n in re.findall(r"results/(spat_[A-Za-z0-9_]+\.csv)", open(f).read()):
        if n not in sc and not n.endswith("_mirror.csv") and n not in [q for _, q in queued]:
            queued.append((Path(f).stem[-5:], n))
Tg = np.exp(th[1:])
for day, n in queued:
    c = centers.get(n, "R"); Cp = R if (c == "R" or c.endswith("latent_ref_y0_m114114.csv")) else rd(ROOT / c)
    d = (rd(ROOT / "results" / n) - Cp)
    dv = d.ravel(); rkn = rho * 1.0
    a_new = np.zeros(A.shape[1]); col = 0
    for b, cc, inside in blocks:
        rows = np.where((band == b) & (cl == inside))[0]
        Zb = Zc[rows]
        for s in gp.SCALES:
            v = svars[s][rows, cc]; norm = np.sqrt(np.mean((v[:, None] * Zb) ** 2, axis=0)) + 1e-12
            a_new[col:col + Zb.shape[1]] = rkn * ((d[rows, cc] * v) @ Zb) / N / norm
            col += Zb.shape[1]
    g_new = Dm.T @ dv / N
    cvec = (A * Tg[gid][None, :]) @ a_new + np.exp(th[0]) * rk * g_new * rkn
    vself = a_new @ (Tg[gid] * a_new) + np.exp(th[0]) * rkn * rkn * float(dv @ dv) / N
    mu = cvec @ w; var = vself - cvec @ (Ci @ cvec)
    e2n = rkn * float(np.mean(d * d))
    print(f"  {day} {n[:44]:44s} lambda_pred {mu / e2n:+.3f} +- {np.sqrt(max(var, 0)) / e2n:.3f}  (E {e2n:.4f})")
