"""Second screen of fire-rule variants for the TEST windows against ORNL's strict node effect (per site x scenario).
Variant = (source of T, source of P for the dryness index, source of P for the wet-month rule, rule threshold, power, array before year 0).
Sources: 'real' = release transform undone (work/cal/cal_maps_cal.npz), 'rel' = monthly channel exactly as released.
For every variant: yearly factors W -> old-stand emulator with ONE global strength c (scan) -> q_em; and per burning site the strength c_site that
would reproduce ORNL's effect (the spread of log10 c_site is the quantity a correct rule must make small).
    python scripts/fire/rule_screen2.py [max never sites]"""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts' / 'fire')); sys.path.insert(0, str(ROOT / 'scripts' / 'ed'))
from cgb import io
import emulator as em, fire_x as fx
from scipy.stats import spearmanr, rankdata
NNEV = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEED = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
d = pd.read_csv(ROOT / 'work/ff/landscape_factor.csv'); d = d[(d.qs == 1) & np.isfinite(d.q_ornl) & (d.seed.abs() <= SEED) & (d.y0_old > 0.3)].copy()
rng = np.random.default_rng(0); nv = d[d.cls == 'never'].site.unique(); nv = rng.choice(nv, min(NNEV, len(nv)), replace=False)
d = d[(d.cls != 'never') | d.site.isin(nv)].reset_index(drop=True)
print('strict site x scenarios by class:', d.cls.value_counts().to_dict())
z = np.load(ROOT / 'work/cal/cal_maps_cal.npz'); al, be = z['alpha'][:2].astype(float), z['beta'][:2].astype(float)
ex = np.load(ROOT / 'work/test_cell_exact.npz')
X = {sc: io.open_test(sc)[0] for sc in ('ssp126', 'ssp585')}
clim = {}
for s, sc in zip(d.site, d.sc):
    x = np.asarray(X[sc][int(s), :, :, :2], dtype=np.float64)
    cell = int(ex['cell'][int(s)]); lat = float(ex['lat'][int(s)]) if np.isfinite(ex['lat'][int(s)]) else 0.0
    prev = fx._train_tp(cell) if cell >= 0 else None
    clim[(int(s), sc)] = dict(Trel=x[:, :, 0] - 273.15, Prel=x[:, :, 1], Treal=(x[:, :, 0] - be[0]) / al[0] - 273.15, Preal=np.clip((x[:, :, 1] - be[1]) / al[1], 0.0, None), lat=lat,
                              prevT=None if prev is None else prev[0][-1], prevP=None if prev is None else prev[1][-1])
a0 = d.y0_old.to_numpy(float) * 1.055; q = d.q_ornl.to_numpy(float)
pos = (d.cls == 'pos').to_numpy(); nev = (d.cls == 'never').to_numpy(); b = (d.cls == 'b').to_numpy(); cl = (d.cls == 'cluster').to_numpy(); burn = q < -0.3; quiet = q > -0.05
CS = np.logspace(-11, -1, 61)
def auc(score, y):
    r = rankdata(score); n1 = y.sum(); n0 = (~y).sum(); return (r[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)
def factors(tsrc, psrc, rsrc, thr, power, own):
    out = []
    for s, sc in zip(d.site, d.sc):
        c = clim[(int(s), sc)]; T = c['T' + tsrc]; P = c['P' + psrc]
        pT = None if (own or c['prevT'] is None) else (c['prevT'] if tsrc == 'real' else al[0] * (c['prevT'] + 273.15) + be[0] - 273.15)
        pP = None if (own or c['prevP'] is None) else (c['prevP'] if psrc == 'real' else al[1] * c['prevP'] + be[1])
        xr = em.xroll(T, P, c['lat'], pT, pP, power)
        if thr is not None: xr = xr * (c['P' + rsrc] < thr)
        out.append(xr.mean(1))
    return np.array(out)
def evaluate(name, Wv):
    Q = np.empty((len(CS), len(d)))
    for i, c in enumerate(CS):
        a, r = em.simulate(a0, Wv, c=c); Q[i] = np.log(np.maximum(a[:, -1], 1e-6) / a0)
    best = None
    for i, c in enumerate(CS):
        off = np.median((q - Q[i])[pos & quiet]); rm = float(np.sqrt(np.mean((Q[i] + off - q)[pos] ** 2)))
        if best is None or rm < best[0]: best = (rm, i, off)
    rm, i, off = best; qe = Q[i]
    # per-site strength that reproduces ORNL at the burning sites (first crossing of the scan), all burnable classes
    tgt = q - off; lc = np.full(len(d), np.nan)
    for j in np.where(burn & (pos | b | cl))[0]:
        k = np.where(Q[:, j] <= tgt[j])[0]
        if len(k): lc[j] = np.log10(CS[k[0]])
    # quiet sites: the largest strength that keeps the emulated effect above -0.1
    lcq = np.full(len(d), np.nan)
    for j in np.where(quiet & (pos | b))[0]:
        k = np.where(Q[:, j] <= -0.1)[0]; lcq[j] = np.log10(CS[k[0]]) if len(k) else 0.0
    mb = pos & burn; got = np.isfinite(lc[mb]); med = np.nanmedian(lc[mb]) if got.any() else np.nan
    conflict = np.mean(lcq[pos & quiet] < med) if np.isfinite(med) else np.nan      # quiet sites that would burn at the burning sites' median strength
    print(f'{name:44s} | c {CS[i]:.0e} | pos sp {spearmanr(qe[pos], q[pos]).correlation:+.2f} auc {auc(-qe[pos & (burn | quiet)], burn[pos & (burn | quiet)]):.3f} rms {rm:.3f} | burning pos: reachable {got.mean():.2f}, log10 c_site median {med:+.1f} sd {np.nanstd(lc[mb]):.2f} iqr {np.nanpercentile(lc[mb], 75) - np.nanpercentile(lc[mb], 25):.2f} | quiet pos burning at that c: {conflict:.2f} | cluster sp {spearmanr(qe[cl], q[cl]).correlation:+.2f} mean {qe[cl].mean():+.2f} (ORNL {q[cl].mean():+.2f}) sd c_site {np.nanstd(lc[cl & burn]):.2f} | b sp {spearmanr(qe[b], q[b]).correlation:+.2f} | never false {np.mean(qe[nev] < -0.1):.3f}')
    return qe
print(f'pos strict {int(pos.sum())} (burning {int((pos & burn).sum())}, quiet {int((pos & quiet).sum())}); cluster {int(cl.sum())} (burning {int((cl & burn).sum())}); class b {int(b.sum())}; never {int(nev.sum())}; no-fire rms at pos {np.sqrt(np.mean((np.median(q[pos & quiet]) - q[pos]) ** 2)):.3f}')
res = {}
SETS = {'a': [('real T, real P, rule 8.33 real (cx17)', ('real', 'real', 'real', 8.33, 10.0, True)), ('real T, real P, no rule', ('real', 'real', 'real', None, 10.0, True))]
             + [(f'real T, real P, rule {t} real', ('real', 'real', 'real', t, 10.0, True)) for t in (0.3, 0.75, 1.0, 1.5, 2.28, 3.0, 4.0, 6.0)]
             + [(f'released T, real P, rule {t} real', ('rel', 'real', 'real', t, 10.0, True)) for t in (0.75, 1.5, 2.28, 4.0)]
             + [(f'released T, released P, rule {t} released', ('rel', 'rel', 'rel', t, 10.0, True)) for t in (7.0, 7.65, 8.33, 9.8)]}
for name, args in SETS[sys.argv[3] if len(sys.argv) > 3 else 'a']:
    res[name] = evaluate(name, factors(*args))
