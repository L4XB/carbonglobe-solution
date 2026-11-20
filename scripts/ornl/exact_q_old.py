"""ORNL scenario effect on OLD stands per exact cell: q = median over old-growth pixels of log(B2060_ssp / B2060_REF),
old-growth = pixels whose REF run barely grows (log B2060_REF / B2021_REF < gmax). Compared with ED and the stack."""
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts/ed')
import numpy as np, pandas as pd
from pathlib import Path
from cgb import io, shift
import dec_eval as de
D = Path('work/ornl2522'); RES = 0.04; ROOT = Path('.')
def p2(stem, band):
    a = np.load(D / f'p2_{stem}_b{band}.npy')
    return np.where(a == -32768, np.nan, a.astype(np.float32) * np.float32(0.01))
ex = np.load('work/test_cell_exact.npz'); lat, lon = ex['lat'], ex['lon']
R0 = p2('Biomass_REF_existing_forest_2021_2030', 0); R1 = p2('Biomass_REF_existing_forest_2051_2060', 9)
S1 = {sc: p2(f'Biomass_GFDL-ESM4_SSP{sc}_CLIM_CO2_existing_forest_2051_2060', 9) for sc in ('126', '585')}
S0 = {sc: p2(f'Biomass_GFDL-ESM4_SSP{sc}_CLIM_CO2_existing_forest_2021_2030', 0) for sc in ('126', '585')}
def cell(a, la, lo):
    r0 = int(round((80.0 - (la + 0.25)) / RES)); r1 = int(round((80.0 - (la - 0.25)) / RES))
    c0 = int(round((lo - 0.25 + 180.0) / RES)); c1 = int(round((lo + 0.25 + 180.0) / RES))
    return a[max(r0, 0):r1, c0:c1].ravel()
gmax = float(sys.argv[1]) if len(sys.argv) > 1 else 0.12
rows = []
for s in range(6171):
    if not np.isfinite(lat[s]) or int(ex['cell'][s]) < 0: continue
    r0 = cell(R0, lat[s], lon[s]); r1 = cell(R1, lat[s], lon[s])
    ok = np.isfinite(r0) & np.isfinite(r1) & (r0 > 1.0)
    if ok.sum() < 8: continue
    g = np.log(r1 / np.maximum(r0, 1e-6))
    old = ok & (g < gmax) & (g > -0.2)
    rec = dict(site=s, n=int(ok.sum()), n_old=int(old.sum()), g_old=float(np.median(g[old])) if old.sum() else np.nan, b_old=float(np.median(r0[old])) if old.sum() else np.nan)
    for sc in ('126', '585'):
        s1 = cell(S1[sc], lat[s], lon[s]); s0 = cell(S0[sc], lat[s], lon[s])
        m = old & np.isfinite(s1) & (s1 > 0)
        if m.sum() >= 4:
            q = np.log(s1[m] / r1[m])
            rec[f'q{sc}'] = float(np.median(q)); rec[f'q{sc}_sd'] = float(np.std(q)); rec[f'q{sc}_n'] = int(m.sum())
            rec[f'd0_{sc}'] = float(np.median(np.log(np.maximum(s0[m], 1e-6) / r0[m])))       # same 2021 state in both runs?
    rows.append(rec)
o = pd.DataFrame(rows).set_index('site')
print(f'cells with existing-forest pixels: {len(o)}; with >= 4 old-growth pixels: {int(o.q585.notna().sum())}; median old pixels per cell {o.n_old.median():.0f}')
print(f'2021 state SSP vs REF among old pixels: median log ratio {o.d0_585.median():+.4f}, sd {o.d0_585.std():.4f}')
print(f'q126 median {o.q126.median():+.3f} sd {o.q126.std():.3f}; q585 median {o.q585.median():+.3f} sd {o.q585.std():.3f}; within-cell sd of q: {o.q585_sd.median():.3f}; corr(q126, q585) {o[["q126", "q585"]].dropna().corr().iloc[0, 1]:+.2f}')
o.to_csv('work/ornl2522/exact_q_old.csv')
# against ED and C (old rows): log change relative to y0
keys = shift.template_keys()
C = shift.read_predictions('results/spatcorr_stack_20261106_s08.csv').astype(float); S = shift.read_predictions('results/spatcorr_stack_20261110_s08.csv').astype(float)
idx = {(int(s), str(sc), int(a)): i for i, (s, sc, a) in enumerate(zip(keys.site, keys.scenario, keys.age))}
strat = de.strata_of_sites(); cl = np.load('work/collapse_cluster_labels.npy') == 0; mx = np.load('work/fire/max_factor_rule833.npy')
parts = [de.load_root(ROOT / r, 'dec') for r in ('work/edv3/full_cx16', 'work/edv3/fire_cx17b')]
seen = set(); keep = []
for q in parts:
    q = q[~q.site.isin(seen)]; seen |= set(q.site.unique()); keep.append(q)
t = pd.concat(keep, ignore_index=True); t = t[t.age >= 150]
rr = []
for (s, sc), q in t.groupby(['site', 'scenario']):
    if s not in o.index or not np.isfinite(o.loc[s, 'q' + sc[3:]]): continue
    ii = [idx[(int(s), sc, int(a))] for a in q.age]
    y0 = (q.y0_agb * 1.055).mean(); ed = (q.e40_a * (q.y0_agb * 1.055 + 0.01) / (q.e0_a.clip(lower=0) + 0.01)).mean()
    rr.append(dict(site=s, scen=sc, qo=o.loc[s, 'q' + sc[3:]], go=o.loc[s, 'g_old'], ded=float(np.log(ed / y0)), dc=float(np.log((C[ii, 1] * de.DIV[1] / de.A * 1.055).mean() / y0)), ds=float(np.log((S[ii, 1] * de.DIV[1] / de.A * 1.055).mean() / y0)),
                   seed=float(np.log((y0 + 0.01) / (q.e0_a.mean() + 0.01))), st=strat[s], cl=bool(cl[s]), mx=float(mx[s]), nold=int(o.loc[s, 'n_old'])))
r = pd.DataFrame(rr)
def rep(q, label):
    if len(q) < 12: return
    cc = lambda a, b: np.corrcoef(q[a], q[b])[0, 1]
    print(f'{label}: n {len(q)}; medians: ORNL q {q["qo"].median():+.3f}, ED change {q["ded"].median():+.3f}, C {q["dc"].median():+.3f}, stack 11-10 {q["ds"].median():+.3f}; sd: ORNL {q["qo"].std():.3f} ED {q["ded"].std():.3f} C {q["dc"].std():.3f} | corr ORNL-ED {cc("qo", "ded"):+.2f}, ORNL-C {cc("qo", "dc"):+.2f}, ORNL-stack {cc("qo", "ds"):+.2f}, ED-C {cc("ded", "dc"):+.2f}; slope ED on ORNL {np.polyfit(q["qo"], q["ded"], 1)[0]:+.2f}')
for sc in ('ssp126', 'ssp585'):
    g = r[(r.scen == sc) & (r.nold >= 10)]
    rep(g[(g.st == 'moist') & (g.mx < 30) & (g.seed.abs() < 0.07)], f'moist fire-never {sc}')
    rep(g[(g.st == 'warm') & (g.mx < 30) & (g.seed.abs() < 0.07)], f'warm fire-never {sc}')
    rep(g[(g.st.isin(["cool", "boreal"])) & (g.mx < 30) & (g.seed.abs() < 0.07)], f'cool + boreal fire-never {sc}')
    rep(g[g.cl & (g.seed.abs() < 0.15)], f'cluster {sc}')
r.to_csv('work/ornl2522/exact_q_old_vs_ed.csv', index=False)
