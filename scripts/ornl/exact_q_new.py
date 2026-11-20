"""ORNL new-forest runs (planted 2020 on bare ground) per exact cell against ED's age-1 rows: scenario contrast and effect vs REF."""
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
NR = p2('Biomass_REF_new_forest_2051_2060', 9); N = {sc: p2(f'Biomass_GFDL-ESM4_SSP{sc}_CLIM_CO2_new_forest_2051_2060', 9) for sc in ('126', '585')}
def cell(a, la, lo):
    r0 = int(round((80.0 - (la + 0.25)) / RES)); r1 = int(round((80.0 - (la - 0.25)) / RES))
    c0 = int(round((lo - 0.25 + 180.0) / RES)); c1 = int(round((lo + 0.25 + 180.0) / RES))
    return a[max(r0, 0):r1, c0:c1].ravel()
rows = []
for s in range(6171):
    if not np.isfinite(lat[s]) or int(ex['cell'][s]) < 0: continue
    r = cell(NR, lat[s], lon[s]); a = cell(N['126'], lat[s], lon[s]); b = cell(N['585'], lat[s], lon[s])
    ok = np.isfinite(r) & np.isfinite(a) & np.isfinite(b) & (r > 0.3)
    if ok.sum() < 5: continue
    q1 = np.log(a[ok] / r[ok]); q5 = np.log(np.maximum(b[ok], 1e-3) / r[ok])
    rows.append(dict(site=s, n=int(ok.sum()), q126=float(np.median(q1)), q585=float(np.median(q5)), q126_sd=float(np.std(q1)), q585_sd=float(np.std(q5)), nref_med=float(np.median(r[ok])), nref_max=float(np.max(r[ok]))))
o = pd.DataFrame(rows).set_index('site')
print(f'cells with new-forest pixels (REF 2060 > 0.3): {len(o)}; q126 median {o.q126.median():+.3f}, q585 {o.q585.median():+.3f}; within-cell sd of q: {o.q585_sd.median():.3f}; corr(q126, q585) {o[["q126", "q585"]].corr().iloc[0, 1]:+.2f}')
o.to_csv('work/ornl2522/exact_q_new.csv')
_, y1 = io.open_test('ssp126'); _, y5 = io.open_test('ssp585'); F = shift.FLOORS; M = shift.M_HEIGHT
Y = {'ssp126': (y1 - F) * M * np.array([1, 1.055, 1, 1, 1, 1, 1]), 'ssp585': (y5 - F) * M * np.array([1, 1.055, 1, 1, 1, 1, 1])}
ages = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500])
keys = shift.template_keys()
C = shift.read_predictions('results/spatcorr_stack_20261106_s08.csv').astype(float); S = shift.read_predictions('results/spatcorr_stack_20261110_s08.csv').astype(float)
idx = {(int(s), str(sc), int(a)): i for i, (s, sc, a) in enumerate(zip(keys.site, keys.scenario, keys.age))}
strat = de.strata_of_sites(); cl = np.load('work/collapse_cluster_labels.npy') == 0; mx = np.load('work/fire/max_factor_rule833.npy')
parts = [de.load_root(ROOT / r, 'dec') for r in ('work/edv3/full_cx16', 'work/edv3/fire_cx17b')]
seen = set(); keep = []
for q in parts:
    q = q[~q.site.isin(seen)]; seen |= set(q.site.unique()); keep.append(q)
t = pd.concat(keep, ignore_index=True); t = t[t.age == 1]
rr = []
for s, q in t.groupby('site'):
    if s not in o.index or len(q) < 2: continue
    q = q.set_index('scenario')
    rec = dict(site=s, st=strat[s], cl=bool(cl[s]), mx=float(mx[s]), nmax=o.loc[s, 'nref_max'])
    for sc in ('ssp126', 'ssp585'):
        chrono = float(np.interp(41, ages, Y[sc][s, :, 1]))          # the lookup at age 41 (REF proxy)
        e = float(q.loc[sc, 'e40_a']) - (float(q.loc[sc, 'e0_a']) - float(q.loc[sc, 'y0_agb']) * 1.055)
        i = idx[(int(s), sc, 1)]
        rec.update({f'ed_{sc[3:]}': np.log(max(e, 0.05) / max(chrono, 0.05)), f'c_{sc[3:]}': np.log(max(C[i, 1] * de.DIV[1] / de.A * 1.055, 0.05) / max(chrono, 0.05)), f's_{sc[3:]}': np.log(max(S[i, 1] * de.DIV[1] / de.A * 1.055, 0.05) / max(chrono, 0.05)), f'o_{sc[3:]}': o.loc[s, 'q' + sc[3:]], f'chrono_{sc[3:]}': chrono})
    rr.append(rec)
r = pd.DataFrame(rr)
r['ratio_units'] = r.chrono_126 / r.nmax
print('age-1 rows with ED and ORNL new forest:', len(r), '; lookup agb at age 41 / ORNL REF new forest 2060 (cell max): median %.2f, 16-84 %%: %.2f..%.2f' % (r.ratio_units.median(), r.ratio_units.quantile(0.16), r.ratio_units.quantile(0.84)))
def rep(q, label):
    if len(q) < 12: return
    cc = lambda a, b: np.corrcoef(q[a], q[b])[0, 1]
    for sc in ('126', '585'):
        print(f'{label} ssp{sc}: n {len(q)}; medians vs the lookup at age 41: ORNL q {q["o_" + sc].median():+.3f}, ED {q["ed_" + sc].median():+.3f}, C {q["c_" + sc].median():+.3f}, stack {q["s_" + sc].median():+.3f}; sd ORNL {q["o_" + sc].std():.3f} ED {q["ed_" + sc].std():.3f} C {q["c_" + sc].std():.3f} | corr ORNL-ED {cc("o_" + sc, "ed_" + sc):+.2f}, ORNL-C {cc("o_" + sc, "c_" + sc):+.2f}, ED-C {cc("ed_" + sc, "c_" + sc):+.2f}')
    q = q.assign(co=q.o_585 - q.o_126, ce=q.ed_585 - q.ed_126, cc_=q.c_585 - q.c_126)
    print(f'{label} scenario contrast (585 - 126): medians ORNL {q.co.median():+.3f}, ED {q.ce.median():+.3f}, C {q.cc_.median():+.3f}; corr ORNL-ED {np.corrcoef(q.co, q.ce)[0, 1]:+.2f}, ORNL-C {np.corrcoef(q.co, q.cc_)[0, 1]:+.2f}')
rep(r[(r.st == 'moist') & (r.mx < 30)], 'moist fire-never')
rep(r[(r.st == 'warm') & (r.mx < 30)], 'warm fire-never')
rep(r[(r.st.isin(['cool', 'boreal'])) & (r.mx < 30)], 'cool + boreal fire-never')
rep(r[r.cl], 'cluster')
r.to_csv('work/ornl2522/exact_q_new_vs_ed.csv', index=False)
