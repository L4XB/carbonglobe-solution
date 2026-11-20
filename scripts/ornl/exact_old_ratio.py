"""ORNL existing-forest pixels in the EXACT cell of each test site: growth ratio B2060 / B2021 of the mature pixels
against ED's and the stack's old-row change at the same site (does ORNL carry the same site-level signal?)."""
import sys, glob; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts/ed')
import numpy as np, pandas as pd
from pathlib import Path
from cgb import io, shift
import dec_eval as de
D = Path('work/ornl2522'); RES = 0.04; ROOT = Path('.')
def p2(stem, band):
    a = np.load(D / f'p2_{stem}_b{band}.npy')
    return np.where(a == -32768, np.nan, a.astype(np.float32) * np.float32(0.01))
ex = np.load('work/test_cell_exact.npz'); lat, lon = ex['lat'], ex['lon']
G = {}
for sc in ('126', '585'):
    G[sc] = (p2(f'Biomass_GFDL-ESM4_SSP{sc}_CLIM_CO2_existing_forest_2021_2030', 0), p2(f'Biomass_GFDL-ESM4_SSP{sc}_CLIM_CO2_existing_forest_2051_2060', 9))
G['ref'] = (p2('Biomass_REF_existing_forest_2021_2030', 0), p2('Biomass_REF_existing_forest_2051_2060', 9))
def cell(a, la, lo):
    r0 = int(round((80.0 - (la + 0.25)) / RES)); r1 = int(round((80.0 - (la - 0.25)) / RES))
    c0 = int(round((lo - 0.25 + 180.0) / RES)); c1 = int(round((lo + 0.25 + 180.0) / RES))
    return a[max(r0, 0):r1, c0:c1].ravel()
keys = shift.template_keys()
C = shift.read_predictions('results/spatcorr_stack_20261106_s08.csv').astype(float)
idx = {(int(s), str(sc), int(a)): i for i, (s, sc, a) in enumerate(zip(keys.site, keys.scenario, keys.age))}
strat = de.strata_of_sites(); cl = np.load('work/collapse_cluster_labels.npy') == 0
mx = np.load('work/fire/max_factor_rule833.npy')
parts = [de.load_root(ROOT / r, 'dec') for r in ('work/edv3/full_cx16', 'work/edv3/fire_cx17b')]
seen = set(); keep = []
for q in parts:
    q = q[~q.site.isin(seen)]; seen |= set(q.site.unique()); keep.append(q)
t = pd.concat(keep, ignore_index=True); t = t[t.age >= 150]
rows = []
for (s, sc), q in t.groupby(['site', 'scenario']):
    if not np.isfinite(lat[s]) or int(ex['cell'][s]) < 0: continue
    b0 = cell(G[sc[3:]][0], lat[s], lon[s]); b1 = cell(G[sc[3:]][1], lat[s], lon[s]); r0 = cell(G['ref'][0], lat[s], lon[s]); r1 = cell(G['ref'][1], lat[s], lon[s])
    ok = np.isfinite(b0) & np.isfinite(b1) & (b0 > 0.5)
    if ok.sum() < 12: continue
    thr = np.percentile(b0[ok], 75); top = ok & (b0 >= thr)
    lr = np.log(b1[top] / b0[top]); lr_all = np.log(b1[ok] / b0[ok])
    okr = np.isfinite(r0) & np.isfinite(r1) & (r0 >= thr)
    ii = [idx[(int(s), sc, int(a))] for a in q.age]
    y0 = (q.y0_agb * 1.055).mean(); ed = (q.e40_a * (q.y0_agb * 1.055 + 0.01) / (q.e0_a.clip(lower=0) + 0.01)).mean(); c = (C[ii, 1] * de.DIV[1] / de.A * 1.055).mean()
    seed = float(np.log((y0 + 0.01) / (q.e0_a.mean() + 0.01)))
    rows.append(dict(site=s, scen=sc, n=int(ok.sum()), b0_top=float(np.median(b0[top])), b0_max=float(np.nanmax(b0[ok])), y0=y0, lo_=float(np.median(lr)), lo_sd=float(np.std(lr)), lo_all=float(np.median(lr_all)),
                     lo_ref=float(np.median(np.log(r1[okr] / r0[okr]))) if okr.sum() > 3 else np.nan, le=float(np.log(ed / y0)), lc=float(np.log(c / y0)), seed=seed, st=strat[s], cl=bool(cl[s]), mx=float(mx[s])))
r = pd.DataFrame(rows)
print('site x scenario with ORNL mature pixels and ED:', len(r))
print('y0 old agb vs ORNL 2021 biomass of the top-quartile pixels: median ratio %.2f; corr(log) %.2f; vs the cell maximum: ratio %.2f, corr %.2f' % (np.median(r.y0 / r.b0_top), np.corrcoef(np.log(r.y0), np.log(r.b0_top))[0, 1], np.median(r.y0 / r.b0_max), np.corrcoef(np.log(r.y0), np.log(r.b0_max))[0, 1]))
def rep(q, label):
    if len(q) < 12: return
    cc = lambda a, b: np.corrcoef(q[a], q[b])[0, 1]
    m = lambda a: q[a].median(); sd = lambda a: q[a].std()
    print(f'{label}: n {len(q)}; medians ORNL {m("lo_"):+.3f} (REF {m("lo_ref"):+.3f}) ED {m("le"):+.3f} C {m("lc"):+.3f}; sd ORNL {sd("lo_"):.3f} ED {sd("le"):.3f} C {sd("lc"):.3f}; within-cell sd of ORNL log ratio {m("lo_sd"):.3f} | corr ORNL-ED {cc("lo_", "le"):+.2f}, ORNL-C {cc("lo_", "lc"):+.2f}, ED-C {cc("le", "lc"):+.2f}')
good = r[r.seed.abs() < 0.07]
for sc in ('ssp126', 'ssp585'):
    rep(good[(good.scen == sc) & (good.st == 'moist') & (good.mx < 30)], f'moist fire-never {sc}')
    rep(good[(good.scen == sc) & (good.st == 'warm') & (good.mx < 30)], f'warm fire-never {sc}')
    rep(good[(good.scen == sc) & (good.st == 'cool') & (good.mx < 30)], f'cool fire-never {sc}')
    rep(r[(r.scen == sc) & r.cl & (r.seed.abs() < 0.15)], f'cluster {sc}')
r.to_csv('work/ornl2522/exact_old_ratio.csv', index=False)
