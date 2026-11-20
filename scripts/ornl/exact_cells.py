"""ORNL DAAC 2522 at the EXACT 0.5-deg cells of the test sites (work/test_cell_exact.npz), page-2 pixels (0.04 deg).
Step 1: is the product constant within an ED cell (new forest), and does REF new forest equal the test lookup (y0)?"""
import sys; sys.path.insert(0, 'src')
import numpy as np
from pathlib import Path
from cgb import io, shift
D = Path('work/ornl2522'); RES = 0.04
def p2(stem, band):
    a = np.load(D / f'p2_{stem}_b{band}.npy')
    return np.where(a == -32768, np.nan, a.astype(np.float32) * np.float32(0.01))
ex = np.load('work/test_cell_exact.npz'); lat, lon = ex['lat'], ex['lon']
_, y1 = io.open_test('ssp126'); _, y5 = io.open_test('ssp585'); F = shift.FLOORS; M = shift.M_HEIGHT
Y1 = (y1 - F) * M; Y5 = (y5 - F) * M
ages = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
def cell_pixels(a, la, lo):
    r0 = int(round((80.0 - (la + 0.25)) / RES)); r1 = int(round((80.0 - (la - 0.25)) / RES))
    c0 = int(round((lo - 0.25 + 180.0) / RES)); c1 = int(round((lo + 0.25 + 180.0) / RES))
    return a[max(r0, 0):r1, c0:c1]
grids = {k: p2(*v) for k, v in dict(ref21=('Biomass_REF_new_forest_2021_2030', 0), ref30=('Biomass_REF_new_forest_2021_2030', 9), ref51=('Biomass_REF_new_forest_2051_2060', 0), ref60=('Biomass_REF_new_forest_2051_2060', 9),
                                    n126=('Biomass_GFDL-ESM4_SSP126_CLIM_CO2_new_forest_2051_2060', 9), n585=('Biomass_GFDL-ESM4_SSP585_CLIM_CO2_new_forest_2051_2060', 9)).items()}
print('grid shape', grids['ref21'].shape)
rows = []
for s in range(6171):
    if not np.isfinite(lat[s]) or int(ex['cell'][s]) < 0: continue
    v = {k: cell_pixels(g, lat[s], lon[s]) for k, g in grids.items()}
    ok = np.isfinite(v['ref30'])
    if ok.sum() < 5: continue
    rec = dict(site=s, n=int(ok.sum()))
    for k in grids:
        x = v[k][np.isfinite(v[k])]
        rec[k] = float(np.median(x)) if x.size else np.nan; rec[k + '_sd'] = float(np.std(x)) if x.size else np.nan
    rows.append(rec)
import pandas as pd
r = pd.DataFrame(rows).set_index('site')
print('sites with new-forest pixels in their exact cell:', len(r), '; median pixels per cell', r.n.median())
for k in grids:
    print(f'   {k}: median cell value {r[k].median():.2f}; within-cell sd: median {r[k + "_sd"].median():.3f}, 90 % {r[k + "_sd"].quantile(0.9):.3f}; cells with sd < 0.02: {np.mean(r[k + "_sd"] < 0.02):.2f}')
s = r.index.to_numpy()
ya = lambda a: 0.5 * (Y1[s, ages.index(a), 1] + Y5[s, ages.index(a), 1])
for nm, col, y in (('y0 age 1 vs REF new 2021', 'ref21', ya(1)), ('y0 age 10 vs REF new 2030', 'ref30', ya(10)), ('y0 age 30 vs REF new 2051', 'ref51', ya(30)), ('y0 interpolated age 40 vs REF new 2060', 'ref60', 0.5 * ya(30) + 0.5 * ya(50)), ('y0 age 30 vs REF new 2060', 'ref60', ya(30)), ('y0 age 50 vs REF new 2060', 'ref60', ya(50))):
    b = r[col].to_numpy(); ok = np.isfinite(b) & (y > 0.05) & (b > 0.05)
    lr = np.log(y[ok] / b[ok])
    print(f'{nm}: n {ok.sum()}; median ratio y0 / ORNL {np.exp(np.median(lr)):.3f}; scatter of log ratio (16-84 %) {0.5 * (np.percentile(lr, 84) - np.percentile(lr, 16)):.3f}; corr(log) {np.corrcoef(np.log(y[ok]), np.log(b[ok]))[0, 1]:+.3f}; within 5 %: {np.mean(np.abs(lr - np.median(lr)) < 0.05):.2f}')
r.to_csv('work/ornl2522/exact_cells_newforest.csv')
