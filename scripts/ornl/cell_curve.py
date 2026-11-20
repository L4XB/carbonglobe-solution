"""Look at the ORNL pixels of one exact cell: scenario growth log(B2060_ssp / B2021) against REF growth log(B2060_REF / B2021)."""
import sys; sys.path.insert(0, 'src')
import numpy as np
from pathlib import Path
from cgb import io, shift
D = Path('work/ornl2522'); RES = 0.04
def p2(stem, band):
    a = np.load(D / f'p2_{stem}_b{band}.npy')
    return np.where(a == -32768, np.nan, a.astype(np.float32) * np.float32(0.01))
ex = np.load('work/test_cell_exact.npz'); lat, lon = ex['lat'], ex['lon']
R0 = p2('Biomass_REF_existing_forest_2021_2030', 0); R1 = p2('Biomass_REF_existing_forest_2051_2060', 9)
S1 = {sc: p2(f'Biomass_GFDL-ESM4_SSP{sc}_CLIM_CO2_existing_forest_2051_2060', 9) for sc in ('126', '585')}
def cell(a, la, lo):
    r0 = int(round((80.0 - (la + 0.25)) / RES)); r1 = int(round((80.0 - (la - 0.25)) / RES))
    c0 = int(round((lo - 0.25 + 180.0) / RES)); c1 = int(round((lo + 0.25 + 180.0) / RES))
    return a[max(r0, 0):r1, c0:c1].ravel()
_, y1 = io.open_test('ssp126'); _, y5 = io.open_test('ssp585'); F = shift.FLOORS; M = shift.M_HEIGHT
ages = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500])
for s in map(int, sys.argv[1:]):
    r0 = cell(R0, lat[s], lon[s]); r1 = cell(R1, lat[s], lon[s]); s126 = cell(S1['126'], lat[s], lon[s]); s585 = cell(S1['585'], lat[s], lon[s])
    ok = np.isfinite(r0) & np.isfinite(r1) & np.isfinite(s126) & np.isfinite(s585) & (r0 > 0.05)
    y0 = 0.5 * ((y1[s, :, 1] - F[1]) + (y5[s, :, 1] - F[1])) * M * 1.055
    print(f'site {s} (lat {lat[s]:.2f} lon {lon[s]:.2f}): {int(ok.sum())} existing-forest pixels; B2021 quantiles 5/25/50/75/95/max: ' + ' '.join(f'{v:.2f}' for v in np.percentile(r0[ok], [5, 25, 50, 75, 95, 100])))
    print('   y0 agb by seed age: ' + ' '.join(f'{v:.2f}' for v in y0))
    gR = np.log(r1[ok] / r0[ok]); g1 = np.log(s126[ok] / r0[ok]); g5 = np.log(s585[ok] / r0[ok]); b = r0[ok]
    order = np.argsort(b)
    # bins of initial biomass
    edges = np.unique(np.percentile(b, np.linspace(0, 100, 13)))
    print('   B2021 bin (median)   n | REF growth | ssp126 growth | ssp585 growth | B2060 REF 126 585 (medians)')
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        m = (b >= lo_) & (b <= hi_)
        if m.sum() < 2: continue
        print(f'   {np.median(b[m]):6.2f} {int(m.sum()):4d} | {np.median(gR[m]):+.3f} (sd {np.std(gR[m]):.3f}) | {np.median(g1[m]):+.3f} | {np.median(g5[m]):+.3f} | {np.median(b[m] * np.exp(gR[m])):6.2f} {np.median(b[m] * np.exp(g1[m])):6.2f} {np.median(b[m] * np.exp(g5[m])):6.2f}')
