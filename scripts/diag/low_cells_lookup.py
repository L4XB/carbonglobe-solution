"""Low agb cells (stack <= 0.02): what does the test's own lookup say at age + 40? Bare (0 kgC) or a grass state (0.3-0.9 kgC, solution value near 0)?"""
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts/ed'); sys.path.insert(0, 'scripts/ornl')
import numpy as np
from cgb import io, shift
import dec_eval as de, ornl_rows as orr
FL_A = 0.401412 / 3.39567
AGES = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]); LA = np.log(AGES)
keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy(); scen = keys.scenario.to_numpy()
S = shift.read_predictions('results/spatcorr_stack_20261110_s08.csv').astype(float)
n = len(S); L = np.zeros(n); L0 = np.zeros(n)
for sc in ('ssp126', 'ssp585'):
    y = (np.asarray(io.open_test(sc)[1][:, :, 1], dtype=np.float64) - shift.FLOORS[1]) * shift.M_HEIGHT * orr.F_AGB      # kgC, (sites, ages)
    rows = np.where(scen == sc)[0]; x = np.log(np.minimum(age[rows] + 40, 500))
    j = np.clip(np.searchsorted(LA, x, side='right') - 1, 0, len(AGES) - 2); f = (x - LA[j]) / (LA[j + 1] - LA[j])
    ys = np.maximum(y[site[rows]], 0.0)
    L[rows] = (1 - f) * ys[np.arange(len(rows)), j] + f * ys[np.arange(len(rows)), j + 1]
    L0[rows] = ys[np.arange(len(rows)), np.searchsorted(AGES, age[rows])]
np.save('work/lookup_agb40_kgC.npy', L.astype(np.float32))
low = S[:, 1] <= 0.02
print(f'low cells: {int(low.sum())}; lookup at age + 40 (kgC): share = 0: {np.mean(L[low] < 0.01):.3f}, 0.01-0.1: {np.mean((L[low] >= 0.01) & (L[low] < 0.1)):.3f}, 0.1-0.3: {np.mean((L[low] >= 0.1) & (L[low] < 0.3)):.3f}, 0.3-1: {np.mean((L[low] >= 0.3) & (L[low] < 1)):.3f}, > 1: {np.mean(L[low] >= 1):.3f}')
sol = lambda kg: kg / 1.204 / 3.39567 - FL_A
m = low & (L >= 0.1)
print(f'low cells whose lookup is above 0.1 kgC: {int(m.sum())}; the lookup-coded value would be {sol(L[m]).mean():+.3f} on average (floor -0.118); sum sq of (lookup-coded + 0.1182): {np.sum((sol(L[m]) + FL_A) ** 2):.1f} squared units')
print('stack values by lookup level (all cells): lookup kgC bin: cells | median stack | share of stack <= 0.02 | lookup-coded solution value')
for lo, hi in ((0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 2.0)):
    mm = (L >= lo) & (L < hi)
    print(f'   {lo:.2f}-{hi:.2f}: {int(mm.sum()):6d} | {np.median(S[mm, 1]):+.3f} | {np.mean(S[mm, 1] <= 0.02):.2f} | {sol((lo + hi) / 2):+.3f}')
