"""Truth of the TRAINING world: how do heights move when agb is lost in the 40 window years? Old rows (seed ages 300 and 500), December of year 40 against
year 0 of the same row; forested sites only; by the size of the agb change. Also the young-to-mid check h40(age 10) against h0(age 50)."""
import sys; sys.path.insert(0, 'src')
import numpy as np
from cgb import io, shift
FL = shift.FLOORS; M = shift.M_HEIGHT
def phys(y, v):   # released units -> physical (height m, agb decoded kgC)
    return (np.asarray(y, dtype=np.float64) - FL[v]) * M
out = []
for a in (300, 500):
    y = io.open_train_y(a)                     # (sites, 41, 12, 7)
    h0 = phys(y[:, 0, 11, 0], 0); h40 = phys(y[:, 40, 11, 0], 0); a0 = phys(y[:, 0, 11, 1], 1); a40 = phys(y[:, 40, 11, 1], 1)
    m = (a0 > 3.0) & (h0 > 8.0) & (a40 > 0.05) & (h40 > 0.5)
    da = np.log(a40[m] / a0[m]); dh = np.log(h40[m] / h0[m])
    out.append((da, dh)); print(f'seed age {a}: {int(m.sum())} forested training sites; agb change median {np.median(da):+.3f}, height change median {np.median(dh):+.3f}')
da = np.concatenate([o[0] for o in out]); dh = np.concatenate([o[1] for o in out])
print('agb log change over the window (old rows): n | median height log change | pct 10 / 90 | share with a height RISE above 5 %')
for lo, hi in ((-9, -1.0), (-1.0, -0.5), (-0.5, -0.25), (-0.25, -0.1), (-0.1, -0.03), (-0.03, 0.03), (0.03, 0.1), (0.1, 9)):
    m = (da >= lo) & (da < hi)
    if m.sum() < 20: continue
    print(f'   [{lo:+.2f}, {hi:+.2f}): {int(m.sum()):6d} | {np.median(dh[m]):+.3f} | {np.percentile(dh[m], 10):+.3f} {np.percentile(dh[m], 90):+.3f} | {np.mean(dh[m] > 0.05):.2f}')
m = da < -0.1
b = np.polyfit(da[m], dh[m], 1); print(f'loss rows (agb change < -0.1): height change = {b[1]:+.3f} {b[0]:+.3f} x agb change (corr {np.corrcoef(da[m], dh[m])[0, 1]:+.2f}, n {int(m.sum())})')
