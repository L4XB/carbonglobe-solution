"""Regime consistency by EMULATION (no ED runs): at burnable sites whose y0 old rows sit below the unburned lookup (burned lookups), find the strength c
at which the emulated lookup (the twin's 40 MERRA-2 years, cycled) ends at the y0 biomass, then run the window years with the SAME c and compare the
emulated 40-year effect with ORNL's strict node effect. A rule variant that makes this tight is a rule that holds in lookup and windows alike.
    python scripts/fire/regime_emul.py"""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts' / 'fire')); sys.path.insert(0, str(ROOT / 'scripts' / 'ed'))
from cgb import io
import emulator as em, fire_x as fx
from scipy.stats import spearmanr
d = pd.read_csv(ROOT / 'work/ff/landscape_factor.csv'); d = d[np.isfinite(d.q_ornl) & (d.cls.isin(['pos', 'b'])) & (d.y0_old > 0.3)].copy()
d['aeq'] = d.y0_old * 1.055 / np.exp(d.seed)        # ED's unburned (weak-fire) lookup biomass of the old rows: y0 / exp(log(y0 / seed))
d['a0'] = d.y0_old * 1.055
burned = d[(d.seed < -0.15)].copy()                  # y0 below the unburned lookup
print('burnable site x scenarios with an ORNL value:', len(d), '| burned lookups (y0 more than 15 % below the unburned lookup):', len(burned), '| of these strict:', int((burned.qs == 1).sum()), 'failed state:', int((burned.qs == 7).sum()))
z = np.load(ROOT / 'work/cal/cal_maps_cal.npz'); al, be = z['alpha'][:2].astype(float), z['beta'][:2].astype(float)
ex = np.load(ROOT / 'work/test_cell_exact.npz'); X = {sc: io.open_test(sc)[0] for sc in ('ssp126', 'ssp585')}
CS = np.logspace(-11, -1, 81)
def run(name, tsrc, thr_real, power=10.0, cycles=8, rows=burned):
    out = []
    for s, sc, a0, aeq, q, qs in zip(rows.site, rows.sc, rows.a0, rows.aeq, rows.q_ornl, rows.qs):
        s = int(s); cell = int(ex['cell'][s])
        if cell < 0: continue
        lat = float(ex['lat'][s]) if np.isfinite(ex['lat'][s]) else 0.0
        Tm, Pm = fx._train_tp(cell)                                         # lookup: the twin's MERRA-2 years (real units)
        x = np.asarray(X[sc][s, :, :, :2], dtype=np.float64)
        Preal = np.clip((x[:, :, 1] - be[1]) / al[1], 0.0, None); Treal = (x[:, :, 0] - be[0]) / al[0] - 273.15; Trel = x[:, :, 0] - 273.15
        Tw = Trel if tsrc == 'rel' else Treal
        Tl = (al[0] * (Tm + 273.15) + be[0] - 273.15) if tsrc == 'rel' else Tm   # the lookup's temperature through the same transform
        xl = em.xroll(Tl, Pm, lat, Tl[-1], Pm[-1], power); xw = em.xroll(Tw, Preal, lat, Tl[-1], Pm[-1], power)
        if thr_real is not None:
            xl = xl * (Pm < thr_real); xw = xw * (Preal < thr_real)
        WL = np.tile(xl.mean(1), cycles)[None, :]; WW = xw.mean(1)[None, :]
        # lookup: start at the unburned level, cycle; the state at the end of the last cycle against y0
        A = np.repeat(np.array([aeq]), len(CS)); a, r = em.simulate(A, np.repeat(WL, len(CS), axis=0) * (CS / CS[0])[:, None], c=CS[0], aeq=A)
        end = a[:, -1]
        k = np.where(end <= a0)[0]
        if len(k) == 0 or k[0] == 0:
            out.append((s, sc, np.nan, np.nan, q, qs)); continue
        i = k[0]; lc = np.log10(CS[i - 1]) + (np.log(end[i - 1] / a0) / np.log(end[i - 1] / end[i])) * (np.log10(CS[i]) - np.log10(CS[i - 1]))
        aw, rw = em.simulate(np.array([a0]), WW, c=10 ** lc, aeq=np.array([aeq]))
        out.append((s, sc, lc, float(np.log(aw[0, -1] / a0)), q, qs))
    t = pd.DataFrame(out, columns=['site', 'sc', 'lc', 'q_em', 'q', 'qs'])
    for nm, m in (('strict', t.qs == 1), ('failed state', t.qs == 7), ('proxies', t.qs.isin([4, 5, 6]))):
        g = t[m & np.isfinite(t.q_em)]
        if len(g) < 8: continue
        off = np.median(g.q - g.q_em)
        print(f'{name:36s} {nm:12s} n {len(g):3d} of {int(m.sum()):3d} tunable | log10 c_lookup median {g.lc.median():+.1f} sd {g.lc.std():.2f} | q_em mean {g.q_em.mean():+.3f} sd {g.q_em.std():.3f} | ORNL q mean {g.q.mean():+.3f} sd {g.q.std():.3f} | corr {np.corrcoef(g.q_em, g.q)[0, 1]:+.2f} spearman {spearmanr(g.q_em, g.q).correlation:+.2f} | rms(q - q_em - median) {np.sqrt(np.mean((g.q - g.q_em - off) ** 2)):.3f} (no model: {g.q.std():.3f})')
    return t
for name, a in (('real T, rule 8.33 (cx17)', ('real', 8.33)), ('real T, no rule', ('real', None)), ('real T, rule 2.28', ('real', 2.28)), ('real T, rule 1.0', ('real', 1.0)),
                ('released T, rule 2.28', ('rel', 2.28)), ('released T, rule 1.0', ('rel', 1.0)), ('released T, rule 8.33', ('rel', 8.33)), ('real T, rule 1.0, power 6', ('real', 1.0, 6.0))):
    t = run(name, *a)
