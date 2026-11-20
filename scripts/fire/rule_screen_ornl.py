"""Screen fire-rule variants for the TEST windows against ORNL's strict node effect, site by site (the answer key that did not exist on 11-10).
For every strict site x scenario: yearly ignition factors W[k] of a variant -> the old-stand emulator (scripts/fire/emulator.py) with a global
strength c -> emulated 40-year effect q_em; compared with ORNL's q_old.
    python scripts/fire/rule_screen_ornl.py [n_sites_per_class]
"""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts' / 'fire')); sys.path.insert(0, str(ROOT / 'scripts' / 'ed'))
import emulator as em, fire_x as fx
from scipy.stats import spearmanr
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 400
d = pd.read_csv(ROOT / 'work/ff/landscape_factor.csv'); d = d[(d.qs == 1) & np.isfinite(d.q_ornl) & (d.seed.abs() <= 0.15)].copy()
rng = np.random.default_rng(0); keep = []
for cls, g in d.groupby('cls'):
    s = g.site.unique(); s = s if len(s) <= NMAX else rng.choice(s, NMAX, replace=False); keep.append(g[g.site.isin(s)])
d = pd.concat(keep).reset_index(drop=True)
print('strict site x scenarios by class:', d.cls.value_counts().to_dict())
VARS = ['fut:own:rule8.33:monthly', 'fut:own:monthly', 'rel:own:rule8.33:monthly', 'rel:own:monthly', 'relp:own:rule8.33:monthly', 'relp:own:monthly', 'fut:own:rule2.28:monthly', 'fut:own:rule20:monthly', 'mer']
W = {}
for v in VARS:
    rows = []
    for s, sc in zip(d.site, d.sc):
        try:
            x = fx.series(v, int(s), sc)
        except Exception as e:  # noqa: BLE001
            x = np.full(40, np.nan)
        rows.append(x.reshape(40, 12).mean(1) if x.size == 480 else x)
    W[v] = np.array(rows)
a0 = d.y0_old.to_numpy(float) * 1.055
pos = (d.cls == 'pos').to_numpy(); nev = (d.cls == 'never').to_numpy(); b = (d.cls == 'b').to_numpy(); cl = (d.cls == 'cluster').to_numpy()
q = d.q_ornl.to_numpy(float); burn = q < -0.3; quiet = q > -0.05
def auc(score, y):                       # P(score of a positive > score of a negative)
    from scipy.stats import rankdata
    r = rankdata(score); n1 = y.sum(); n0 = (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
print(f'pos strict: {int(pos.sum())} (burning q < -0.3: {int((pos & burn).sum())}, quiet q > -0.05: {int((pos & quiet).sum())}); never {int(nev.sum())}, class b {int(b.sum())}, cluster {int(cl.sum())}')
print('variant                      | best c | pos: spearman(q_em, q) | AUC burning vs quiet | rms(q_em + off - q) | false burns never (q_em < -0.1) | class b spearman | cluster: mean q_em vs q, spearman | log10 mean W: pos burning / pos quiet / never (medians)')
for v in VARS:
    Wv = W[v]; ok = np.isfinite(Wv).all(axis=1)
    best = None
    for c in np.logspace(-10, -2, 33):
        a, r = em.simulate(a0[ok], Wv[ok], c=c)
        qe = np.full(len(d), np.nan); qe[ok] = np.log(np.maximum(a[:, -1], 1e-6) / a0[ok])
        m = pos & ok
        if (m & burn).sum() < 5: continue
        off = np.nanmedian((q - qe)[m & quiet]) if (m & quiet).any() else 0.0
        rm = float(np.sqrt(np.nanmean((qe + off - q)[m] ** 2)))
        if best is None or rm < best[0]:
            best = (rm, c, qe.copy(), off)
    rm, c, qe, off = best
    m = pos & ok
    sp = spearmanr(qe[m], q[m]).correlation; au = auc(-qe[m & (burn | quiet)], burn[m & (burn | quiet)])
    lw = np.log10(np.maximum(Wv.mean(1), 1e-30))
    print(f'{v:28s} | {c:.1e} | {sp:+.2f} | {au:.3f} | {rm:.3f} (no-fire baseline {np.sqrt(np.nanmean((np.nanmedian(q[m & quiet]) - q[m]) ** 2)):.3f}) | {np.nanmean(qe[nev & ok] < -0.1):.3f} | {spearmanr(qe[b & ok], q[b & ok]).correlation:+.2f} | {np.nanmean(qe[cl & ok]):+.2f} vs {np.nanmean(q[cl & ok]):+.2f}, {spearmanr(qe[cl & ok], q[cl & ok]).correlation:+.2f} | {np.nanmedian(lw[m & burn]):+.1f} / {np.nanmedian(lw[m & quiet]):+.1f} / {np.nanmedian(lw[nev & ok]):+.1f}')
np.savez(ROOT / 'work/fire/rule_screen_W.npz', site=d.site.to_numpy(), sc=d.sc.to_numpy(), q=q, cls=d.cls.to_numpy(), a0=a0, **{v.replace(':', '_').replace('.', 'p'): W[v] for v in VARS})
