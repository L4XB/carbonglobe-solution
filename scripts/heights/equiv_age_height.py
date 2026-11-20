"""Model-free height from the agb: every site's y0 chronosequence is its own development path (agb(age), height(age)); the height that belongs to a
year-40 agb is read at the EQUIVALENT age where the y0 agb profile reaches that agb (growing part of the profile only).
    python scripts/heights/equiv_age_height.py results/cand48_A.csv A      -> work/heights/heq_A.npy (solution units, NaN where the path does not apply)
Report: against ED's heights and the stack on the rows where ED exists, by class / stratum / seed-age band."""
import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts/ed')
import numpy as np
from pathlib import Path
from cgb import io, shift
fn, tag = sys.argv[1], sys.argv[2]
FL = shift.FLOORS; FLA = 0.401412 / 3.39567
keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy(); scen = keys.scenario.to_numpy()
F = shift.read_predictions(fn).astype(float); S = shift.read_predictions('results/spatcorr_stack_20261110_s08.csv').astype(float)
AG = np.array(io.SEED_AGES, dtype=float); LA = np.log(AG)
heq = np.full(len(keys), np.nan); aeq = np.full(len(keys), np.nan)
for sc in io.SCENARIOS:
    _, y0 = io.open_test(sc); y0 = np.asarray(y0, dtype=np.float64)
    hp = 1.14114 * np.clip(y0[:, :, 0] - FL[0], 0, None); ap = 1.204 * np.clip(y0[:, :, 1] - FL[1], 0, None)      # physical m, kgC, (6171, 15)
    rows = np.where(scen == sc)[0]; a40 = (F[rows, 1] + FLA) * 3.39567 * 1.204
    for i in rows[::15] if False else []: pass
    s_ = site[rows]; ai = np.searchsorted(AG, age[rows])
    for k in range(len(rows)):
        a_prof = ap[s_[k]]; h_prof = hp[s_[k]]
        # growing part: up to the first maximum of the running max that the profile reaches within 3 %
        jmax = int(np.argmax(a_prof >= 0.97 * a_prof.max())) if a_prof.max() > 0.3 else -1
        if jmax < 2: continue
        ag = np.maximum.accumulate(a_prof[:jmax + 1])
        if a40[k] <= ag[0] or a40[k] >= ag[-1] * 1.0: continue          # outside the growing path: no reading
        if not np.all(np.diff(ag) >= 0): continue
        le = np.interp(np.log(a40[k] + 0.02), np.log(ag + 0.02), LA[:jmax + 1])
        aeq[rows[k]] = np.exp(le); heq[rows[k]] = np.interp(le, LA, np.log(h_prof + 0.1))
heq = (np.exp(heq) - 0.1) / (10.90645 * 1.14114)
Path('work/heights').mkdir(exist_ok=True); np.save(f'work/heights/heq_{tag}.npy', heq); np.save(f'work/heights/aeq_{tag}.npy', aeq)
print(f'equivalent-age heights for {fn}: {int(np.isfinite(heq).sum())} rows of {len(heq)}')
import dec2, dec_eval as de
t = dec2.load_roots([Path(r) for r in ('work/edv3/full_cx16', 'work/edv3/fire_cx17b', 'work/edv3/fire_cx17b_srv')]); r = dec2.predict2(t, keys); E = r['h']
strat = np.asarray(de.strata_of_sites())[site]; mxf = np.load('work/fire/max_factor_rule833.npy')[site]; cl = (np.load('work/collapse_cluster_labels.npy') == 0)[site]
hc = np.load('work/lookup_h40_m.npy') / (10.90645 * 1.14114)
print('fire-never rows with ED and an equivalent-age reading: stratum, seed ages | n | median log ratio to the y0 height at age + 40: ED, equivalent-age, stack | rms log: stack / ED, equiv / ED, equiv / stack | corr of the increments over the chronosequence: (equiv, ED), (stack, ED) | squared units stack - ED, equiv - ED')
for st in ('boreal', 'cool', 'warm', 'moist'):
    for nm, lo, hi in (('1', 1, 1), ('10-30', 10, 30), ('50-70', 50, 70), ('100-150', 100, 150)):
        m = ~cl & (mxf < 30) & (strat == st) & (age >= lo) & (age <= hi) & np.isfinite(E) & np.isfinite(heq) & (S[:, 0] > 0.05) & (hc > 0.05) & (E > 0.02)
        if m.sum() < 50: continue
        c = lambda a, b: np.corrcoef(a, b)[0, 1]
        print(f'   {st:6s} {nm:7s}: {int(m.sum()):5d} | {np.median(np.log(E[m] / hc[m])):+.3f} {np.median(np.log(heq[m] / hc[m])):+.3f} {np.median(np.log(S[m, 0] / hc[m])):+.3f} | {np.sqrt(np.mean(np.log(S[m, 0] / E[m]) ** 2)):.3f} {np.sqrt(np.mean(np.log(heq[m] / E[m]) ** 2)):.3f} {np.sqrt(np.mean(np.log(heq[m] / S[m, 0]) ** 2)):.3f} | {c(heq[m] - hc[m], E[m] - hc[m]):+.2f} {c(S[m, 0] - hc[m], E[m] - hc[m]):+.2f} | {np.sum((S[m, 0] - E[m]) ** 2):6.1f} {np.sum((heq[m] - E[m]) ** 2):6.1f}')
