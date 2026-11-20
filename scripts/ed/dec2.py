#!/usr/bin/env python
"""Chronosequence decode of exact-input ED rows (11-10 14:10 UTC; scripts/diag/chrono_hybrid.py, three_hat_decodes.py).

    truth40(age) ~ y0 profile at age + 40  x  ED40(age) / ED's own first-record profile at age + 40      (both columns)

i.e. ED supplies the scenario effect on its own chronosequence and the level comes from the test's lookup. Against the
standard decode (ED40 x y0 / first record at the SAME age, gate 0.07) the implied error (three-cornered hat with ORNL and
the stack, solution units) is lower in every seed-mismatch bin (old rows 0.0077 vs 0.0082 below 0.02, 0.0008 vs 0.0056 at
0.02-0.07, 0.0063 vs 0.0122 at 0.07-0.25), so rows up to a mismatch of 0.25 (old) / 0.12 (younger) stay usable.
Age-1 rows keep the old rule (additive agb, raw height, start states small against year 40).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

AGES = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]); LA = np.log(AGES)
DIV = np.array([10.90645, 3.39567]); A = 1.0 / 1.14114; EPS = np.array([0.1, 0.01]); AGB_F = 1.055
FL_A = 0.401412 / 3.39567
# memory of a start-state mismatch after the 40 window years by stratum: (agb, height) at the seed ages MEM_AGES (training world, 120 cells; the boreal
# zone keeps the chrono decode: there ED itself runs 5-10 % below the truth's training runs and the chrono decode was the best one for old stands)
MEM_AGES = np.array([1.0, 10.0, 30.0, 100.0, 150.0, 250.0, 500.0])
MEMORY = {z: (np.array([0.0, 0.1, 0.1, 0.1, 0.4, 0.45, 0.45]), np.array([0.05, 0.15, 0.3, 0.4, 0.4, 0.4, 0.4])) for z in ("cool", "warm", "moist", "dry")}


def predict2(t: pd.DataFrame, keys: pd.DataFrame, gate_old: float = 0.25, gate_young: float = 0.12, gate_h: float = 0.25, qmax_a: float = 0.9, qmax_h: float = 0.5,
             gate_h2: float | None = None, h_site_median: int = 0, memory_zone: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """ED on the template rows in solution units: 'h' and 'a' (agb floor-coded), NaN where unusable; 'mis_h', 'mis_a' the seed mismatch."""
    index = {(int(s), str(sc), int(a)): i for i, (s, sc, a) in enumerate(zip(keys.site, keys.scenario, keys.age))}
    n = len(keys)
    out = {k: np.full(n, np.nan) for k in ("h", "a", "mis_h", "mis_a", "q_h", "q_a")}
    x = np.log(np.minimum(AGES + 40, 500))
    for (s, sc), g in t.groupby(["site", "scenario"], sort=False):
        g = g.drop_duplicates("age").set_index("age").reindex(AGES)
        v = g[["y0_height", "y0_agb", "e0_h", "e0_a", "e40_h", "e40_a"]].to_numpy(dtype=float)
        if not np.isfinite(v).all():
            continue
        y0 = np.column_stack([v[:, 0], v[:, 1] * AGB_F]); e0 = np.maximum(v[:, 2:4], 0.0); e40 = np.maximum(v[:, 4:6], 0.0)
        rows = np.array([index[(int(s), str(sc), int(a))] for a in AGES])
        mis = np.abs(np.log((y0 + EPS) / (e0 + EPS)))
        pred = np.full((len(AGES), 2), np.nan); q = np.full((len(AGES), 2), np.nan)
        for c in (0, 1):
            yc = np.exp(np.interp(x, LA, np.log(y0[:, c] + EPS[c]))); ec = np.exp(np.interp(x, LA, np.log(e0[:, c] + EPS[c])))
            q[:, c] = np.log((e40[:, c] + EPS[c]) / ec)
            pred[:, c] = yc * (e40[:, c] + EPS[c]) / ec - EPS[c]
        if memory_zone is not None and memory_zone[int(s)] in MEMORY:
            # MEMORY decode (11-11 01:50 UTC, scripts/ed/train_memory_eval.py): with the same model and window forcing the truth's year-40 state is
            # ED40 x (y0 / e0)^m whatever the reason for the start mismatch; in the training world m is 0.0-0.2 for seed ages up to 100 and ~0.4 for old
            # stands outside the boreal zone (rms error on rows with a start mismatch of 3-25 %: cool young agb 0.015 against 0.088 chrono / 0.114 hybrid).
            # The chrono decode applies the full lookup mismatch at age + 40, which in the test is as large as the one at the seed age (slope 0.8-1.0).
            ma, mh = MEMORY[memory_zone[int(s)]]
            for c, mm in ((0, mh), (1, ma)):
                sm = np.log((y0[:, c] + EPS[c]) / (e0[:, c] + EPS[c]))
                pm = (e40[:, c] + EPS[c]) * np.exp(np.interp(LA, np.log(MEM_AGES), mm) * sm) - EPS[c]
                yc = np.exp(np.interp(x, LA, np.log(y0[:, c] + EPS[c])))
                pred[:, c] = pm; q[:, c] = np.log((pm + EPS[c]) / yc)          # q stays "effect on the y0 chronosequence at age + 40"
        old = AGES >= 200
        ok_a = np.where(old, mis[:, 1] < gate_old, mis[:, 1] < gate_young) & (np.abs(q[:, 1]) < qmax_a)      # fire runs: qmax_a 3 (a burned stand loses far more than e^-0.9)
        ok_h = (mis[:, 0] < gate_h) & (np.abs(q[:, 0]) < qmax_h)
        if gate_h2 is not None:
            # the decode divides by the ED lookup at age + 40, interpolated between two seed-age snapshots: both must match y0 as well (11-10: single
            # jagged snapshots - site 6081 age 200: ED 12.5 m against y0 21.5 m - sent the neighbouring age-150 row to 34.6 m)
            jb = np.clip(np.searchsorted(LA, x, side="right") - 1, 0, len(LA) - 2)
            ok_h &= np.maximum(mis[jb, 0], mis[jb + 1, 0]) < gate_h2
        if h_site_median:
            # mature rows (seed ages >= 150): ED's height response is a SITE property - in the training world the cell median over the mature ages follows
            # the truth (slope 0.75, corr 0.6 across boreal cells) while its age-to-age scatter inside a cell is noise (slope 0.08; 11-11 00:20 UTC,
            # scripts/ed/train_site_level.py). With >= h_site_median usable mature rows every mature row of the site x scenario takes the median response
            # on its own y0 chronosequence value - also the rows whose own seed failed the gates.
            mat = AGES >= 150
            use = mat & ok_h
            if use.sum() >= h_site_median:
                qm = float(np.median(q[use, 0]))
                yc_h = np.exp(np.interp(x, LA, np.log(y0[:, 0] + EPS[0])))
                pred[mat, 0] = yc_h[mat] * np.exp(qm) - EPS[0]; q[mat, 0] = qm
                ok_h = ok_h | mat
        # age 1: the organisers' start state is one growth year younger than ours; year 40 hardly depends on it
        small = np.maximum(e0[0], y0[0]) <= 0.35 * np.maximum(e40[0], 1e-9)
        pred[0, 1] = max(y0[0, 1] + (e40[0, 1] - e0[0, 1]), 0.0); pred[0, 0] = e40[0, 0]
        ok_a[0] = bool(small[1]); ok_h[0] = bool(small[0])
        out["a"][rows] = np.where(ok_a, pred[:, 1] * A / AGB_F / DIV[1] - FL_A, np.nan)
        out["h"][rows] = np.where(ok_h, pred[:, 0] * A / DIV[0], np.nan)
        out["mis_h"][rows] = mis[:, 0]; out["mis_a"][rows] = mis[:, 1]; out["q_h"][rows] = q[:, 0]; out["q_a"][rows] = q[:, 1]
    return out


def load_roots(roots: list[Path]) -> pd.DataFrame:
    """December rows of several run roots, first root wins for a site."""
    import dec_eval as de
    tabs = []; seen: set[int] = set()
    for r in roots:
        if not Path(r).exists():
            continue
        q = de.load_root(Path(r), "dec")
        if len(q) == 0:
            continue
        q = q[~q.site.isin(seen)]; seen |= set(q.site.unique()); tabs.append(q)
    return pd.concat(tabs, ignore_index=True)
