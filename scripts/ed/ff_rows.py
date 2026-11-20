#!/usr/bin/env python
"""Row-level predictions from the forced-fire ED runs (scripts/ed/forced_fire.py) for the candidate builder (--ff-rows).

Per site x scenario x seed age the RAW year-40 December state of the final forced run, corrected for what is left of the seed's miss of y0:

    pred = raw x exp(m x log(y0 / ED seed))        (m = memory of the start state after 40 years)

Training validation (work/edv3/ff_train1, 44 twins of fire-possible test sites, scripts/ed/ff_train_eval.py / ff_train_memory.py; rms of
log(prediction / truth) at year 40, chronosequence -> raw forced-fire state):
    low / young seeds (y0 below 12 m)                 heights 0.39 -> 0.06 (all fire levels)
    forest seeds that burn (agb change < -0.3)        unburned lookup: heights 0.31 -> 0.05 (slope of the truth's change on ED's 0.93, corr 0.96);
                                                      tuned (burned) lookup: 0.28 -> 0.17 (below 20 m: 0.35 -> 0.12; 20-30 m: no gain -> array hb)
    forest seeds, quiet                               heights 0.06 -> 0.10: NO gain (the raw state carries the seed miss; memory 0.3-0.45)
    burned-state seeds (>= 30 m with <= 6 kgC)        heights 0.14 -> 0.14: NO skill
    agb, all types                                    0.56-0.89 -> 0.12-0.31; memory of the seed miss ~0.5
    memory of the seed miss, heights: burning rows 0.0-0.1, quiet rows 0.3-0.45 (age 1: 0.35 / 0.7)
So the arrays written are
    h     heights where the validation shows skill: low / young seeds (memory 0.1 burning, 0.45 quiet) and forest seeds that burn in ED's own run (raw)
    hq    heights of the quiet forest rows (memory 0.45): a separate question (level and CO2 pattern against the stack), not part of h
    a     agb of every row (memory 0.5); the builder uses the ages below 150 (the old rows are the tuning target)
    em_a  row effect of the agb against the y0 chronosequence at age + 40 (chr_a, kgC), memory 0.4: apply_ff.py --o-step takes the within-site PATTERN
          of the old rows from it (the level stays the file's own)
Gates: the site x scenario met its old-row target (|miss| <= --max-miss); seed miss in height <= 0.35 for burning rows, <= 0.15 for quiet rows; seed
miss in agb <= 0.8. "Burning" = the fire-induced agb effect of the row below -0.3: ED's own effect (year 40 against its first-record profile at
age + 40) minus the main-pass (no-fire) effect of the same row where the lookup is the main pass's, the net effect at tuned-lookup sites.

Output work/ff/ff_rows.npz in SOLUTION units (height = m / 1.14114 / 10.90645, agb = kgC / 1.204 / 3.39567 - 0.1182) plus diagnostics.
    python scripts/ed/ff_rows.py work/edv3/ff_pos_srv work/edv3/ff_pos_mac work/edv3/ff_t1 [--out work/ff/ff_rows.npz]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

FL_A = 0.401412 / 3.39567
AGES = np.array([1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]); LA = np.log(AGES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", default="work/ff/ff_rows.npz")
    ap.add_argument("--max-miss", type=float, default=0.25)
    ap.add_argument("--mem-h-burn", type=float, default=0.1)
    ap.add_argument("--mem-h-quiet", type=float, default=0.45)
    ap.add_argument("--mem-a", type=float, default=0.5)
    ap.add_argument("--mem-pat", type=float, default=0.4, help="memory of the seed miss in the row effect used for the old-row PATTERN (training: 0.3-0.5 equal)")
    ap.add_argument("--regime", action="store_true", help="roots of REGIME runs (window fire = the lookup's strength x a scenario factor, no target): the site gate is "
                    "the lookup match (|lookup ratio| <= --max-lookup) instead of the old-row target")
    ap.add_argument("--max-lookup", type=float, default=0.15)
    args = ap.parse_args()
    keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy(); scen = keys.scenario.to_numpy()
    idx = {(int(s), sc, int(a)): i for i, (s, sc, a) in enumerate(zip(site, scen, age))}
    n = len(keys)
    out = {k: np.full(n, np.nan) for k in ("h", "hq", "hb", "a", "h_raw", "a_raw", "seed_h", "seed_a", "cw", "L", "miss", "qa_old", "qa_row", "lookup_fa", "em_a", "chr_a")}
    typ = np.zeros(n, dtype=np.int8)       # 1 low / young seed, 2 forest seed, 3 burned-state seed
    seen: set[int] = set(); nsite = 0
    mp = ROOT / "work/ff/main_pass_row_effect.npz"
    QA_MAIN = np.load(mp)["qa"] if mp.exists() else None
    out["fire"] = np.full(n, np.nan)
    for r in args.roots:
        rp = ROOT / r
        if not (rp / "tune").exists():
            continue
        for jf in sorted((rp / "tune").glob("site*.json")):
            s = int(jf.stem[4:])
            if s in seen or not (rp / "csv" / f"site{s:05d}.csv").exists():
                continue
            j = json.load(open(jf)); t = pd.read_csv(rp / "csv" / f"site{s:05d}.csv"); seen.add(s); nsite += 1
            for sc, g in t.groupby("scenario"):
                g = g.drop_duplicates("age").set_index("age").reindex(AGES)
                if g.end_h_m12.isna().any():
                    continue
                last = [w for w in j["windows"] if w["sc"] == sc]
                qa_old = min(last, key=lambda w: abs(w["qa"] - j["target"][sc]))["qa"] if last else np.nan
                y0h = g.y0_height.to_numpy(float); y0a = g.y0_agb.to_numpy(float) * 1.055
                e0h = np.maximum(g.pre_h_m12.to_numpy(float), 0); e0a = np.maximum(g.pre_agb_m12.to_numpy(float), 0)
                sh = np.log((y0h + 0.3) / (e0h + 0.3)); sa = np.log((y0a + 0.05) / (e0a + 0.05))
                hr = np.maximum(g.end_h_m12.to_numpy(float), 0); ar = np.maximum(g.end_agb_m12.to_numpy(float), 0)
                prof = np.exp(np.interp(np.log(np.minimum(AGES + 40, 500)), LA, np.log(e0a + 0.01)))
                qa = np.log((ar + 0.01) / prof)
                # "burning" = the FIRE-induced agb effect of the row below -0.3 (training validation: forest seeds below that threshold 0.15-0.37 -> 0.03-
                # 0.05, above it no gain over the chronosequence). In the training world the net effect is the fire effect; in the test the CO2 / climate
                # gain hides part of it (4054 ssp585: no-fire run +0.25, forced run -0.14), so where the lookup is the main pass's the main-pass row (weak
                # fire ~ none) is the no-fire reference - but only its GAIN: the main pass burns a little itself at some sites (4054 ssp126: -0.12), and
                # a negative reference would hide that fire. At tuned-lookup sites the row's own profile carries the lookup regime: net effect.
                ii = np.array([idx.get((s, sc, int(a)), -1) for a in AGES])
                ref = np.where(ii >= 0, QA_MAIN[np.maximum(ii, 0)], np.nan) if (QA_MAIN is not None and j["L"] <= 4e-8) else np.full(len(AGES), np.nan)
                fire = np.where(np.isfinite(ref), qa - np.maximum(ref, 0.0), qa)
                burn = fire < -0.3
                tp = np.where((y0h >= 30.0) & (y0a <= 6.0), 3, np.where(y0h >= 12.0, 2, 1))
                mh = np.where(burn, args.mem_h_burn, args.mem_h_quiet)
                hp = np.minimum(hr * np.exp(mh * np.clip(sh, -0.5, 0.5)), 35.0); hraw = np.minimum(hr, 35.0)
                # agb memory of the seed miss: 0.5 for the seed ages 10+, NONE at age 1 (training: raw 0.157, memory 0.5 0.202, memory 1 0.287; in the test
                # ED's age-1 seed sits 0.4-0.5 above y0 on the 0.05 offset, a memory term would pull the row 22 % down - the raw row is within 0.02 +- 0.08
                # of ORNL's measured new-forest effect at quiet sites)
                apred = ar * np.exp(np.where(AGES == 1, 0.0, args.mem_a) * np.clip(sa, -0.8, 0.8))
                # row effect against the y0 chronosequence at age + 40 (kgC, offset 0.05), with the memory of the seed miss: the within-site PATTERN of
                # the old rows (training, 18 burning cells: a flat old-row effect at the true level misses the truth's rows by 0.18-0.21, ED's pattern
                # by 0.06-0.09, corr 0.90, slope 1.04; cells with a mean effect above -0.3: no gain at -0.3..-0.15, 0.053 -> 0.032 at -0.15..0)
                chron = np.maximum(np.exp(np.interp(np.log(np.minimum(AGES + 40, 500)), LA, np.log(y0a + 0.05))) - 0.05, 0.0)
                em = np.log((ar + 0.05) / (chron + 0.05)) + args.mem_pat * np.clip(sa, -0.8, 0.8)
                ok_site = (abs(j["lookup_fa"]) <= args.max_lookup and j["L"] > 4e-8) if args.regime else (j["miss"][sc] <= args.max_miss)
                for k, a in enumerate(AGES):
                    i = idx.get((s, sc, int(a)))
                    if i is None:
                        continue
                    out["h_raw"][i] = hr[k] / 1.14114 / 10.90645; out["a_raw"][i] = ar[k] / 1.204 / 3.39567 - FL_A
                    out["seed_h"][i] = sh[k]; out["seed_a"][i] = sa[k]; out["cw"][i] = j["c"][sc]; out["L"][i] = j["L"]; out["miss"][i] = j["miss"][sc]
                    out["qa_old"][i] = qa_old; out["qa_row"][i] = qa[k]; out["lookup_fa"][i] = j["lookup_fa"]; typ[i] = tp[k]; out["fire"][i] = fire[k]
                    out["chr_a"][i] = chron[k]
                    if not ok_site:
                        continue
                    seed_ok_h = abs(sh[k]) <= (0.35 if burn[k] else 0.15)
                    if abs(sa[k]) <= 0.8 or AGES[k] == 1:
                        out["a"][i] = apred[k] / 1.204 / 3.39567 - FL_A
                        out["em_a"][i] = em[k]
                    if not seed_ok_h:
                        continue
                    if tp[k] == 1:
                        out["h"][i] = hp[k] / 1.14114 / 10.90645
                    elif tp[k] == 2 and burn[k] and (j["L"] <= 4e-8 or y0h[k] < 20.0):
                        # burning forest seeds: all of them where the lookup is the unburned one (validation 0.31 -> 0.05, slope of truth on ED 0.93),
                        # at sites with a tuned (burned) lookup only below 20 m (0.35 -> 0.12; 20-30 m: 0.10-0.22 -> 0.19-0.21, no gain)
                        out["h"][i] = hraw[k] / 1.14114 / 10.90645
                    elif tp[k] == 2 and burn[k]:
                        out["hb"][i] = hraw[k] / 1.14114 / 10.90645
                    elif tp[k] == 2:
                        out["hq"][i] = hp[k] / 1.14114 / 10.90645
    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(ROOT / args.out, typ=typ, **out)
    has = np.isfinite(out["h_raw"])
    print(f"{nsite} sites with a forced run, {int(has.sum())} rows; target missed (> {args.max_miss}): {int((has & (out['miss'] > args.max_miss)).sum())} rows; "
          f"h (young / low seeds + burning forest rows): {int(np.isfinite(out['h']).sum())}, hq (quiet forest rows): {int(np.isfinite(out['hq']).sum())}, "
          f"burned-state seeds (no height): {int((has & (typ == 3)).sum())}, agb rows: {int(np.isfinite(out['a']).sum())} -> {args.out}")


if __name__ == "__main__":
    main()
