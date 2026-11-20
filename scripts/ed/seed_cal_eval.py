#!/usr/bin/env python
"""Cold-site seed calibration: how much does choosing among trend-shifted spin-ups improve the y0 fit?

Reads spin-up-only diagnostics work/edv3/diag_<prefix><variant>_<site>.log (lines "age a height h / y0 y | agb ...")
and the c400 base spin-up (work/edv3/csv/site<id>.csv spin_* columns, shift 0) and reports per band:
  * rms offset-log misfit (height eps 0.1 m, agb eps 0.01 kgC) of the base, of each shift, of the per-site best single
    shift (min total loss over the diagnostic ages) and of the per-age best candidate (run_site.select_seeds' rule);
  * how often each shift wins.
    python scripts/ed/seed_cal_eval.py --prefix cold_ --variants shiftp15,shiftp30,shiftp45
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAT = re.compile(r"age\s+(\d+)\s+height\s+([\d.]+) / y0\s+([\d.]+) \| agb\s+([\d.]+) / y0\s+([\d.]+)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="cold_")
    ap.add_argument("--variants", default="shiftp15,shiftp30,shiftp45")
    args = ap.parse_args()
    rows = []
    for v in args.variants.split(","):
        mnum = re.search(r"(\d+)$", v)
        # numeric variants are trend shifts (shiftp30 -> +30, shiftm10 -> -10); others (geotwin) get key 999
        shift = 999.0 if mnum is None else float(mnum.group(1)) * (-1 if "m" in v.replace("shift", "")[:1] else 1)
        for f in glob.glob(str(ROOT / "work" / "edv3" / f"diag_{args.prefix}{v}_*.log")):
            site = int(re.search(r"_(\d+)\.log$", f).group(1))
            for line in open(f):
                m = PAT.search(line)
                if m:
                    a, h, yh, b, ya = (float(g) for g in m.groups())
                    rows.append(dict(site=site, shift=shift, age=int(a), h=h, yh=yh, a=b, ya=ya))
    d = pd.DataFrame(rows)
    nshift = d.groupby("site")["shift"].nunique()
    full = nshift[nshift == nshift.max()].index
    d = d[d.site.isin(full)]
    base = []
    for site in full:
        p = ROOT / "work" / "edv3" / "csv" / f"site{site:05d}.csv"
        t = pd.read_csv(p); t = t[t.scenario == "ssp126"].set_index("age")
        for a in sorted(d[d.site == site].age.unique()):
            r = d[(d.site == site) & (d.age == a)].iloc[0]
            base.append(dict(site=site, shift=0.0, age=a, h=float(t.spin_height_m[a]), yh=r.yh, a=float(t.spin_agb_kgC[a]), ya=r.ya))
    d = pd.concat([d, pd.DataFrame(base)], ignore_index=True)
    d["lh"] = np.log((d.h.clip(lower=0) + 0.1) / (d.yh + 0.1)); d["la"] = np.log((d.a.clip(lower=0) + 0.01) / (d.ya + 0.01))
    d["loss"] = d.lh ** 2 + d.la ** 2
    d["band"] = np.where(d.age <= 30, "young", np.where(d.age <= 150, "mid", "old"))
    tot = d.groupby(["site", "shift"]).loss.sum().unstack()
    site_best = tot.idxmin(axis=1)
    per_age = d.loc[d.sort_values(["loss", "shift"]).groupby(["site", "age"]).head(1).index]
    single = d[d.apply(lambda r: r["shift"] == site_best[r.site], axis=1)]
    print(f"{len(full)} sites with all {int(nshift.max())} shifts (+ base); ages {sorted(d.age.unique())}")
    out = []
    for name, g in [(f"shift {s:+.0f}", d[d["shift"] == s]) for s in sorted(d["shift"].unique())] + [("site-best single", single), ("per-age best", per_age)]:
        for b in ("young", "mid", "old"):
            x = g[g.band == b]
            out.append(dict(choice=name, band=b, n=len(x), rms_h=np.sqrt(np.mean(x.lh ** 2)), bias_h=x.lh.mean(), rms_a=np.sqrt(np.mean(x.la ** 2)), bias_a=x.la.mean()))
    print(pd.DataFrame(out).round(3).to_string(index=False))
    print("site-best single shift counts:", site_best.value_counts().sort_index().to_dict())
    print("per-age winners by band:", per_age.groupby(["band", "shift"]).size().unstack(fill_value=0).to_dict("index"))
    print("per-site best shift and total loss by shift:")
    print(pd.concat([tot.round(2), site_best.rename("best")], axis=1).to_string())


if __name__ == "__main__":
    main()
