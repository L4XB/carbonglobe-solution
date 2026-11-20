#!/usr/bin/env python
"""Register the files of a probe design in work/spat_centers.json so that scripts/fire/consistency_all.py (and every fit built on it) uses them:
the candidate (the file without any factor) gets the 11-10 stack as centre, every probe file gets the candidate.
    python scripts/register_centres.py work/logs/cand50_design.json [--stack results/spatcorr_stack_20261110_s08.csv] [--dry]
Only files that carry a board score are registered; a backup of the json is written next to it."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("design"); ap.add_argument("--stack", default="results/spatcorr_stack_20261110_s08.csv"); ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    des = json.load(open(ROOT / args.design)); sc = json.load(open(ROOT / "results/board_scores.json")); cf = ROOT / "work/spat_centers.json"; cen = json.load(open(cf))
    cand = [f for f, row in zip(des["files"], des["include"]) if sum(row) == 0]
    assert len(cand) == 1, "the design needs exactly one file without factors"
    new = {}
    for f in des["files"]:
        if sc.get(f) is None:
            print(f"   {f}: no board score yet - skipped"); continue
        new[f] = args.stack if f == cand[0] else f"results/{cand[0]}"
    if sc.get(cand[0]) is None:
        new = {k: v for k, v in new.items() if k == cand[0]}
        print("   the candidate has no score: probes cannot be registered")
    for k, v in new.items():
        print(f"   {k} -> centre {v}" + (" (was " + str(cen[k]) + ")" if k in cen and cen[k] != v else ""))
    if not args.dry and new:
        shutil.copy(cf, cf.with_name(cf.name + f".bak_{time.strftime('%m%d_%H%M', time.gmtime())}"))
        cen.update(new); cf.write_text(json.dumps(cen, indent=1)); print(f"wrote {cf.relative_to(ROOT)} ({len(cen)} entries)")


if __name__ == "__main__":
    main()
