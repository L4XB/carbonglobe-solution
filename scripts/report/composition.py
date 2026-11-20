#!/usr/bin/env python
"""Where do the values of a finished candidate come from? Shares per source for the technical report.

    python scripts/report/composition.py work/logs/day50_spec.json results/cand50_A.csv

Reads the part labels of the base recipe's builder runs (work/part_td_<key>.npy, written by scripts/build_candidate2.py through
scripts/design_run.py) and the finished file, and prints cell counts and second-moment weights per source group.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cgb import shift  # noqa: E402
import design_run as dr  # noqa: E402

FL = 0.401412 / 3.39567


def main() -> None:
    spec = json.load(open(ROOT / sys.argv[1])); P = shift.read_predictions(ROOT / sys.argv[2]).astype(np.float64)
    S = shift.read_predictions(ROOT / "results/spatcorr_stack_20261110_s08.csv").astype(np.float64)
    r0 = {k: v for k, v in spec["base"].items() if k != "mix"}
    keys = shift.template_keys(); site = keys.site.to_numpy()
    cl = (np.load(ROOT / "work/collapse_cluster_labels.npy") == 0)[site]; mxf = np.load(ROOT / "work/fire/max_factor_rule833.npy")[site]
    pos = ~cl & (mxf >= 1e3)
    for v6 in (False, True):
        args = dr.builder_args(r0, v6); key = hashlib.sha1(" ".join(args).encode()).hexdigest()[:10]
        f = ROOT / f"work/part_td_{key}.npy"
        print(f"rows {'v6' if v6 else 'v4'}: key {key} labels {'found' if f.exists() else 'MISSING'}")
        if not f.exists():
            continue
        part = np.load(f)
        steps = {**__import__('build_candidate2').DEFAULT_STEPS, **r0.get("steps", {})}
        for c, cn, veg in ((1, "agb", (S[:, 1] + FL) > 0.02), (0, "height", S[:, 0] > 0.02)):
            lab = part[:, c].astype(str); w = (S[:, c] + (FL if c == 1 else 0.0)) ** 2
            grp = np.full(len(lab), "blend (unchanged)", dtype=object)
            if c == 1:
                grp[np.char.startswith(lab, "ed_")] = "ED on the test forcing"
                grp[np.char.startswith(lab, "both_")] = "ED + projections"
                grp[np.isin(lab, ("low", "midlow"))] = "low-biomass coding"
                other = (lab != "") & ~np.char.startswith(lab, "ed_") & ~np.char.startswith(lab, "both_") & ~np.isin(lab, ("low", "midlow"))
                grp[other] = "projections"
            else:
                grp[np.char.startswith(lab, "edh")] = "ED heights"
                grp[(lab != "") & ~np.char.startswith(lab, "edh")] = "other height parts"
            zero = np.array([steps.get(x, 1.0) == 0.0 for x in lab]) & (lab != "")
            grp[zero] = "blend (part at step 0)"
            print(f"  {cn}: vegetated cells {int(veg.sum())} of {len(veg)}")
            for g in sorted(set(grp)):
                m = (grp == g) & veg
                print(f"     {g:26s} cells {int(m.sum()):6d} ({100 * m.sum() / veg.sum():5.1f} %)  second-moment weight {100 * w[m].sum() / w[veg].sum():5.1f} %   mean |file - blend| {np.mean(np.abs(P[m, c] - S[m, c])) if m.any() else 0:.4f}")
            if c == 0:
                m = pos & veg & (np.abs(P[:, 0] - S[:, 0]) > 1e-9)
                print(f"     heights changed at fire-possible sites (forced-fire runs, levels): {int(m.sum())} cells, weight {100 * w[m].sum() / w[veg].sum():.1f} %")
        d = P - S
        print(f"  file vs blend: changed cells agb {int((np.abs(d[:, 1]) > 1e-9).sum())}, height {int((np.abs(d[:, 0]) > 1e-9).sum())}; squared change {np.sum(d ** 2):.0f} units ({np.sum(d ** 2) / d.size:.5f} in score units)")


if __name__ == "__main__":
    main()
