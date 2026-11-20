#!/usr/bin/env python
"""Joint readout by FAMILIES of probes: one run of scripts/readout_joint.py per family, each with its own prior-scale scan.

    python scripts/readout_families.py --design work/logs/cand50_design.json:work/logs/day50_probes.json --design ...:...
        [--families "grp_a_080_pos-forest_50-500;grp_a_070_nevb-drift-shift_50-500"]   (';' between families, '+' inside one; the probes
                                                                                       that are not named form the last family)
        [--scores f=v,...] [--scan 0.15,0.3,0.6,1,1.7] [further arguments of readout_joint.py]

Why: the scan of readout_joint.py weighs ONE common scale of all prior widths by the evidence. That removes the cost of acting on noise, but
one real signal among many null readings is shrunk with them. Probes that test a separate hypothesis with a wide prior (the fire-regime
levels of 11-14) therefore get their own scan. Synthetic boards (11-12, scripts/sim/synth_two_stage.py --families, 8 probes): a 3 % level
error on the fire-possible forest block alone gives +64 units on the private half with the common scale, +94 with that probe as its own
family (oracle +106); null world -1 against +1; random errors at the prior widths +165 against +163. Every probe as its own family is
slightly worse in the null world (-6), so only the named probes are separated.
Prints every family's readout and, as the last line, the steps of all probes in design order for scripts/second_stage.sh.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", action="append", required=True)
    ap.add_argument("--families", default="")
    args, rest = ap.parse_known_args()
    names = []; files = {}
    for item in args.design:
        dp, _ = item.split(":"); des = json.load(open(ROOT / dp)); inc = des["include"]
        for j, f in enumerate(des["factors"]):
            row = [i for i, r in enumerate(inc) if r[j] == 1]; names.append(f); files[f] = des["files"][row[0]]
    fams = [f.split("+") for f in args.families.split(";") if f]
    for fam in fams:
        for n in fam:
            assert n in names, f"family member {n} is not a probe of these designs ({names})"
    named = {n for fam in fams for n in fam}; left = [n for n in names if n not in named]
    fams = fams + ([left] if left else [])
    steps = {}
    for fam in fams:
        cmd = [sys.executable, "scripts/readout_joint.py"] + sum((["--design", it] for it in args.design), []) + rest + (["--only", ",".join(fam)] if len(fams) > 1 else [])
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        txt = "\n".join(ln for ln in (out.stdout + out.stderr).splitlines() if "Warning" not in ln)
        print(f"--- family: {', '.join(fam)}"); print(txt)
        line = [ln for ln in txt.splitlines() if "steps for scripts/second_stage.sh" in ln]
        if out.returncode != 0 or not line:
            sys.exit(f"readout of the family {fam} failed")
        for a in line[-1].split(": ")[-1].split(","):
            f, m = a.split("="); steps[f] = m
    missing = [files[n] for n in names if files[n] not in steps]
    if missing:
        sys.exit(f"no step for {missing}")
    print("   steps for scripts/second_stage.sh (all families): " + ",".join(f"{files[n]}={steps[files[n]]}" for n in names))


if __name__ == "__main__":
    main()
