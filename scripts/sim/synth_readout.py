#!/usr/bin/env python
"""Run a probe design on synthetic boards: synthetic public scores of the design's files (3 decimals), the readout of scripts/readout_probes.py,
and - known only here - the exact value each probe should have read on the public half and what the private half would have wanted.
    python scripts/sim/synth_readout.py work/logs/cand50_design.json work/logs/day50_probes.json work/sim/synth1 [work/sim/synth2 ...]
For a probe file F = B + d:  m = <d, T - B> / <d, d> on a half;  level / contrast probes: delta = s x m."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift
des = json.load(open(ROOT / sys.argv[1])); pr = json.load(open(ROOT / sys.argv[2])); runs = sys.argv[3:]
files = des["files"]; fac = des["factors"]; inc = np.array(des["include"]); ia = int(np.where(inc.sum(1) == 0)[0][0])
P = [shift.read_predictions(ROOT / "results" / f).astype(np.float64) for f in files]; B = P[ia]
for run in runs:
    T = np.load(ROOT / (run + "_truth.npy")).astype(np.float64); pub = np.load(ROOT / (run + "_public.npy"))
    sc = [round(float(np.mean((p[pub] - T[pub]) ** 2)), 3) * 1000 for p in P]
    out = subprocess.run([sys.executable, "scripts/readout_probes.py", sys.argv[1], sys.argv[2]] + [f"{s:.0f}" for s in sc], cwd=ROOT, capture_output=True, text=True).stdout
    print(f"== {run}: scores {[int(s) for s in sc]}; true score of the candidate public {np.mean((B[pub] - T[pub]) ** 2) * 1000:.2f} / private {np.mean((B[~pub] - T[~pub]) ** 2) * 1000:.2f}")
    for j, f in enumerate(fac):
        i = int(np.where(inc[:, j] == 1)[0][0]); d = P[i] - B
        m_pub = float(np.sum((d * (T - B))[pub]) / np.sum((d * d)[pub])); m_prv = float(np.sum((d * (T - B))[~pub]) / np.sum((d * d)[~pub]))
        e_ratio = float(np.sum((d * d)[pub]) / (0.5 * np.sum(d * d)))
        line = [l for l in out.splitlines() if l.strip().startswith(f)]
        s = pr[f].get("s"); scale = s if s else 1.0
        print(f"   {f:12s} exact on the public half {100 * scale * m_pub:+.2f} %, private half {100 * scale * m_prv:+.2f} % (x {'s' if s else '1'}); public energy / half of all-rows energy {e_ratio:.3f} | readout: {line[0].strip()[len(f):].strip()[:95] if line else '?'}")
