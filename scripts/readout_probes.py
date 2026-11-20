#!/usr/bin/env python
"""Readout of a night of MEASUREMENT files: one candidate A plus single exaggerated probes (work/logs/<prefix>_design.json, inclusion = identity).

Probe kinds (work/logs/<prefix>_probes.json, {factor: {...}}):
  contrast  scenario contrast of a block x (1 +- s) (scripts/apply_contrast.py): same formula as a level probe, delta = the contrast step the truth wants.
  level  heights of a block x (1 - s):   S_probe - S_A = E (1 - 2 delta / s),  E = s^2 S2 = the probe's energy, delta = level excess of the file
         (truth = file x (1 - delta)). Recommended base level factor 1 - delta.
  kappa  a group of builder parts at step t (in A: steps s_i, full-step energies e_i, rescaled to the probe's realised energy):
         S_probe - S_A = sum_i e_i (t - s_i)(t + s_i - 2 kappa)   for ONE common kappa of the group (truth = stack + kappa x deviation).
         Recommended: every part of the group at step kappa; expected gain sum_i e_i (s_i - kappa)^2.
Posterior by Monte Carlo over the exact rounding intervals (scores are rounded to 0.001 = 370.26 squared units), Gaussian priors from the file.

    python scripts/readout_probes.py work/logs/cand50_design.json work/logs/day50_probes.json [scores x 0.001 in file order] [--base 23.25,0.35]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
Q = 370.26


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("design"); ap.add_argument("probes")
    ap.add_argument("scores", nargs="*", type=float)
    ap.add_argument("--base", default="", help="prior of A's true score 'mean,sd' in quanta (default: its rounded score +- 0.29)")
    ap.add_argument("--n", type=int, default=2000000)
    args = ap.parse_args()
    des = json.load(open(ROOT / args.design)); pr = json.load(open(ROOT / args.probes))
    files = des["files"]; fac = des["factors"]; inc = np.array(des["include"], float)
    if args.scores:
        sc = np.array(args.scores, float)
    else:
        bs = json.load(open(ROOT / "results/board_scores.json")); sc = np.array([float(bs[f]) * 1000 for f in files])
    assert len(sc) == len(files)
    ia = int(np.where(inc.sum(1) == 0)[0][0])                      # the candidate file (no probe)
    rng = np.random.default_rng(0); n = args.n
    bm, bsd = (float(x) for x in args.base.split(",")) if args.base else (sc[ia], 0.45)
    SA = rng.normal(bm * Q, bsd * Q, n) if args.base else rng.uniform((sc[ia] - 0.5) * Q, (sc[ia] + 0.5) * Q, n)
    ok = np.round(SA / Q) == sc[ia]
    par = {}; lines = []
    for j, f in enumerate(fac):
        p = pr[f]; E = float(des["energies"][f]); rows = np.where(inc[:, j] == 1)[0]; assert len(rows) == 1, "single probes only"
        x = rng.normal(p["prior"][0], p["prior"][1], n)
        if p["type"] in ("level", "contrast"):
            # the PUBLIC energy of the block is known only to +- eps (random half of the sites; eps = sqrt(sum e_site^2) / sum e_site): error (s / 2) eps on delta
            Ep = E * (1.0 + float(p.get("eps", 0.0)) * rng.standard_normal(n))
            dS = Ep * (1.0 - 2.0 * x / p["s"])
        else:
            e = np.array([v[0] for v in p["parts"].values()], float); s = np.array([v[1] for v in p["parts"].values()], float); t = float(p["t"])
            e = e * E / float(np.sum(e * (t - s) ** 2))               # rescale to the realised probe energy
            dS = np.sum(e * (t - s) * (t + s)) - 2.0 * x * np.sum(e * (t - s))
            p["_e"] = e; p["_s"] = s
        ok &= np.round((SA + dS) / Q) == sc[rows[0]]
        par[f] = x
    acc = int(ok.sum())
    print(f"scores (x 0.001): " + ", ".join(f"{f} {v:.0f}" for f, v in zip(files, sc)) + f"; accepted samples {acc} of {n}")
    if acc < 50:
        print("too few samples: the priors exclude this outcome - widen them in the probes file or pass --base"); sys.exit(2)
    print(f"   candidate {files[ia]}: true score {SA[ok].mean() / Q:.2f} +- {SA[ok].std() / Q:.2f} (x 0.001)")
    steps = []; level = []; gain_tot = 0.0
    for f in fac:
        p = pr[f]; x = par[f][ok]; m, sd = float(x.mean()), float(x.std())
        if p["type"] == "contrast":
            # scenario-contrast probe (scripts/apply_contrast.py at +s): delta = step of the same tool that the truth wants (x (1 + delta) on ssp585 rows, x (1 - delta) on ssp126 rows)
            S2 = float(des["energies"][f]) / p["s"] ** 2
            print(f"   {f:11s} CONTRAST step {100 * m:+.2f} % +- {100 * sd:.2f} % (prior {100 * p['prior'][0]:+.1f} +- {100 * p['prior'][1]:.1f}); apply_contrast --s {m:.4f} on the block; expected gain {S2 * m * m:.0f} units")
            gain_tot += S2 * m * m; continue
        if p["type"] == "level":
            S2 = float(des["energies"][f]) / p["s"] ** 2; g = S2 * (x ** 2).mean() - S2 * sd ** 2
            print(f"   {f:11s} LEVEL excess {100 * m:+.2f} % +- {100 * sd:.2f} % (prior {100 * p['prior'][0]:+.1f} +- {100 * p['prior'][1]:.1f}); base level factor {1 - m:.4f}; expected gain {S2 * m * m:.0f} units")
            level.append(f"{p.get('block', '?')}={1 - m:.4f}"); gain_tot += S2 * m * m
        else:
            e, s = p["_e"], p["_s"]; kk = float(np.clip(m, 0.0, 1.3)); g = float(np.sum(e * ((s - m) ** 2 - (kk - m) ** 2)))
            print(f"   {f:11s} common KAPPA {m:+.2f} +- {sd:.2f} (prior {p['prior'][0]} +- {p['prior'][1]}); all parts -> step {kk:.2f}; expected gain {g:.0f} units "
                  + "(" + ", ".join(f"{nm} {si:.2f}: {ei * ((si - m) ** 2 - (kk - m) ** 2):+.0f}" for nm, ei, si in zip(p["parts"], e, s)) + ")")
            steps += [f"{nm}={kk:.2f}" for nm in p["parts"]]; gain_tot += g
            # assumption-free alternative: the best multiple m of the probe direction itself (steps s_i + m (t - s_i)); its gain m^2 E never
            # relies on the parts sharing one kappa - prefer it when the common-kappa move pulls single parts far against their current step
            t = float(p["t"]); L = float(np.sum(e * (t - s))); E = float(np.sum(e * (t - s) ** 2)); mm = float(np.sum(e * (t - s) * (m - s)) / E)
            sbar = float(np.sum(e * (t - s) * s) / L); sh = m - sbar
            print(f"               lever-weighted mean of the current steps {sbar:.2f} -> the group's average is off by {sh:+.2f}; uniform shift: steps "
                  + ", ".join(f"{nm}={si + sh:.2f}" for nm, si in zip(p["parts"], s)) + f"; gain {sh * sh * float(np.sum(e)):.0f} units if every part is off by the same amount")
            print(f"               direction multiple m = {mm:+.2f}: steps " + ", ".join(f"{nm}={si + mm * (t - si):.2f}" for nm, si in zip(p["parts"], s)) + f"; gain {mm * mm * E:.0f} units (assumption-free)")
    print(f"   expected gain of all corrections {gain_tot:.0f} units -> next candidate ~{(SA[ok].mean() - gain_tot) / Q:.2f} (x 0.001)")
    print("   --base-steps " + ",".join(steps)); print("   --base-level " + ",".join(level))


if __name__ == "__main__":
    main()
