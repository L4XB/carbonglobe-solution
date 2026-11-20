#!/usr/bin/env python
"""Readout of a weighing design with any inclusion pattern (leave-one-out with or without a scored base, pairs, ...).

Model in squared units (one score quantum Q = 370.26): score(file f) = b - sum_i include[f][i] * g_i, with b the score of the base recipe (the file
that contains none of the factors; it need not be submitted) and g_i the gain of factor i. Every submitted score is rounded to 0.001, i.e. the true
score is uniform within +-0.5 quantum. Unknowns get wide Gaussian priors (gains: mean 0, sd --gain-sd; base: --base mean,sd in quanta when it is not
submitted or known from an earlier day); the posterior is the prior conditioned on the linear equations, averaged over the rounding intervals.

Design file (JSON): {"files": [...], "factors": [...], "include": [[0/1 per factor] per file], "steps": {factor: step}, optional "base": [mean, sd]
(quanta), optional "energies": {factor: full-step energy}}. Scores come from results/board_scores.json or from the command line (x 0.001).

    python scripts/readout_design.py work/logs/day49_design.json [s1 s2 ...] [--base 26,1] [--gain-sd 600]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
Q = 370.26


def posterior(X: np.ndarray, y: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian prior N(mu, diag(sd^2)) conditioned on X theta = y (exact). Returns the mean and covariance."""
    S = np.diag(sd ** 2); G = S @ X.T @ np.linalg.pinv(X @ S @ X.T)
    return mu + G @ (y - X @ mu), S - G @ X @ S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("design")
    ap.add_argument("scores", nargs="*", type=float, help="scores of the design's files in units of 0.001 (default: results/board_scores.json)")
    ap.add_argument("--base", default=None, help="prior of the base score 'mean,sd' in quanta (overrides the design file)")
    ap.add_argument("--gain-sd", type=float, default=600.0, help="prior sd of every gain, squared units")
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()
    d = json.load(open(ROOT / args.design)); files = d["files"]; fact = d["factors"]; inc = np.array(d["include"], float); steps = d.get("steps", {})
    k = len(fact); nf = len(files)
    if args.scores:
        s = np.array(args.scores, float)
    else:
        sc = json.load(open(ROOT / "results/board_scores.json")); s = np.array([sc.get(f) if sc.get(f) is not None else np.nan for f in files], float) * 1000.0
    print("design: " + "; ".join(f"{f.replace('.csv', '')} = base" + "".join(f" + {fact[i]}" for i in range(k) if inc[j, i]) for j, f in enumerate(files)))
    print("scores (x 0.001): " + ", ".join(f"{f.replace('.csv', '')} {v:.0f}" for f, v in zip(files, s)))
    if len(s) != nf or not np.isfinite(s).all():
        print("not all files are scored yet"); return
    base = args.base.split(",") if args.base else d.get("base")
    bmu, bsd = (float(base[0]) * Q, float(base[1]) * Q) if base else (float(np.max(s)) * Q, 20.0 * Q)
    X = np.hstack([np.ones((nf, 1)), -inc])                                   # columns b, g_1..g_k
    mu = np.array([bmu] + [0.0] * k); sd = np.array([bsd] + [args.gain_sd] * k)
    rng = np.random.default_rng(0); th = np.empty((args.n, k + 1))
    for t in range(args.n):
        y = (s + rng.uniform(-0.5, 0.5, size=nf)) * Q
        m, C = posterior(X, y, mu, sd)
        th[t] = rng.multivariate_normal(m, C + 1e-9 * np.eye(k + 1))
    E = dict(d.get("energies", {}))
    print(f"base (no factor): {th[:, 0].mean() / Q:.2f} +- {th[:, 0].std() / Q:.2f} (x 0.001)" + ("" if base else "  [no prior given]"))
    print("factor | gain (squared units; quanta) +- sd | P(gain > 0) | gain relative to the mean gain +- sd | kappa +- sd (if the energy is known)")
    g = th[:, 1:]; rel = g - g.mean(axis=1, keepdims=True)
    for i, f in enumerate(fact):
        line = f"   {f:4s} | {g[:, i].mean():+7.0f} ({g[:, i].mean() / Q:+.2f} q) +- {g[:, i].std():.0f} | {np.mean(g[:, i] > 0):.2f} | {rel[:, i].mean():+7.0f} +- {rel[:, i].std():.0f}"
        if f in E and E[f] > 0:
            st = float(steps.get(f, 1.0)); E1 = E[f] / st ** 2; kap = (g[:, i] / E1 + st ** 2) / (2 * st)
            line += f" | {kap.mean():+.2f} +- {kap.std():.2f} (energy {E[f]:.0f} at step {st}; best step {np.clip(kap.mean(), 0, 2):.2f} x the direction)"
        print(line)
    print("expected score (x 0.001) of the base + a set of factors (best first):")
    combos = []
    for r in range(k + 1):
        for c in itertools.combinations(range(k), r):
            tot = th[:, 0] - (g[:, list(c)].sum(axis=1) if c else 0.0)
            combos.append((tot.mean() / Q, tot.std() / Q, "+".join(fact[j] for j in c) or "base alone"))
    for m, sdv, nm in sorted(combos)[:8]:
        print(f"   {nm:24s} {m:6.2f} +- {sdv:.2f}")


if __name__ == "__main__":
    main()
