#!/usr/bin/env python
"""Readout of the weighing design of 11-12: K and the four files K + (all factors but one) of the factors R, H, C, B.

Model (squared units; one score quantum = 370.26): score(K) = b, score(F_i) = b - sum of the gains of the three factors in F_i. With G_i = score(K) -
score(F_i): sum of all gains = (G_1 + G_2 + G_3 + G_4) / 3 and gain_i = that sum - G_i. Every score is rounded to 0.001 (uniform +-0.5 quantum, sd 107
units), so each gain has an sd of 0.94 x 107 = 101 units; the builds are additive to ~20-40 units (notes 11-11 04:30 UTC).

For each factor the kappa of its direction follows from gain = (2 kappa s - s^2) E1 (E1 = full-step energy, s = the step applied in the file), and the
best step from kappa. A Monte Carlo over the rounding intervals gives the probability that a gain is positive and the expected score of each
combination of factors.

    python scripts/readout_design49.py [sK sF1 sF2 sF3 sF4]      (scores in units of 0.001; default: results/board_scores.json)
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
Q = 370.26
FILES = ("cand49_K.csv", "cand49_F1.csv", "cand49_F2.csv", "cand49_F3.csv", "cand49_F4.csv")
FACT = ("R", "H", "C", "B")                       # F_i leaves out FACT[i - 1]
STEP = {"R": 1.0, "H": 1.0, "C": 0.6, "B": 1.0}  # step at which the factor's direction is applied (C: 0.6 of the way to the chronosequence)
_DES = ROOT / "work/logs/day49_design.json"       # written by the batch script that actually ran: {"files": [...5], "factors": [...4], "steps": {...}}
if _DES.exists():
    _d = json.load(open(_DES)); FILES = tuple(_d["files"]); FACT = tuple(_d["factors"]); STEP = {k: float(v) for k, v in _d["steps"].items()}


def energies() -> dict[str, float]:
    from cgb import shift
    P = {n: shift.read_predictions(ROOT / "results" / n).astype(float) for n in FILES}
    K = P[FILES[0]]; S = sum(P[f] - K for f in FILES[1:]) / 3.0
    return {FACT[i]: float(np.sum((S - (P[FILES[i + 1]] - K)) ** 2)) for i in range(4)}


def main() -> None:
    if len(sys.argv) >= 6:
        s = np.array([float(x) for x in sys.argv[1:6]])
    else:
        sc = json.load(open(ROOT / "results/board_scores.json")); s = np.array([sc.get(f) if sc.get(f) is not None else np.nan for f in FILES], float) * 1000.0
    print("design: " + ", ".join(f"F{i + 1} = all but {f}" for i, f in enumerate(FACT)) + f"; steps {STEP}")
    print("scores (x 0.001): " + ", ".join(f"{f.replace('cand49_', '').replace('.csv', '')} {v:.0f}" for f, v in zip(FILES, s)))
    if not np.isfinite(s).all():
        print("not all five files are scored yet"); return
    try:
        E = energies()
    except Exception as e:  # noqa: BLE001
        print(f"(energies not available: {e}; using the dry-run values)"); E = {"R": 340.0, "H": 319.0, "C": 309.0, "B": 385.0}
    X = np.array([[1, 0, 0, 0, 0], [1, 0, -1, -1, -1], [1, -1, 0, -1, -1], [1, -1, -1, 0, -1], [1, -1, -1, -1, 0]], float)     # columns b, g_R, g_H, g_C, g_B
    rng = np.random.default_rng(0); n = 200000
    y = (s[None, :] + rng.uniform(-0.5, 0.5, size=(n, 5))) * Q                     # true scores inside their rounding intervals, squared units
    th = np.linalg.solve(X, y.T).T                                                  # (n, 5)
    g = th[:, 1:]; point = np.linalg.solve(X, s * Q)[1:]
    print(f"base K: {s[0]:.0f}; gains of the composites against K (quanta): " + ", ".join(f"F{i + 1} {s[0] - s[i + 1]:+.0f}" for i in range(4)))
    print("factor | gain (squared units, quanta) +- sd | P(gain > 0) | energy at the applied step | kappa +- sd | best step of the direction | gain at the best step")
    for i, f in enumerate(FACT):
        st = STEP[f]; E1 = E[f] / st ** 2
        kap = (g[:, i] / E1 + st ** 2) / (2 * st)
        kb = float(np.clip(kap.mean(), 0.0, 2.0))
        print(f"   {f}   | {point[i]:+7.0f} ({point[i] / Q:+.2f} q) +- {g[:, i].std():.0f} | {np.mean(g[:, i] > 0):.2f} | {E[f]:6.0f} | {kap.mean():+.2f} +- {kap.std():.2f} | {kb:.2f} (in units of the applied step: {kb / st:.2f}) | {kb ** 2 * E1:+.0f}")
    print("expected score (x 0.001) of K + a set of factors at the applied steps (mean over the rounding intervals; the best combination first):")
    combos = []
    for r in range(5):
        for c in itertools.combinations(range(4), r):
            tot = th[:, 0] - (g[:, list(c)].sum(axis=1) if c else 0.0)
            combos.append((tot.mean() / Q, tot.std() / Q, "+".join(FACT[k] for k in c) or "K alone"))
    for m, sd, nm in sorted(combos)[:8]:
        print(f"   {nm:10s} {m:6.2f} +- {sd:.2f}")
    tot = th[:, 0] - sum(np.clip((g[:, i] / (E[f] / STEP[f] ** 2) + STEP[f] ** 2) / (2 * STEP[f]), 0, 2).mean() ** 2 * (E[f] / STEP[f] ** 2) for i, f in enumerate(FACT))
    print(f"all four at their best steps: {tot.mean() / Q:.2f} (x 0.001)")


if __name__ == "__main__":
    main()
