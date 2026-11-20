#!/usr/bin/env python
"""Synthetic board: a pseudo-truth = <truth base file> + site-coherent noise of realistic energy, a random public half of the SITES, and the public
score (3 decimals) of every file that carries a real board score. Tests the estimators end to end (level fit, probe noise, re-absorption):

    python scripts/sim/synth_board.py --truth-base results/cand50p_A.csv --seed 1 --energy 7800 --out work/sim/synth1
writes work/sim/synth1_scores.json (use with CA_SCORES=...), work/sim/synth1_truth.npy, work/sim/synth1_public.npy (site mask).
Noise model: per site x scenario x column one relative effect for the mature rows and a correlated one (0.6) for the young rows, sd by site class
(fire-possible / cluster large, fire-never small), plus independent row noise; scaled to the requested total energy.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--truth-base", required=True); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--energy", type=float, default=7800.0)
    ap.add_argument("--out", required=True); ap.add_argument("--split", default="site", choices=("site", "sitescen", "row")); ap.add_argument("--row-share", type=float, default=0.15)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    keys = shift.template_keys(); site = keys.site.to_numpy(); age = keys.age.to_numpy(); scen = (keys.scenario.to_numpy() == "ssp585").astype(int)
    B = shift.read_predictions(ROOT / args.truth_base).astype(np.float64)
    mf = np.load(ROOT / "work/fire/max_factor_rule833.npy"); cl = np.load(ROOT / "work/collapse_cluster_labels.npy") == 0
    cls = np.where(cl, 3, np.where(mf >= 1e3, 2, np.where(mf >= 30, 1, 0)))[site]              # never, b, pos, cluster
    sd_rel = np.array([[0.03, 0.04], [0.05, 0.08], [0.12, 0.30], [0.10, 0.22]])                   # relative site effect (height, agb) by class
    FL = 0.401412 / 3.39567; val = np.column_stack([B[:, 0], B[:, 1] + FL])
    key = site * 2 + scen; nk = 6171 * 2; mature = age >= 50
    eps = np.zeros_like(B)
    for c in (0, 1):
        z1 = rng.standard_normal(nk); z2 = 0.6 * z1 + 0.8 * rng.standard_normal(nk)              # mature / young effects of a site x scenario
        zs = rng.standard_normal(6171)                                                             # part common to both scenarios of a site
        eff = np.where(mature, z1[key], z2[key]) * 0.8 + 0.6 * zs[site]
        eps[:, c] = val[:, c] * sd_rel[cls, c] * eff + val[:, c] * sd_rel[cls, c] * np.sqrt(args.row_share) * rng.standard_normal(len(B))
    eps[val <= 1e-6] = 0.0                                                                        # bare / floor cells are exact
    eps *= np.sqrt(args.energy / np.sum(eps ** 2))
    T = B + eps
    if args.split == "site":
        pub_site = rng.random(6171) < 0.5; pub = pub_site[site]
    elif args.split == "sitescen":
        pk = rng.random(nk) < 0.5; pub = pk[key]; pub_site = None
    else:
        pub = rng.random(len(B)) < 0.5; pub_site = None
    sc = json.load(open(ROOT / "results/board_scores.json")); out = {}
    for n, s in sc.items():
        f = ROOT / "results" / n
        if s is None or not f.exists():
            continue
        F = shift.read_predictions(f).astype(np.float64)
        out[n] = round(float(np.mean((F[pub] - T[pub]) ** 2)), 3)
    Path(ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / (args.out + "_scores.json"), "w"), indent=0); np.save(ROOT / (args.out + "_truth.npy"), T.astype(np.float32)); np.save(ROOT / (args.out + "_public.npy"), pub)
    e_cls = [float(np.sum(eps[cls == k] ** 2)) for k in range(4)]
    print(f"synthetic truth: noise energy {np.sum(eps ** 2):.0f} (never {e_cls[0]:.0f}, b {e_cls[1]:.0f}, fire-possible {e_cls[2]:.0f}, cluster {e_cls[3]:.0f}); public rows {int(pub.sum())} of {len(pub)}; "
          f"{len(out)} scores written; score of the truth base itself (public / private): {np.mean((B[pub] - T[pub]) ** 2):.5f} / {np.mean((B[~pub] - T[~pub]) ** 2):.5f}")


if __name__ == "__main__":
    main()
