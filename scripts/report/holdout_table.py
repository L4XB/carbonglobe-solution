#!/usr/bin/env python
"""Hold-out scores of the learned emulators, read from the result files that training wrote (numbers for the technical report).

    python scripts/report/holdout_table.py [work/models]

The hold-out is the site split of src/cgb/holdout.py: 6,171 training sites that were never fitted, in two halves ("public" and
"private", named after the leaderboard halves they imitate). Scores are mean squared errors of year-40 height and biomass in scored units.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    root = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "work/models")
    rows = []
    for f in sorted(root.glob("*/results.json")):
        d = json.loads(f.read_text()); tag = f.parent.name
        if "folds" in d:                                    # GRU latent rollout: per-fold table, first row = persistence, second = the model
            try:
                pub = {r["name"]: r["score"] for r in d["folds"]["public"]["rows"]}; prv = {r["name"]: r["score"] for r in d["folds"]["private"]["rows"]}
            except (KeyError, TypeError):
                continue
            if tag not in pub:
                continue
            rows.append((tag, "GRU", len(d.get("log", [])), pub[tag], prv[tag], pub.get("persistence"), prv.get("persistence")))
        elif "public" in d and isinstance(d["public"], dict) and "score" in d["public"]:      # LightGBM
            rows.append((tag, "LightGBM", None, d["public"]["score"], d["private"]["score"], d["public"].get("persistence"), d["private"].get("persistence")))
    print(f"{'model':32s} {'kind':9s} {'epochs':>6s} {'half 1':>8s} {'half 2':>8s} {'mean':>8s} | persistence")
    for tag, kind, ep, a, b, pa, pb in rows:
        pers = f"{(pa + pb) / 2:.3f}" if pa is not None and pb is not None else "-"
        print(f"{tag:32s} {kind:9s} {str(ep) if ep else '-':>6s} {a:8.4f} {b:8.4f} {(a + b) / 2:8.4f} | {pers}")
    print(f"{len(rows)} models. Quoted in the report: latent_noco2 (GRU, 8 epochs), latent_coress20pze24 (GRU, 24 epochs, random start years), direct_gbdt2 (LightGBM).")


if __name__ == "__main__":
    main()
