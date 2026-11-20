#!/usr/bin/env python
"""Second-stage candidate: a scored base file plus MULTIPLES OF PROBE DIRECTIONS,   C = base + sum_j m_j (P_j - base).

Every probe file P_j of the direct batches (scripts/direct_batch.py) and of the 11-13 design is the base with one block changed, so a reading
"the block's level excess is delta" (probe step s) is applied exactly by m = delta / s (level and contrast probes alike; for a kappa probe at
step t on parts at step s use m = (kappa - s) / (t - s)). No tool chain, no rebuild: the result is an exact linear combination of files that are
on the board. scripts/readout_joint.py prints the m of every probe.

    python scripts/second_stage.py --base results/cand50_A.csv --out results/cand51_C.csv --steps cand50_A1.csv=0.111,cand50_A2.csv=-0.110,cand51_P2.csv=0.09
Heights stay within [0, max(35 m, base)], agb above the floor (a clipped cell is reported). The file is read back and compared with the array.
Prints the energy of every step (= m^2 E_j, the gain if the reading is exact), the cross terms between steps (overlapping blocks) and the totals.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

HCAP = 35.0 / 1.14114 / 10.90645
FL = 0.401412 / 3.39567


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--out", required=True); ap.add_argument("--steps", required=True)
    args = ap.parse_args()
    B = shift.read_predictions(ROOT / args.base).astype(np.float64)
    steps = [(it.split("=")[0], float(it.split("=")[1])) for it in args.steps.split(",") if it]
    D = []; tot = np.zeros_like(B)
    for f, m in steps:
        d = shift.read_predictions(ROOT / "results" / f).astype(np.float64) - B; D.append(m * d); tot += m * d
        print(f"   {f:18s} m {m:+.4f}  probe energy {np.sum(d * d):8.1f}  step energy {m * m * np.sum(d * d):7.2f}  (heights {m * m * np.sum(d[:, 0] ** 2):.2f}, agb {m * m * np.sum(d[:, 1] ** 2):.2f})")
    cross = sum(2.0 * float(np.sum(D[i] * D[j])) for i in range(len(D)) for j in range(i + 1, len(D)))
    C = B + tot
    C[:, 0] = np.clip(C[:, 0], 0.0, np.maximum(HCAP, B[:, 0])); C[:, 1] = np.maximum(C[:, 1], -FL)
    clipped = float(np.sum((C - B - tot) ** 2))
    shift.write_predictions(C, ROOT / args.out)
    R = shift.read_predictions(ROOT / args.out).astype(np.float64)
    dev = float(np.abs(R - C).max()); assert dev < 1e-6, f"file differs from the array after the write: {dev}"
    d = R - B
    print(f"{args.base} -> {args.out}: {len(steps)} steps, sum of step energies {sum(float(np.sum(x * x)) for x in D):.2f}, cross terms {cross:+.2f}, clipped {clipped:.3f}; "
          f"change against the base: heights {np.sum(d[:, 0] ** 2):.2f} on {int((np.abs(d[:, 0]) > 1e-9).sum())} cells, agb {np.sum(d[:, 1] ** 2):.2f} on {int((np.abs(d[:, 1]) > 1e-9).sum())} cells; "
          f"column means {B[:, 0].mean():.4f} {B[:, 1].mean():.4f} -> {R[:, 0].mean():.4f} {R[:, 1].mean():.4f}")


if __name__ == "__main__":
    main()
