#!/usr/bin/env python
"""Yearly agb trajectories of ED on TRAINING cells (run with ED_SAVE_TRAJ=1) against the organisers' truth.
    python scripts/fire/traj_compare.py work/edv3/train_x1 [more roots] [--show 43296,44369]
Per root: rms of log(agb_k ED / truth) at k = 10, 20, 30, 40 over cells x ages (old / mid / young), correlation of the yearly
log ratios, and the yearly ratio rows of the cells named in --show (age 300).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import io  # noqa: E402

AGES = [1, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
M = 1.14114


def load_ed(root: Path, cell: int, age: int) -> np.ndarray | None:
    tr = []
    for sc in ("ssp126", "ssp585"):
        f = root / "traj" / f"{cell}_{sc}_a{age}.csv"
        if not f.exists():
            return None
        tr.append(pd.read_csv(f).agb_kgC.to_numpy())
    return np.mean(tr, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("roots", nargs="+"); ap.add_argument("--show", default="")
    args = ap.parse_args()
    show = [int(c) for c in args.show.split(",") if c]
    np.set_printoptions(linewidth=250, precision=0, suppress=True)
    for r in args.roots:
        root = Path(r)
        cells = sorted({int(f.name.split("_")[0]) for f in (root / "traj").glob("*_ssp126_a300.csv")})
        acc = {b: [] for b in ("old", "mid", "young")}; yl = []
        for c in cells:
            for a in AGES:
                ed = load_ed(root, c, a)
                if ed is None:
                    continue
                tr = np.asarray(io.open_train_y(a)[c, :, 0, 1], dtype=np.float64) * M
                band = "old" if a >= 150 else "mid" if a >= 50 else "young"
                if a == 1:
                    continue
                acc[band].append([np.log(max(ed[k], 0.02) / max(tr[k], 0.02)) - np.log(max(ed[0], 0.02) / max(tr[0], 0.02)) for k in (10, 20, 30, 40)])
                if a == 300:
                    yl.append((np.diff(np.log(np.maximum(ed, 0.02))), np.diff(np.log(np.maximum(tr, 0.02)))))
                    if c in show:
                        print(f"  cell {c} age 300 yearly ratio x100  truth: " + " ".join(f"{v:3.0f}" for v in 100 * tr[1:] / tr[:-1]))
                        print(f"  {'':29s}    ED: " + " ".join(f"{v:3.0f}" for v in 100 * ed[1:] / ed[:-1]))
        e = np.concatenate([x[0] for x in yl]); t = np.concatenate([x[1] for x in yl])
        txt = " | ".join(f"{b}: rms of log(ED / truth) change since year 0 at k=10/20/30/40 " + "/".join(f"{v:.3f}" for v in np.sqrt(np.mean(np.array(acc[b]) ** 2, axis=0)))
                         + " bias " + "/".join(f"{v:+.3f}" for v in np.mean(np.array(acc[b]), axis=0)) for b in acc if acc[b])
        print(f"{root.name}: {len(cells)} cells | corr of yearly log ratios (age 300) {np.corrcoef(e, t)[0, 1]:.3f}; truth burn years (< -0.06) {int((t < -0.06).sum())}, ED {int((e < -0.06).sum())}, both {int(((t < -0.06) & (e < -0.06)).sum())}")
        print("   " + txt)


if __name__ == "__main__":
    main()
