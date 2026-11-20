#!/usr/bin/env python
"""Board support of candidate directions, split into collapse-cluster probes and the rest.

For a direction X (an .npy array (N, 2) taken as given, or a prediction file minus the 10-10 stack) the support from a
probe group is the energy-weighted correlation between the direction's implied lambdas and the measured ones, and the
least-squares step along X those probes support (`board_agreement.py`, per group). A probe counts as a cluster probe
when more than half of its energy sits on the collapse cluster's sites. Probe directions built off the span give
meaningless steps here - pass the raw direction arrays (work/dir_*.npy).

    ./.venv/bin/python scripts/agree_split.py work/dir_cap_remove_out.npy results/direct_x_m114.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cgb import shift  # noqa: E402
from local_eval import readouts  # noqa: E402


def main() -> None:
    B, rho, Dm, y, fac, sc = readouts()
    e2 = rho * (Dm ** 2).mean(0) * fac; lam = y / e2
    site = shift.template_keys()["site"].to_numpy()
    cl = np.load(ROOT / "work" / "cluster0_modes.npy")[:, 0] != 0
    cell_site = np.repeat(site, 2)
    conc = np.array([np.sum(Dm[cl[cell_site], k] ** 2) / np.sum(Dm[:, k] ** 2) for k in range(Dm.shape[1])])
    groups = {"cluster": conc > 0.5, "other": conc <= 0.5}
    print("probes:", {k: int(v.sum()) for k, v in groups.items()})
    for f in sys.argv[1:]:
        X = np.load(f) if f.endswith(".npy") else shift.read_predictions(f).astype(np.float64) - B
        proj = rho * (Dm.T @ X.ravel()) / Dm.shape[0] / e2
        out = f"{Path(f).name:34s}"
        for gname, g in groups.items():
            c = np.sum(e2[g] * proj[g] * lam[g]) / np.sqrt(max(np.sum(e2[g] * proj[g] ** 2) * np.sum(e2[g] * lam[g] ** 2), 1e-30))
            st = np.sum(e2[g] * proj[g] * lam[g]) / max(np.sum(e2[g] * proj[g] ** 2), 1e-30)
            out += f" | {gname}: corr {c:+.3f} step {st:+.3f}"
        print(out)


if __name__ == "__main__":
    main()
