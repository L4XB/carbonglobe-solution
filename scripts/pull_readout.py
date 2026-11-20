#!/usr/bin/env python
"""Readout of scored pull probes: lambda along the probe and the implied fraction of the raw pull.

A pull probe D is (part of) a raw pull P = clip(target - B) on a block, projected and scaled. If the residual along
the pull is kappa P, E[(T - B) D] = kappa E[P D] (plus whatever else D catches), so kappa ~ lambda E[D^2] / E[P D].
Reads lambda exactly like `spatial_correction.py` (stack-centred probes, rho E_all, mirror pairs if present).

    ./.venv/bin/python scripts/pull_readout.py spat_dir_u_oa_out.csv:work/dir_u_oa_out.npy ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402


def main() -> None:
    sc = json.load(open(ROOT / "results" / "board_scores.json"))
    rd = lambda p: shift.read_predictions(p).astype(np.float64)
    B = rd(ROOT / "results" / "stack_20261010.csv")
    C = sc["zeros.csv"]; ps = rd(ROOT / "results" / "persistence_shifted.csv")
    rho = 2 * (sc["persistence_shifted.csv"] - 2 * sc["persistence_s05.csv"] + C) / float(np.mean(ps ** 2))
    s_B = sc["stack_20261010.csv"] if "stack_20261010.csv" in sc else 0.050
    for arg in sys.argv[1:]:
        name, raw = arg.split(":") if ":" in arg else (arg, "")
        if name not in sc:
            print(f"{name:34s} not scored yet"); continue
        D = rd(ROOT / "results" / name) - B
        e2 = rho * float(np.mean(D * D))
        mir = name.replace(".csv", "_mirror.csv")
        if mir in sc:
            xr = (sc[name] - sc[mir]) / 4; e2 = (sc[name] + sc[mir] - 2 * s_B) / 2
        else:
            xr = (sc[name] - s_B - e2) / 2
        lam = -xr / e2
        sig = np.sqrt((0.00029 / 2) ** 2 * 2) / e2
        out = f"{name:34s} board {sc[name]:.3f}  lambda {lam:+.3f} (+-{sig:.3f} rounding)"
        if raw:
            P = np.load(ROOT / raw)
            kappa = lam * float(np.mean(D * D)) / float(np.mean(P * D))
            out += f"  raw-pull fraction kappa {kappa:+.2f}  (corr probe/pull {np.sum(P*D)/np.sqrt(np.sum(P*P)*np.sum(D*D)):.2f}, raw pull energy {np.mean(P*P):.5f} -> gain at kappa ~{kappa**2*np.mean(P*P):.5f})"
        print(out)


if __name__ == "__main__":
    main()
