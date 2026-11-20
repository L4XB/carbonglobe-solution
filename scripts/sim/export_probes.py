#!/usr/bin/env python
"""Export every scored spatial/direction probe (D = file - centre), its readout inputs and the row->site map for the
public/private split simulation (`scripts/sim/split_sim.py`, runs on the home server).

    ./.venv/bin/python scripts/sim/export_probes.py --out work/sim_probes.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="work/sim_probes.npz")
    args = ap.parse_args()
    sc = json.load(open(ROOT / "results" / "board_scores.json"))
    rd = lambda p: shift.read_predictions(p).astype(np.float64)
    R = rd(ROOT / "results" / "latent_ref_y0_m114114.csv")
    B = rd(ROOT / "results" / "stack_20261010.csv")
    C = sc["zeros.csv"]; ps = rd(ROOT / "results" / "persistence_shifted.csv")
    rho = 2 * (sc["persistence_shifted.csv"] - 2 * sc["persistence_s05.csv"] + C) / float(np.mean(ps ** 2))
    centers = json.loads((ROOT / "work" / "spat_centers.json").read_text())
    names = sorted(n for n in sc if (n.startswith("spat_") or n.startswith("perp_spat_")) and not n.endswith("_mirror.csv"))
    D, s_k, cen_is_R, s_c = [], [], [], []
    for n in names:
        c = centers.get(n, "R"); isR = c == "R" or c.endswith("latent_ref_y0_m114114.csv")
        Cp = R if isR else rd(ROOT / c)
        D.append((rd(ROOT / "results" / n) - Cp).astype(np.float32))
        s_k.append(sc[n]); cen_is_R.append(isR); s_c.append(np.nan if isR else sc[Path(c).name])
    site = shift.template_keys()["site"].to_numpy().astype(np.int32)
    dev = np.zeros_like(B); dev[:, 1] = B[:, 1] - 0.888 * R[:, 1]
    np.savez_compressed(ROOT / args.out, D=np.stack(D), names=np.array(names), s_k=np.array(s_k), cen_is_R=np.array(cen_is_R),
                        s_c=np.array(s_c), rho=rho, site=site, B=B.astype(np.float32), R=R.astype(np.float32), dev=dev.astype(np.float32))
    print(f"exported {len(names)} probes to {args.out}")


if __name__ == "__main__":
    main()
