#!/usr/bin/env python
"""Project a C-centred direction off the span of all scored board files (public metric, mild ridge), so that a probe
of it measures only what the past probes cannot say.
    python scripts/fire/project_span.py work/fire/dir_mer_gs_pr.npy [more] [--ridge 1e-3]
Writes <name>_perp.npy next to each input and prints the energy before / after.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "fire"))
import consistency_all as ca  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--ridge", type=float, default=1e-3)
    args = ap.parse_args()
    L = ca.load(); D = L["D"]; rho = L["rho"]; names = L["names"]
    nrm = rho / (D.shape[1] * 2)
    E = np.einsum("nrc,nrc->n", D, D) * nrm
    ok = E > 2e-5
    Dk = D[ok]
    G = np.einsum("nrc,mrc->nm", Dk, Dk).astype(np.float64) * nrm
    lam = args.ridge * np.trace(G) / len(G)
    for path in args.dirs:
        h = np.load(ROOT / path).astype(np.float64)
        x = np.einsum("nrc,rc->n", Dk, h) * nrm
        a = np.linalg.solve(G + lam * np.eye(len(G)), x)
        hp = h - np.einsum("n,nrc->rc", a, Dk)
        xp = np.einsum("nrc,rc->n", Dk, hp) * nrm
        out = Path(path).with_name(Path(path).stem + "_perp.npy"); np.save(ROOT / out, hp)
        print(f"{Path(path).name}: E_all {np.mean(h ** 2):.5f} -> {np.mean(hp ** 2):.5f} (kept {np.mean(hp ** 2) / np.mean(h ** 2):.2f}); "
              f"max |<h, D_p>| / sd before {np.max(np.abs(x)) / 0.00041:.1f}, after {np.max(np.abs(xp)) / 0.00041:.1f}; cos(h, h_perp) {np.sum(h * hp) / np.sqrt(np.sum(h ** 2) * np.sum(hp ** 2)):.3f}; rows touched {int((np.abs(hp).sum(1) > 1e-9).sum())} -> {out}")


if __name__ == "__main__":
    main()
