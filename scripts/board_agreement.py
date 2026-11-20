#!/usr/bin/env python
"""How well does a model file agree with everything the board has measured?

For every scored probe direction D_k the board gives lambda_k = E_pub[(T - B) D_k] / E_pub[D_k^2]. A model M implies
its own lambda_k(M) = E[(s M - B) D_k] / E[D_k^2] (s = 0.876 or 1). The energy-weighted correlation between the two
vectors says whether moving the stack toward M moves it the way the board measured; the least-squares step is the
fraction of the way to M the board supports. Ranks all given files (or the default model files).

    ./.venv/bin/python scripts/board_agreement.py [results/direct_x_m114.csv ...]
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cgb import shift  # noqa: E402
from local_eval import readouts  # noqa: E402


def main() -> None:
    files = sys.argv[1:] or sorted(glob.glob(str(ROOT / "results" / "latent_*_m114.csv")) + glob.glob(str(ROOT / "results" / "direct_*_m114.csv")))
    B, rho, Dm, y, fac, sc = readouts()
    e2 = rho * (Dm ** 2).mean(0) * fac
    lam = y / e2
    rows = []
    for f in files:
        try:
            M = shift.read_predictions(f).astype(np.float64)
        except Exception:
            continue
        for s in (0.876, 1.0):
            proj = rho * (Dm.T @ (s * M - B).ravel()) / Dm.shape[0] / e2
            c = np.sum(e2 * proj * lam) / np.sqrt(max(np.sum(e2 * proj ** 2) * np.sum(e2 * lam ** 2), 1e-30))
            g = np.sum(e2 * proj * lam) / max(np.sum(e2 * proj ** 2), 1e-30)
            rows.append((c, g, Path(f).name, s))
    rows.sort(reverse=True)
    for c, g, f, s in rows[:15]:
        print(f"  corr {c:+.3f}  step {g:+.3f}  {f} x{s}")


if __name__ == "__main__":
    main()
