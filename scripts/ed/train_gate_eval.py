#!/usr/bin/env python
"""Does the test pipeline's fit gate rescue ED where the spin-up is mismatched? Training-site check.

For each training root (output of train_site.py) and band, compares persistence and the hybrid ED (alpha 1, cap 2)
on the rows that pass the offset-log fit gate |log((y0 + eps) / (e0 + eps))| <= ln 2 (eps 0.1 m, 0.01 kgC, raw
units, as in the test readouts), and reports the share of rows kept and lambda = E[(y40 - y0)(h - y0)] / E[(h - y0)^2]
on the kept rows (the board's step-size reading with persistence as the centre).

    python scripts/ed/train_gate_eval.py work/edv3/train_c400 work/edv3/train_c400_s30
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ed")); sys.path.insert(0, str(ROOT / "scripts"))
from ed_transfer import DIV  # noqa: E402
from train_eval import BANDS, hybrid, load  # noqa: E402

EPS = {"h": 0.1, "a": 0.01}


def main() -> None:
    roots = [Path(r) for r in sys.argv[1:]]
    frames = {r.name: load(r) for r in roots}
    common = set.intersection(*[set(f.site) for f in frames.values()])
    print(f"{len(common)} common training sites")
    out = []
    for name, d in frames.items():
        d = d[d.site.isin(common)].reset_index(drop=True)
        band = np.full(len(d), "?", dtype=object)
        for b, (lo, hi) in BANDS.items():
            band[(d.age >= lo) & (d.age <= hi)] = b
        for col, cname in (("h", "height"), ("a", "agb")):
            div = DIV[cname]
            y0r, e0r = d[f"y0_{col}"].to_numpy(), d[f"e0_{col}"].to_numpy()
            gate = np.abs(np.log((y0r + EPS[col]) / (np.maximum(e0r, 0) + EPS[col]))) <= np.log(2.0)
            y0, y40 = y0r / div, d[f"y40_{col}"].to_numpy() / div
            h = hybrid(d[f"x40_{col}"].to_numpy() / div, e0r / div, y0, 2.0)
            for b in BANDS:
                m = band == b
                k = m & gate
                dt, dh = (y40 - y0)[k], (h - y0)[k]
                out.append(dict(root=name, col=cname, band=b, n=int(m.sum()), kept=float(k.sum() / max(m.sum(), 1)),
                                pers_all=np.mean((y40 - y0)[m] ** 2), hyb_all=np.mean((h - y40)[m] ** 2),
                                pers_kept=np.mean(dt ** 2) if k.any() else np.nan, hyb_kept=np.mean((dh - dt) ** 2) if k.any() else np.nan,
                                lam_kept=float(dt @ dh / max(dh @ dh, 1e-30)) if k.any() else np.nan,
                                gated_mse=np.mean(np.where(k, h, y0)[m] - y40[m]) ** 2 * 0 + np.mean((np.where(k, h, y0) - y40)[m] ** 2)))
    t = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print(t.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
