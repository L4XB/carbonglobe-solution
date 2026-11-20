#!/usr/bin/env python
"""Screen pattern pulls toward every model file, per block, outside the collapse cluster.

For each model M and block b: direction = clip(0.876 M - B - block mean, +-25% of B) on the block's rows outside the
cluster. Reports its raw energy, the board support from the non-cluster probes (corr and least-squares step, as
`agree_split.py`) and its novel share against R + every scored and planned spatial probe. A rough expected gain if the
supported step is right is step^2 x novel energy (capped at step 1). Writes work/screen_pulls.csv.

    ./.venv/bin/python scripts/screen_pulls.py [--blocks ya,yh,ma,mh,oa,oh]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cgb import shift  # noqa: E402
from local_eval import readouts  # noqa: E402

BLOCKS = {"ya": ("young", 1), "yh": ("young", 0), "ma": ("mid", 1), "mh": ("mid", 0), "oa": ("old", 1), "oh": ("old", 0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blocks", default="ya,yh,ma,mh,oa,oh")
    ap.add_argument("--inside", action="store_true", help="pulls on the collapse cluster's rows, support from the cluster probes")
    ap.add_argument("--out", default="work/screen_pulls.csv")
    args = ap.parse_args()
    rd = lambda p: shift.read_predictions(p).astype(np.float64)
    B, rho, Dm, y, fac, sc = readouts()
    e2 = rho * (Dm ** 2).mean(0) * fac; lam = y / e2
    keys = shift.template_keys(); site = keys["site"].to_numpy(); masks = shift.band_masks(keys["age"].to_numpy())
    cl = np.load(ROOT / "work" / "cluster0_modes.npy")[:, 0] != 0; inside = cl[site]
    cell_site = np.repeat(site, 2)
    conc = np.array([np.sum(Dm[cl[cell_site], k] ** 2) / np.sum(Dm[:, k] ** 2) for k in range(Dm.shape[1])])
    g = conc > 0.5 if args.inside else conc <= 0.5
    # span of scored + planned spatial probes (and R)
    R = rd(ROOT / "results" / "latent_ref_y0_m114114.csv")
    centers = json.loads((ROOT / "work" / "spat_centers.json").read_text())

    def dev(n):
        c = centers.get(n, "R"); Cp = R if (c == "R" or c.endswith("latent_ref_y0_m114114.csv")) else rd(ROOT / c)
        return (rd(ROOT / "results" / n) - Cp).ravel()
    planned = []
    for f in sorted(glob.glob(str(ROOT / "scripts" / "submit_batch_day*.sh"))):
        planned += [Path(x).name for x in re.findall(r"submit (results/spat_[^ ]+\.csv)", Path(f).read_text())]
    names = sorted(set(n for n in sc if n.startswith(("spat_", "perp_spat_"))) | {p for p in planned if not p.endswith("_mirror.csv")})
    Q, _ = np.linalg.qr(np.stack([R.ravel()] + [dev(n) for n in names], 1))
    files = sorted(glob.glob(str(ROOT / "results" / "direct_*_m114.csv")) + glob.glob(str(ROOT / "results" / "latent_*_m114.csv")))
    out = []
    for f in files:
        M = 0.876 * rd(f)
        for bk in args.blocks.split(","):
            band, col = BLOCKS[bk]
            m = masks[band] & (inside if args.inside else ~inside)
            d = M[m, col] - B[m, col]
            D = np.zeros_like(B)
            D[m, col] = np.clip(d - d.mean(), -0.25 * np.maximum(B[m, col], 0.02), 0.25 * np.maximum(B[m, col], 0.02))
            x = D.ravel(); en = float(np.mean(x * x))
            if en < 1e-6:
                continue
            proj = rho * (Dm.T @ x) / Dm.shape[0] / e2
            num = np.sum(e2[g] * proj[g] * lam[g])
            corr = num / np.sqrt(max(np.sum(e2[g] * proj[g] ** 2) * np.sum(e2[g] * lam[g] ** 2), 1e-30))
            step = num / max(np.sum(e2[g] * proj[g] ** 2), 1e-30)
            perp = x - Q @ (Q.T @ x); nov = float(np.mean(perp * perp))
            gain = min(max(step, 0.0), 1.0) ** 2 * nov
            out.append((gain, Path(f).name, bk, en, corr, step, nov / en))
    out.sort(reverse=True)
    with open(ROOT / args.out, "w") as fh:
        fh.write("model,block,energy,support_corr,support_step,novel_share,expected_gain\n")
        for gain, n, bk, en, c, st, ns in out:
            fh.write(f"{n},{bk},{en:.6f},{c:.4f},{st:.4f},{ns:.4f},{gain:.7f}\n")
    for gain, n, bk, en, c, st, ns in out[:30]:
        print(f"{n:38s} {bk}  energy {en:.5f}  support {c:+.3f} step {st:+.3f}  novel {ns:.2f}  -> {gain:.6f}")


if __name__ == "__main__":
    main()
