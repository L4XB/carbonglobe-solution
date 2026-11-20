#!/usr/bin/env python
"""How does the PUBLIC-fitted regional correction (scripts/spatial_correction.py, the "spatcorr stack") carry over to the private half?

On a synthetic board (scripts/sim/synth_board.py: truth = <truth base> + site noise, random public half of the sites, rounded public scores of
every scored file) the correction is fitted exactly as on the real board (CA_SCORES=<synthetic scores>), then B + s x (fitted - B) is scored
on BOTH halves against the synthetic truth for a range of scales s.

    python scripts/sim/synth_stack.py --boards work/sim/synth1,work/sim/synth2 [--ridge 1e-4] [--extra "--site-noise 0.5"] [--tag r1e4]
Prints per board the gain over the base (squared units, all-rows scale) on the public and on the private half per scale, the best scale of
each half and the transfer ratio private gain / public gain at 0.8.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cgb import shift  # noqa: E402

U = 370260.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", required=True); ap.add_argument("--ridge", default="1e-4"); ap.add_argument("--extra", default=""); ap.add_argument("--tag", default="x")
    ap.add_argument("--base", default="results/stack_20261010.csv"); ap.add_argument("--scales", default="0.4,0.6,0.7,0.8,0.9,1.0,1.2")
    args = ap.parse_args()
    B = shift.read_predictions(ROOT / args.base).astype(np.float64); scales = [float(x) for x in args.scales.split(",")]
    site = shift.template_keys().site.to_numpy(); tot = []
    for board in args.boards.split(","):
        out = f"results/synthsc_{Path(board).name}_{args.tag}.csv"
        cmd = [sys.executable, "scripts/spatial_correction.py", "--base", args.base, "--out", out, "--scale", "1.0"]
        if float(args.ridge) > 0:
            cmd += ["--ridge", args.ridge]
        cmd += args.extra.split()
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "CA_SCORES": board + "_scores.json"})
        if res.returncode != 0:
            print(f"{board}: spatial_correction failed: {res.stderr[-300:]}"); continue
        pg = [ln for ln in res.stdout.splitlines() if "predicted public gain" in ln]
        P = shift.read_predictions(ROOT / out).astype(np.float64); T = np.load(ROOT / (board + "_truth.npy")).astype(np.float64); pub = np.load(ROOT / (board + "_public.npy"))
        if pub.shape[0] != P.shape[0]:
            pub = pub[site]
        mse = lambda X, m: float(np.sum((X[m] - T[m]) ** 2)) / (m.sum() * 2)
        row = []
        for s in scales:
            X = B + s * (P - B); row.append(((mse(B, pub) - mse(X, pub)) * U, (mse(B, ~pub) - mse(X, ~pub)) * U))
        row = np.array(row); tot.append(row)
        i8 = scales.index(0.8) if 0.8 in scales else 0
        print(f"{Path(board).name}: base public {mse(B, pub) * 1000:.2f} private {mse(B, ~pub) * 1000:.2f}; energy of the fitted correction {np.sum((P - B) ** 2):.0f}; "
              + (pg[0].split(';')[-2].strip() if pg else ''))
        print("   scale   : " + " ".join(f"{s:7.2f}" for s in scales))
        print("   public  : " + " ".join(f"{x:7.0f}" for x in row[:, 0]) + f"   best scale {scales[int(np.argmax(row[:, 0]))]}")
        print("   private : " + " ".join(f"{x:7.0f}" for x in row[:, 1]) + f"   best scale {scales[int(np.argmax(row[:, 1]))]}; private / public at 0.8: {row[i8, 1] / row[i8, 0]:.2f}")
        os.remove(ROOT / out)
    if tot:
        m = np.mean(tot, axis=0)
        print(f"MEAN over {len(tot)} boards (ridge {args.ridge} {args.extra}):")
        print("   scale   : " + " ".join(f"{s:7.2f}" for s in scales))
        print("   public  : " + " ".join(f"{x:7.0f}" for x in m[:, 0]) + f"   best scale {scales[int(np.argmax(m[:, 0]))]}")
        print("   private : " + " ".join(f"{x:7.0f}" for x in m[:, 1]) + f"   best scale {scales[int(np.argmax(m[:, 1]))]}")


if __name__ == "__main__":
    main()
