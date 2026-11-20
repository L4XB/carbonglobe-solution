#!/usr/bin/env python
"""Write a submission candidate: the latent file with the GBDT blended into old-age agb.

``mort_gbdt.py`` measured, cross-fitted over the two holdout folds, that
``w_l * latent + w_g * gbdt`` on the old seed ages (>= 150) cuts the old-age agb
error by 11-15 % with weights that agree between folds. This applies the fold-
averaged weights to the competition predictions: the latent file's agb column on
vegetated old rows becomes ``y0 + w_l * (latent - y0) + w_g * gbdt_change``;
height, the young and middle ages, and bare rows are left exactly as they were.
Two files, one per detector variant; the position-free one is invariant to any
permutation of the forcing years by construction.

Against a reference that carries a band correction from the board (a ``lambda``
on the old band's change, a ``factor`` on its level; ``cgb.shift``), the blend
goes **first** and the correction on top, because the weights were fitted where
no shift exists and blending after the correction would let the detector's
undamped losses back in at their weight. ``--lam`` and ``--factor`` take the
same per-band dicts ``cgb.shift.damp`` / ``cgb.shift.scale`` take, as JSON, and
are applied to the blended file; ``--damp-then-blend`` builds the wrong order
for comparison only.

Nothing is submitted here. Validation is ``cgb.submission.validate``.

    ./.venv/bin/python scripts/mort_blend_candidate.py --latent results/latent_v1x2_m114.csv
    ./.venv/bin/python scripts/mort_blend_candidate.py --latent results/latent_v1x2_m114.csv \
        --lam '{"old": [1.0, 0.8]}' --factor '{"old": [0.85, 1.0]}' --tag day1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import io, shift, submission  # noqa: E402
from cgb import mortality as mort  # noqa: E402

M_ALL = 1.14114
OUT_DIR = io.ROOT / "work" / "models" / "mort_detector"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=Path, default=io.ROOT / "results" / "latent_v1x2_m114.csv")
    ap.add_argument("--lam", type=str, default=None, help='per-band damping of the change, JSON, e.g. \'{"old": [1.0, 0.8]}\' (height, agb)')
    ap.add_argument("--factor", type=str, default=None, help='per-band factor on the level, JSON, e.g. \'{"old": [0.85, 1.0]}\'')
    ap.add_argument("--tag", type=str, default=None, help="suffix for the output names when a correction is applied")
    ap.add_argument("--out-dir", type=Path, default=io.ROOT / "results")
    ap.add_argument("--damp-then-blend", action="store_true", help="also write the wrong order, for the comparison only")
    args = ap.parse_args()
    lam = json.loads(args.lam) if args.lam else None
    factor = json.loads(args.factor) if args.factor else None
    ages = keys_ages = None
    metrics = json.loads((io.ROOT / "work" / "metrics" / "mort_gbdt.json").read_text())
    y_mean = io.load_stats()["y_mean"].astype(np.float64)
    old = [j for j, a in enumerate(io.SEED_AGES) if a >= 150]
    frame = pd.read_csv(args.latent, dtype={"id": str}, keep_default_na=False)
    print(f"base {args.latent.name}: sha256 {hashlib.sha256(args.latent.read_bytes()).hexdigest()}")
    keys = submission.parse_ids(frame["id"])
    frames = {s: io.open_test(s)[1].astype(np.float64) for s in io.SCENARIOS}
    floors = np.concatenate([f.reshape(-1, 7) for f in frames.values()]).min(axis=0)

    ages = keys["age"].to_numpy()
    base_pers = shift.persistence()  # decoded persistence, scaled, template order

    def correct(values: np.ndarray) -> np.ndarray:
        out = values
        if lam:
            out = shift.damp(out, base_pers, lam, ages)
        if factor:
            out = shift.scale(out, factor, ages)
        return out

    for vname, tag in (("gbdt_position_free", "gbdtposfree"), ("gbdt_full", "gbdtfull")):
        w = np.mean([metrics["blend_weights"][vname][f] for f in ("public", "private")], axis=0)
        out = frame.copy()
        agb = out["agb"].to_numpy(dtype=np.float64).copy()
        changed = 0
        for sc in io.SCENARIOS:
            raw0 = M_ALL * (frames[sc] - floors)  # (6171, 15, 7) physical
            y0 = raw0[:, :, 1] / y_mean[1]  # scaled
            g = np.load(OUT_DIR / f"pred_change_{vname}_{sc}.npy").astype(np.float64)  # (8, 6171) scaled change
            sel = (keys["scenario"].to_numpy() == sc)
            for jj, j in enumerate(old):
                age = io.SEED_AGES[j]
                rows = np.flatnonzero(sel & (keys["age"].to_numpy() == age))
                sites = keys["site"].to_numpy()[rows]
                veg = raw0[sites, j, 1] > mort.BARE_AGB
                base = y0[sites, j]
                lat = agb[rows]
                new = base + w[0] * (lat - base) + w[1] * g[jj, sites]
                new = np.maximum(new, 0.0)
                agb[rows[veg]] = new[veg]
                changed += int(veg.sum())
        out["agb"] = agb
        suffix = f"_{tag}" + (f"_{args.tag}" if args.tag else "")
        if lam or factor:
            blended = out.loc[:, ["height", "agb"]].to_numpy(dtype=np.float64)
            corrected = correct(blended)
            out["height"], out["agb"] = corrected[:, 0], corrected[:, 1]
            if args.damp_then_blend:
                # the wrong order: correct the reference first, then blend the detector into it
                ref_c = correct(frame.loc[:, ["height", "agb"]].to_numpy(dtype=np.float64))
                wrong = frame.copy()
                wrong["height"], wrong["agb"] = ref_c[:, 0], ref_c[:, 1]
                agb_w = wrong["agb"].to_numpy(dtype=np.float64).copy()
                for sc in io.SCENARIOS:
                    raw0 = M_ALL * (frames[sc] - floors)
                    y0 = raw0[:, :, 1] / y_mean[1]
                    g = np.load(OUT_DIR / f"pred_change_{vname}_{sc}.npy").astype(np.float64)
                    sel = (keys["scenario"].to_numpy() == sc)
                    for jj, j in enumerate(old):
                        rows = np.flatnonzero(sel & (keys["age"].to_numpy() == io.SEED_AGES[j]))
                        sites = keys["site"].to_numpy()[rows]
                        veg = raw0[sites, j, 1] > mort.BARE_AGB
                        base = y0[sites, j]
                        new = np.maximum(base + w[0] * (agb_w[rows] - base) + w[1] * g[jj, sites], 0.0)
                        agb_w[rows[veg]] = new[veg]
                d = agb_w - out["agb"].to_numpy()
                print(f"  damp-then-blend differs from blend-then-damp on {int((d != 0).sum()):,} rows, mean |delta| {np.abs(d[d != 0]).mean():.4f}, "
                      f"agb mean {agb_w.mean():.4f} vs {out['agb'].mean():.4f}")
        report = submission.validate(out)
        path = args.out_dir / f"{args.latent.stem}{suffix}.csv"
        if not report.ok:
            print(report.render())
            raise SystemExit(f"{tag}: REJECTED")
        out.to_csv(path, index=False, float_format="%.6f")
        back = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
        rep2 = submission.validate(back)
        d = back["agb"].to_numpy() - frame["agb"].to_numpy()
        print(f"{path.name}: weights (latent, gbdt) = {np.round(w, 3)}; rows changed {changed:,}; "
              f"agb mean {frame['agb'].mean():.4f} -> {back['agb'].mean():.4f}; mean |delta| on changed rows {np.abs(d[d != 0]).mean():.4f}; "
              f"re-read validation {'OK' if rep2.ok else 'REJECTED'}")
        print(f"  sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
