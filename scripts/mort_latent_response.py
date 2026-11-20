#!/usr/bin/env python
"""What does the latent model's predicted old-stand loss follow -- the deficit, or the heat?

``mort_heat_vs_dry.py`` shows ED collapses stands on a precipitation deficit and
not on heat alone. If the latent rollout has learned that, its predicted loss on
the scenario runs should track the deficit indices the way the holdout truth does,
and not the late-run warming that ssp585 adds. This script correlates, per site
(mean over the old seed ages of the relative 40-year agb change), the holdout
truth, the holdout prediction and the two scenario predictions with the site's
drought and heat summaries. Spearman, vegetated sites only.

    ./.venv/bin/python scripts/mort_latent_response.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cgb import cache, holdout, io, submission  # noqa: E402
from cgb import mortality as mort  # noqa: E402
from mort_trigger import build_indices, rankdata  # noqa: E402

M_ALL = 1.14114


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])


def summaries(monthly: np.ndarray) -> dict[str, np.ndarray]:
    ix = build_indices(monthly)
    fill = lambda x: np.where(np.isfinite(x), x, np.nanmin(x, axis=1, keepdims=True))  # noqa: E731
    return {
        "spi12 max": fill(ix["spi12"]).max(axis=1),
        "wb12 max": fill(ix["wb12"]).max(axis=1),
        "vpd6 max": fill(ix["vpd6"]).max(axis=1),
        "tanom max": ix["tanom"].max(axis=1),
        "tanom y31-40": ix["tanom"][:, 30:].mean(axis=1),
        "spi12 max y31-40": fill(ix["spi12"])[:, 30:].max(axis=1),
        "vpd6 max y31-40": fill(ix["vpd6"])[:, 30:].max(axis=1),
    }


def main() -> None:
    div = cache.scoring_divisors()
    y_mean = io.load_stats()["y_mean"].astype(np.float64)
    old = [j for j, a in enumerate(io.SEED_AGES) if a >= 150]
    traj = cache.open_traj()
    split = holdout.make_split(seed=42)
    rows = {}
    # holdout: truth and prediction, sites in split.public then split.private order
    sites = np.concatenate([split.public, split.private])
    pred = np.concatenate(
        [np.load(io.ROOT / "work" / "models" / "latent_v1" / f"pred_{f}.npy")[:, :, -1, 1] for f in ("public", "private")], axis=1
    ).astype(np.float64)
    st = np.asarray(traj[old][:, sites], dtype=np.float64)
    y0 = st[:, :, 0, 1] / div[1]
    y40 = st[:, :, 40, 1] / div[1]
    veg = st[:, :, 0, 1] > mort.BARE_AGB
    rel_true = np.where(veg, (y40 - y0) / np.where(veg, y0, 1), np.nan)
    rel_pred = np.where(veg, (pred[old] - y0) / np.where(veg, y0, 1), np.nan)
    ok = veg.sum(axis=0) >= 4
    s_hold = summaries(np.asarray(mort.open_monthly("train")[sites]))
    rows["holdout truth"] = (np.nanmean(rel_true, axis=0), s_hold, ok)
    rows["holdout latent"] = (np.nanmean(rel_pred, axis=0), s_hold, ok)
    # scenarios
    frame = pd.read_csv(io.ROOT / "results" / "latent_v1_m114.csv", dtype={"id": str}, keep_default_na=False)
    keys = submission.parse_ids(frame["id"])
    frames = {s: io.open_test(s)[1].astype(np.float64) for s in io.SCENARIOS}
    floors = np.concatenate([f.reshape(-1, 7) for f in frames.values()]).min(axis=0)
    for sc in io.SCENARIOS:
        raw0 = M_ALL * (frames[sc] - floors)
        y0t = np.transpose(raw0[:, old, 1]) / y_mean[1]
        sel = keys["scenario"].to_numpy() == sc
        k = keys.loc[sel]
        p = np.full((6171, 15), np.nan)
        p[k["site"].to_numpy(), [io.SEED_AGES.index(a) for a in k["age"].to_numpy()]] = frame.loc[sel, "agb"].to_numpy(dtype=np.float64)
        pt = np.transpose(p[:, old])
        vegt = np.transpose(raw0[:, old, 1]) > mort.BARE_AGB
        relt = np.where(vegt, (pt - y0t) / np.where(vegt, y0t, 1), np.nan)
        rows[f"{sc} latent"] = (np.nanmean(relt, axis=0), summaries(np.asarray(mort.open_monthly(sc))), vegt.sum(axis=0) >= 4)

    keys_s = list(s_hold.keys())
    print("== Spearman of the site's mean relative 40-year agb change (old ages) with its forcing summaries ==")
    print("   (negative = more stress, more loss)")
    print(f"  {'series':16s} {'sites':>6s} " + " ".join(f"{k:>16s}" for k in keys_s))
    for name, (v, s, ok) in rows.items():
        print(f"  {name:16s} {int(ok.sum()):6d} " + " ".join(f"{spearman(v[ok], s[k][ok]):16.3f}" for k in keys_s))
    # partial view: within the late band, is the predicted loss at the hottest sites larger than at the driest?
    print("\n== mean relative change of the tenth of sites with the largest late-run heat vs the largest late-run deficit ==")
    for name, (v, s, ok) in rows.items():
        hot = s["tanom y31-40"] >= np.nanquantile(s["tanom y31-40"][ok], 0.9)
        dry = s["spi12 max y31-40"] >= np.nanquantile(s["spi12 max y31-40"][ok], 0.9)
        print(f"  {name:16s} hottest 10% {np.nanmean(v[ok & hot]):+.3f}   driest 10% {np.nanmean(v[ok & dry]):+.3f}   all {np.nanmean(v[ok]):+.3f}")


if __name__ == "__main__":
    main()
