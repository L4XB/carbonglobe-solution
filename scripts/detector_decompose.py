#!/usr/bin/env python
"""Split the collapse detector's predicted old-age agb change into its VPD/heat and water-balance legs.

The position-free LightGBM detector (``scripts/mort_gbdt.py``) predicts the 40-year agb change of the old
seed ages from drought-index summaries, climate normals, soil and the year-0 pools. On the competition
forcing its losses arrive mostly through the evaporative-demand axis, which the training set supports
thinly (``notes/mortality.md`` Sec. 2-3). Neutralising one family of sequence features (every row gets
the median over the competition sites of that scenario) and predicting again isolates the part of the
predicted change that the family carries. The two legs become probe directions on old-age agb:
F = R + (p - p_without_family) on the vegetated old rows.

    <lightgbm venv>/bin/python scripts/detector_decompose.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from cgb import cache, io, shift  # noqa: E402
from cgb import mortality as mort  # noqa: E402
from mort_detector import M_ALL, site_features  # noqa: E402

OUT = ROOT / "work" / "models" / "mort_detector"
FAMILIES = {"vpd": ("vpd6", "vpd12", "tanom"), "wb": ("wb12", "wb6", "spi12")}


def main() -> None:
    spec = json.load(open(OUT / "gbdt_position_free_features.json"))
    names, cols, thr = spec["names"], spec["cols"], {k: tuple(v) for k, v in spec["thr"].items()}
    booster = lgb.Booster(model_file=str(OUT / "gbdt_position_free.txt"))
    y_mean = io.load_stats()["y_mean"].astype(np.float64)
    old = [io.SEED_AGES.index(a) for a in (150, 200, 250, 300, 350, 400, 450, 500)]
    frames = {s: io.open_test(s)[1].astype(np.float64) for s in io.SCENARIOS}
    floors = np.concatenate([fr.reshape(-1, 7) for fr in frames.values()]).min(axis=0)
    R = shift.read_predictions(ROOT / "results" / "latent_ref_y0_m114114.csv").astype(np.float64)
    keys = shift.template_keys()
    sc_k, site_k, age_k = keys["scenario"].to_numpy(), keys["site"].to_numpy(), keys["age"].to_numpy()
    out = {fam: R.copy() for fam in list(FAMILIES) + ["base"]}
    for sc in io.SCENARIOS:
        mt = mort.open_monthly(sc)
        xt = np.load(cache.CACHE_DIR / f"test_{sc}_annual.npy", mmap_mode="r")
        sx, sx_names = site_features(np.asarray(mt), np.asarray(xt[:, 0, list(cache.STATIC_DRIVERS)], dtype=np.float64), thr)
        raw0 = M_ALL * (frames[sc] - floors)
        n = raw0.shape[0]
        state0 = (np.transpose(raw0[:, old, :4], (1, 0, 2)) / y_mean[:4]).reshape(8 * n, 4)
        x = np.concatenate([np.repeat(sx[None], 8, axis=0).reshape(8 * n, -1), state0, np.repeat(np.arange(8), n)[:, None]], axis=1)
        veg = (np.transpose(raw0[:, old, 1]) > mort.BARE_AGB).reshape(-1)
        full_names = sx_names + ["y0_height", "y0_agb", "y0_soil", "y0_lai", "age_idx"]
        assert [full_names[c] for c in cols] == names, "feature layout changed"
        preds = {}
        for fam in ["base"] + list(FAMILIES):
            xx = x.copy()
            if fam != "base":
                for j, nm in enumerate(full_names):
                    if nm.split("_")[0] in FAMILIES[fam]:
                        xx[:, j] = np.median(sx[:, j])
            p = np.zeros(len(veg))
            p[veg] = booster.predict(xx[veg][:, cols])
            preds[fam] = p.reshape(8, n)
        for fam in FAMILIES:
            leg = preds["base"] - preds[fam]
            print(f"{sc}: {fam} leg of the predicted old agb change: mean {leg.mean():+.4f} sd {leg.std():.4f}; base mean {preds['base'].mean():+.4f}; corr(leg, base) {np.corrcoef(leg.ravel(), preds['base'].ravel())[0, 1]:+.2f}")
        m = sc_k == sc
        for i, a in enumerate((150, 200, 250, 300, 350, 400, 450, 500)):
            r = np.where(m & (age_k == a))[0]
            s = site_k[r]
            out["base"][r, 1] = R[r, 1] + preds["base"][i, s]
            for fam in FAMILIES:
                out[fam][r, 1] = R[r, 1] + preds["base"][i, s] - preds[fam][i, s]
    for fam in FAMILIES:
        shift.write_predictions(out[fam], ROOT / "results" / f"detprobe_old_agb_{fam}leg.csv")
    print("wrote results/detprobe_old_agb_{vpd,wb}leg.csv; nothing submitted")


if __name__ == "__main__":
    main()
